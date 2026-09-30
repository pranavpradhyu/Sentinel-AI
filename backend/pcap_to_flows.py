"""
pcap_to_flows.py - turn a raw packet capture into NSL-KDD style flow records
============================================================================
Reads a .pcap / .pcapng with scapy, reassembles packets into bidirectional
flows (5-tuple), and derives the NSL-KDD feature schema the models expect:

  * basic features       (duration, protocol, service, flag, bytes, land ...)
    are computed directly from packet headers.
  * time-traffic features (count, srv_count, *error_rate, same/diff_srv_rate)
    are derived over a 2-second sliding window of connection starts, following
    the original NSL-KDD derived-feature definitions.
  * host-traffic features (dst_host_*) are derived over the last 100
    connections to each destination host.
  * content features      (hot, num_failed_logins, logged_in ...) require
    application-layer payload and are set to 0 - exactly the situation for
    modern encrypted traffic, so the models are trained to tolerate it.

This is a dependency-light, fully runnable alternative to CICFlowMeter. See the
README for wiring CICFlowMeter's richer 80-feature output to a CIC-trained model.
"""

from collections import defaultdict, deque

# NSL-KDD service names keyed by well-known port
PORT_SERVICE = {
    20: "ftp_data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    37: "time", 43: "whois", 53: "domain", 79: "finger", 80: "http",
    88: "kerberos", 110: "pop_3", 111: "sunrpc", 113: "auth", 119: "nntp",
    123: "ntp_u", 135: "loc_srv", 139: "netbios_ssn", 143: "imap4",
    179: "bgp", 389: "ldap", 443: "http_443", 445: "microsoft_ds",
    512: "exec", 513: "login", 514: "shell", 515: "printer", 520: "efs",
    540: "uucp", 543: "klogin", 544: "kshell", 587: "smtp", 993: "imap4",
    995: "pop_3", 1080: "socks", 3306: "sql_net", 6000: "X11", 8001: "http_8001",
}


def _service(dport, sport):
    return PORT_SERVICE.get(dport) or PORT_SERVICE.get(sport) or "private"


def _tcp_flag(f):
    """Approximate the NSL-KDD connection-state flag from observed TCP flags."""
    syn, synack, fin, rst_o, rst_r, ack, data = (
        f["syn"], f["synack"], f["fin"], f["rst_o"], f["rst_r"], f["ack"], f["data"])
    if rst_o:
        return "RSTO"
    if rst_r:
        return "REJ" if syn and not data else "RSTR"
    if syn and synack and fin:
        return "SF"
    if syn and synack and (ack or data):
        return "SF"
    if syn and not synack:
        return "S0"
    if syn and fin and not synack:
        return "SH"
    return "OTH" if not syn else "S1"


def extract_flows(path, max_packets=200000):
    """Parse a capture file and return a list of NSL-KDD feature dicts."""
    from scapy.all import PcapReader, IP, TCP, UDP, ICMP  # lazy import

    flows = {}          # key -> aggregate
    order = []          # chronological flow keys as they first appear

    def key(a, pa, b, pb, proto):
        # canonical, direction-agnostic key
        return tuple(sorted([(a, pa), (b, pb)])) + (proto,)

    n = 0
    for pkt in PcapReader(path):
        if n >= max_packets:
            break
        if IP not in pkt:
            continue
        n += 1
        ip = pkt[IP]
        t = float(pkt.time)
        size = len(pkt)

        if TCP in pkt:
            proto, l4 = "tcp", pkt[TCP]
            sport, dport = int(l4.sport), int(l4.dport)
        elif UDP in pkt:
            proto, l4 = "udp", pkt[UDP]
            sport, dport = int(l4.sport), int(l4.dport)
        elif ICMP in pkt:
            proto, l4 = "icmp", pkt[ICMP]
            sport, dport = 0, 0
        else:
            continue

        k = key(ip.src, sport, ip.dst, dport, proto)
        f = flows.get(k)
        if f is None:
            f = flows[k] = {
                "src": ip.src, "dst": ip.dst, "sport": sport, "dport": dport,
                "proto": proto, "t0": t, "t1": t, "src_bytes": 0, "dst_bytes": 0,
                "urgent": 0, "wrong_fragment": 0, "orig": (ip.src, sport),
                "syn": False, "synack": False, "fin": False,
                "rst_o": False, "rst_r": False, "ack": False, "data": False,
            }
            order.append(k)
        f["t1"] = t
        # direction: does this packet come from the flow originator?
        from_orig = (ip.src, sport) == f["orig"]
        f["src_bytes" if from_orig else "dst_bytes"] += size
        if getattr(ip, "frag", 0) or (int(ip.flags) & 1):
            f["wrong_fragment"] += 1

        if proto == "tcp":
            fl = int(l4.flags)
            SYN, ACK, FIN, RST, URG = 0x02, 0x10, 0x01, 0x04, 0x20
            if fl & URG:
                f["urgent"] += 1
            if fl & SYN and fl & ACK:
                f["synack"] = True
            elif fl & SYN:
                f["syn"] = True
            if fl & FIN:
                f["fin"] = True
            if fl & ACK:
                f["ack"] = True
            if fl & RST:
                if from_orig:
                    f["rst_o"] = True
                else:
                    f["rst_r"] = True
            if len(l4.payload) > 0:
                f["data"] = True

    # ---- assemble base records in connection-start order
    conns = []
    for k in order:
        f = flows[k]
        service = _service(f["dport"], f["sport"]) if f["proto"] != "icmp" else "ecr_i"
        flag = _tcp_flag(f) if f["proto"] == "tcp" else "SF"
        conns.append({
            "t": f["t0"], "dst": f["dst"], "service": service, "flag": flag,
            "rec": {
                "duration": max(0, int(f["t1"] - f["t0"])),
                "protocol_type": f["proto"], "service": service, "flag": flag,
                "src_bytes": int(f["src_bytes"]), "dst_bytes": int(f["dst_bytes"]),
                "land": 1 if (f["src"] == f["dst"] and f["sport"] == f["dport"]) else 0,
                "wrong_fragment": int(f["wrong_fragment"]), "urgent": int(f["urgent"]),
            },
        })

    _add_time_traffic(conns)
    _add_host_traffic(conns)
    return [c["rec"] for c in conns]


def _rate(num, den):
    return round(num / den, 2) if den else 0.0


def _add_time_traffic(conns):
    """count / srv_count / serror / rerror / same-diff srv over a 2s window."""
    err_flags = {"S0", "S1", "S2", "S3"}       # SYN errors
    rej_flags = {"REJ", "RSTO", "RSTR"}        # rejections
    window = deque()
    for i, c in enumerate(conns):
        t = c["t"]
        window.append(i)
        while window and conns[window[0]]["t"] < t - 2.0:
            window.popleft()
        same_host = [conns[j] for j in window if conns[j]["dst"] == c["dst"]]
        same_srv = [conns[j] for j in window if conns[j]["service"] == c["service"]]
        cnt = len(same_host)
        scnt = len(same_srv)
        r = c["rec"]
        r["count"] = cnt
        r["srv_count"] = scnt
        r["serror_rate"] = _rate(sum(x["flag"] in err_flags for x in same_host), cnt)
        r["srv_serror_rate"] = _rate(sum(x["flag"] in err_flags for x in same_srv), scnt)
        r["rerror_rate"] = _rate(sum(x["flag"] in rej_flags for x in same_host), cnt)
        r["srv_rerror_rate"] = _rate(sum(x["flag"] in rej_flags for x in same_srv), scnt)
        r["same_srv_rate"] = _rate(sum(x["service"] == c["service"] for x in same_host), cnt)
        r["diff_srv_rate"] = _rate(sum(x["service"] != c["service"] for x in same_host), cnt)
        r["srv_diff_host_rate"] = _rate(sum(x["dst"] != c["dst"] for x in same_srv), scnt)
        # content features we cannot see from headers
        for z in ("hot", "num_failed_logins", "logged_in", "num_compromised",
                  "root_shell", "su_attempted", "num_root", "num_file_creations",
                  "num_shells", "num_access_files", "num_outbound_cmds",
                  "is_host_login", "is_guest_login"):
            r[z] = 0


def _add_host_traffic(conns):
    """dst_host_* features over the last 100 connections to each host."""
    err_flags = {"S0", "S1", "S2", "S3"}
    rej_flags = {"REJ", "RSTO", "RSTR"}
    hist = defaultdict(lambda: deque(maxlen=100))
    for c in conns:
        h = hist[c["dst"]]
        prev = list(h)
        r = c["rec"]
        cnt = len(prev)
        srv = [x for x in prev if x["service"] == c["service"]]
        r["dst_host_count"] = cnt
        r["dst_host_srv_count"] = len(srv)
        r["dst_host_same_srv_rate"] = _rate(len(srv), cnt)
        r["dst_host_diff_srv_rate"] = _rate(sum(x["service"] != c["service"] for x in prev), cnt)
        r["dst_host_same_src_port_rate"] = _rate(
            sum(x["rec"]["src_bytes"] == r["src_bytes"] for x in prev), cnt)
        r["dst_host_srv_diff_host_rate"] = _rate(
            sum(x["dst"] != c["dst"] for x in srv), len(srv))
        r["dst_host_serror_rate"] = _rate(sum(x["flag"] in err_flags for x in prev), cnt)
        r["dst_host_srv_serror_rate"] = _rate(sum(x["flag"] in err_flags for x in srv), len(srv))
        r["dst_host_rerror_rate"] = _rate(sum(x["flag"] in rej_flags for x in prev), cnt)
        r["dst_host_srv_rerror_rate"] = _rate(sum(x["flag"] in rej_flags for x in srv), len(srv))
        h.append(c)
