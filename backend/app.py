"""
SentinelAI - Flask backend
==========================
Serves the dashboard and exposes a small REST API around the trained models.

Endpoints
  GET  /                 dashboard (single-page app)
  GET  /api/meta         model classes, metrics, feature importance, matrix
  GET  /api/stream       classify N sampled live flows (monitor feed)
  GET  /api/flow/<id>    one demo flow + full per-feature explanation
  POST /api/analyze      classify a single flow supplied as JSON
  POST /api/upload       classify an uploaded CSV of flows (NSL-KDD schema)
"""

import io
import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `pcap_to_flows` imports under gunicorn too
ART = os.path.join(HERE, "..", "model", "artifacts")
FRONT = os.path.join(HERE, "..", "frontend")

app = Flask(__name__, static_folder=os.path.join(FRONT, "static"), static_url_path="/static")

# ------------------------------------------------------------- load models
PIPE = joblib.load(os.path.join(ART, "pipeline.joblib"))
ANOM = joblib.load(os.path.join(ART, "anomaly.joblib"))
AE = joblib.load(os.path.join(ART, "autoencoder.joblib"))
META = json.load(open(os.path.join(ART, "meta.json")))
DEMO = json.load(open(os.path.join(ART, "demo_flows.json")))

CLASSES = META["classes"]
FEATURES = META["features"]
CATEGORICAL = META["categorical"]
NUMERIC = META["numeric"]

_booster = PIPE.named_steps["clf"].get_booster()
_pre = PIPE.named_steps["pre"]
_ohe = _pre.named_transformers_["cat"]
_FEAT_NAMES = list(_ohe.get_feature_names_out(CATEGORICAL)) + NUMERIC


def _to_frame(records):
    df = pd.DataFrame(records)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURES]


def _anomaly_iso(df):
    """Isolation-Forest strangeness: (is_anomaly, 0..1 score) per row."""
    pre = ANOM["pipeline"].named_steps["pre"].transform(df)
    raw = ANOM["pipeline"].named_steps["iso"].score_samples(pre)
    thr = ANOM["threshold"]
    score = np.clip((thr - raw) / (abs(thr) + 1e-6) * 0.5 + 0.5, 0, 1)
    return (raw < thr), score


def _anomaly_ae(df):
    """Deep autoencoder strangeness from reconstruction error."""
    z = AE["pre"].transform(df)
    z = np.asarray(z.todense()) if hasattr(z, "todense") else np.asarray(z)
    err = np.mean((z - AE["model"].predict(z)) ** 2, axis=1)
    thr = AE["threshold"]
    score = np.clip(err / (thr * 2 + 1e-9), 0, 1)
    return (err > thr), score


def _classify(df, engine="isolation_forest", sensitivity=1.0):
    """engine: 'isolation_forest' | 'autoencoder' | 'both'.
    sensitivity scales how readily the anomaly layer fires (0.5..1.5)."""
    proba = PIPE.predict_proba(df)
    idx = proba.argmax(1)
    iso_flag, iso_s = _anomaly_iso(df)
    ae_flag, ae_s = _anomaly_ae(df)
    # sensitivity lowers the effective bar for calling something anomalous
    iso_flag = iso_flag | (iso_s > (1.0 - 0.5 * (sensitivity - 1) - 0.5))
    ae_flag = ae_flag | (ae_s > (1.0 - 0.5 * (sensitivity - 1) - 0.5))
    if engine == "autoencoder":
        anom_flag, anom_s = ae_flag, ae_s
    elif engine == "both":
        anom_flag, anom_s = (iso_flag | ae_flag), np.maximum(iso_s, ae_s)
    else:
        anom_flag, anom_s = iso_flag, iso_s

    out = []
    for i in range(len(df)):
        verdict = CLASSES[idx[i]]
        confidence = float(proba[i][idx[i]])
        threat = 1.0 - float(proba[i][0])
        zero_day = bool(verdict == "Normal" and anom_flag[i])
        out.append({
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "threat_score": round(threat, 3),
            "anomaly_score": round(float(anom_s[i]), 3),
            "iso_score": round(float(iso_s[i]), 3),
            "deep_score": round(float(ae_s[i]), 3),
            "zero_day": zero_day,
            "probabilities": {c: round(float(p), 3) for c, p in zip(CLASSES, proba[i])},
        })
    return out


def _explain(record, top=8):
    """Per-feature SHAP-style contributions using XGBoost pred_contribs."""
    df = _to_frame([record])
    x = _pre.transform(df)
    import xgboost as xgb
    dm = xgb.DMatrix(x, feature_names=_FEAT_NAMES)
    contribs = _booster.predict(dm, pred_contribs=True)  # (1, n_class, n_feat+1)
    proba = PIPE.predict_proba(df)[0]
    cls = int(proba.argmax())
    row = contribs[0][cls][:-1]                          # drop bias term
    # aggregate one-hot columns back to their source feature
    agg = {}
    for name, val in zip(_FEAT_NAMES, row):
        src = next((c for c in CATEGORICAL if name.startswith(c + "_")), name)
        agg[src] = agg.get(src, 0.0) + float(val)
    ranked = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:top]
    return [{
        "feature": f,
        "value": record.get(f, 0),
        "contribution": round(v, 4),
        "direction": "raises threat" if v > 0 else "lowers threat",
    } for f, v in ranked]


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return send_from_directory(FRONT, "index.html")


@app.route("/manifest.json")
def manifest():
    return send_from_directory(FRONT, "manifest.json")


@app.route("/service-worker.js")
def sw():
    return send_from_directory(FRONT, "service-worker.js", mimetype="application/javascript")


@app.route("/api/meta")
def api_meta():
    return jsonify(META)


def _opts(source):
    engine = source.get("engine", "isolation_forest")
    if engine not in ("isolation_forest", "autoencoder", "both"):
        engine = "isolation_forest"
    try:
        sens = float(source.get("sensitivity", 1.0))
    except (TypeError, ValueError):
        sens = 1.0
    return engine, max(0.5, min(1.5, sens))


@app.route("/api/stream")
def api_stream():
    n = min(int(request.args.get("n", 6)), 24)
    engine, sens = _opts(request.args)
    picks = np.random.default_rng().choice(len(DEMO), size=n, replace=False)
    records = [DEMO[int(i)]["flow"] for i in picks]
    results = _classify(_to_frame(records), engine, sens)
    feed = []
    for j, i in enumerate(picks):
        r = results[j]
        r["id"] = int(i)
        r["protocol"] = DEMO[int(i)]["flow"]["protocol_type"]
        r["service"] = DEMO[int(i)]["flow"]["service"]
        r["src_bytes"] = DEMO[int(i)]["flow"]["src_bytes"]
        feed.append(r)
    return jsonify({"feed": feed})


@app.route("/api/flow/<int:fid>")
def api_flow(fid):
    if fid < 0 or fid >= len(DEMO):
        return jsonify({"error": "No flow with that id."}), 404
    engine, sens = _opts(request.args)
    rec = DEMO[fid]["flow"]
    result = _classify(_to_frame([rec]), engine, sens)[0]
    result["explanation"] = _explain(rec)
    result["flow"] = rec
    result["ground_truth"] = DEMO[fid].get("truth")
    return jsonify(result)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(force=True, silent=True) or {}
    engine, sens = _opts(body)
    rec = body.get("flow", body)
    result = _classify(_to_frame([rec]), engine, sens)[0]
    result["explanation"] = _explain(rec)
    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "Attach a CSV file under the 'file' field."}), 400
    engine, sens = _opts(request.form)
    try:
        raw = f.read().decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(raw))
        if df.shape[1] >= 41 and not set(FEATURES).issubset(df.columns):
            df = pd.read_csv(io.StringIO(raw), header=None)
            names = FEATURES + (["label", "difficulty"][: df.shape[1] - len(FEATURES)])
            df = df.iloc[:, :len(names)]
            df.columns = names
    except Exception as e:
        return jsonify({"error": f"Could not read that CSV: {e}"}), 400
    return jsonify(_batch_report(df.head(5000), engine, sens))


@app.route("/api/pcap", methods=["POST"])
def api_pcap():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "Attach a .pcap or .pcapng capture under 'file'."}), 400
    engine, sens = _opts(request.form)
    tmp = os.path.join("/tmp", "sentinel_" + os.urandom(6).hex() + ".pcap")
    try:
        f.save(tmp)
        from pcap_to_flows import extract_flows
        flows = extract_flows(tmp)
    except ImportError:
        return jsonify({"error": "PCAP support needs scapy — run: pip install scapy"}), 500
    except Exception as e:
        return jsonify({"error": f"Could not parse that capture: {e}"}), 400
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    if not flows:
        return jsonify({"error": "No IP flows found in that capture."}), 400
    report = _batch_report(pd.DataFrame(flows[:5000]), engine, sens)
    report["source"] = "pcap"
    return jsonify(report)


def _batch_report(df, engine, sens):
    results = _classify(_to_frame(df.to_dict("records")), engine, sens)
    summary = {c: 0 for c in CLASSES}
    zero_days = 0
    for r in results:
        summary[r["verdict"]] += 1
        zero_days += int(r["zero_day"])
    rows = [{"row": i + 1, "verdict": r["verdict"], "confidence": r["confidence"],
             "threat_score": r["threat_score"], "zero_day": r["zero_day"]}
            for i, r in enumerate(results[:500])]
    return {
        "total": len(results),
        "summary": summary,
        "attacks": len(results) - summary["Normal"],
        "zero_days": zero_days,
        "rows": rows,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
