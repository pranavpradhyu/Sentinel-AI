# SentinelAI — AI-Powered Network Threat Detection

An end-to-end intrusion detection platform. It reads network-flow records,
classifies each one into an attack family, and independently flags flows that
look *nothing like* normal traffic — the signature of a novel, never-before-seen
("zero-day") attack. Every verdict comes with a plain-language explanation of
**why** the model decided what it did.

Built as a single deployable web app with a mobile-installable (PWA) dashboard.

```
                 ┌─────────────────────────────────────────┐
   network flow →│  XGBoost classifier   → attack family    │→ verdict
                 │  Isolation Forest      → strangeness score│→ zero-day flag
                 │  pred_contribs (SHAP)  → per-feature "why" │→ explanation
                 └─────────────────────────────────────────┘
```

---

## Why several models instead of one

A single classifier can only recognise attacks that resemble its training data.
NSL-KDD's test set deliberately contains attack variants the training set never
shows — so a classifier alone misses many of them (this is a well-known,
honest limitation of the benchmark, not a bug).

SentinelAI pairs the classifier with **two independent anomaly engines**, each
trained only on benign traffic — they never see an attack, they just learn the
shape of "normal" and score how far a new flow falls outside it:

- **Isolation Forest** — a tree ensemble that isolates outliers by random splits.
- **Deep autoencoder** — a bottleneck neural net (96→32→8→32→96) trained to
  reconstruct benign flows. Reconstruction error is the anomaly score: benign
  traffic rebuilds cleanly, novel/attack traffic does not. (Implemented with
  scikit-learn's `MLPRegressor` so the whole stack stays light enough for a
  free-tier deploy — no TensorFlow or PyTorch.)

When the classifier says *Normal* but an anomaly engine says *this is deeply
abnormal*, the flow is surfaced as a **suspected zero-day** — the case a
supervised model is structurally unable to catch. You can switch between the two
engines (or run both) live in the dashboard. On the held-out test set the
Isolation Forest flags 60% of attacks and the autoencoder 54%; because they
model "normal" differently they catch *different* intrusions, and together they
reach **67%** — the argument for a layered, multi-paradigm design.

## Measured performance (NSL-KDD held-out test set, 22,544 flows)

| Metric | Value | Meaning |
|---|---|---|
| Attack ROC-AUC | **0.96** | benign-vs-attack separation |
| Detection rate | **0.67** | share of real attacks caught by the classifier |
| False-alarm rate | **0.03** | benign flows wrongly flagged |
| Anomaly catch rate | **0.60** | attacks the Isolation Forest flags independently |
| Overall accuracy | **0.78** | 5-class family accuracy |

The classifier is strongest on high-volume attacks (DoS ROC, Probe). Rare
families (R2L, U2R) are where the anomaly engine earns its place — precisely the
argument for the two-model design.

---

## Project structure

```
sentinel-ai/
├── model/
│   ├── train.py              # full training pipeline (run once)
│   └── artifacts/            # generated: models, metrics, demo flows
├── backend/
│   ├── app.py                # Flask API + serves the dashboard
│   └── pcap_to_flows.py      # scapy: raw .pcap → NSL-KDD flow features
├── frontend/
│   ├── index.html            # single-page SOC console
│   ├── manifest.json         # PWA manifest (installable on mobile)
│   ├── service-worker.js     # offline app-shell cache
│   └── static/{css,js,icons} # dashboard styles, logic, app icons
├── data/                     # NSL-KDD (download step below)
├── requirements.txt
├── Procfile / render.yaml    # one-click deploy config
└── README.md
```

## Run it locally

```bash
# 1. install
pip install -r requirements.txt

# 2. get the dataset (only needed to retrain — artifacts are already included)
cd data
curl -sSL -o KDDTrain.txt "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
curl -sSL -o KDDTest.txt  "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"
cd ..

# 3. (optional) retrain the models — takes ~1 minute
python model/train.py

# 4. launch
python backend/app.py           # dev server → http://localhost:5000
# or, production:
gunicorn backend.app:app --workers 1 --threads 4 --bind 0.0.0.0:5000
```

## Deploy to Render (free tier)

Push to GitHub, then either point Render at the repo (it reads `render.yaml`),
or create a Web Service manually with:

- **Build:** `pip install -r requirements.txt`
- **Start:** `gunicorn backend.app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`

The trained artifacts are committed, so no training runs on the server.

## Install on mobile

Open the deployed URL in a phone browser → **Add to Home Screen**. The manifest
and service worker make it launch full-screen like a native app and keep the
shell working offline. The UI is mobile-first and reflows to a single column.

---

## API

| Route | Method | Purpose |
|---|---|---|
| `/api/meta` | GET | classes, metrics, feature importance, confusion matrix, anomaly-engine stats |
| `/api/stream?n=&engine=&sensitivity=` | GET | classify N sampled flows (drives the live feed) |
| `/api/flow/<id>` | GET | one flow + full per-feature explanation |
| `/api/analyze` | POST | classify a single flow (JSON) — powers the Playground |
| `/api/upload` | POST | classify an uploaded CSV capture (NSL-KDD schema) |
| `/api/pcap` | POST | reassemble flows from a raw `.pcap` and classify them |

`engine` is `isolation_forest` \| `autoencoder` \| `both`; `sensitivity` is
`0.5`–`1.5`. On POST routes they go in the body / form fields.

```bash
curl -X POST localhost:5000/api/analyze -H "Content-Type: application/json" \
  -d '{"engine":"both","flow":{"protocol_type":"tcp","service":"http","flag":"SF","src_bytes":491,"count":2}}'
```

## Using the dashboard

- **Live flow feed** — real held-out flows stream through the engine; pause,
  step, and click any row to inspect it. A rolling threat gauge tracks the room.
- **Detection controls** — switch the anomaly engine (Isolation Forest /
  Autoencoder / Both) and drag a sensitivity slider; the whole app re-scores live.
- **Playground** — build a flow from scratch with sliders and dropdowns, or load
  a preset (*Normal web*, *Port scan*, *SYN flood*, *Password guess*), and watch
  the verdict, class probabilities, both anomaly scores, and the per-feature
  explanation update in real time. The fastest way to build intuition for *how*
  traffic features betray an attack.
- **Analyze a capture** — drop a CSV of flows, or a raw `.pcap`, for a scored report.

## From raw packets → flows (PCAP path)

`backend/pcap_to_flows.py` uses **scapy** (pure Python) to reassemble packets
into bidirectional flows and derive the NSL-KDD feature schema directly from
headers: basic features (duration, protocol, service, flag, bytes, land…) exactly,
time- and host-traffic features (count, srv_count, *error rates, same/diff-srv
rates, dst_host_*) over sliding windows, and content features set to 0 (they need
payload — the norm for encrypted traffic). Drop a `.pcap` in the **Analyze**
tab and it is parsed and scored server-side.

**Production path (CICFlowMeter).** For richer, modern features, generate flows
with [CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter), which emits ~80
statistical features per bidirectional flow from a live interface or a pcap:

```bash
# CICFlowMeter (Java) → flows.csv, then train a CIC-feature model and score it
./CICFlowMeter -f capture.pcap -c flows.csv
```

Because CICFlowMeter's schema differs from NSL-KDD's, pair it with a model
trained on CIC-IDS2017/2018 (see Research directions). The scapy path here is the
dependency-light, immediately-runnable alternative; the CICFlowMeter path is the
route to production-grade features.

## Real-world uses

- **SOC triage assistant** — rank a flood of network flows by threat so analysts
  look at the riskiest first, with a reason attached to each.
- **Edge / IoT monitoring** — the models are small (sklearn + XGBoost, a few MB)
  and run on modest hardware, unlike deep-learning IDS.
- **Security education** — the explainability view makes it a teaching tool for
  *how* traffic features betray an attack.
- **Batch forensics** — drop an exported capture in and get a scored report.

## Research directions

- **Adversarial robustness** — test evasion (feature-space perturbations, e.g.
  FGSM/boundary attacks) and adversarial training as a defence.
- **Deep anomaly detection** — a bottleneck autoencoder ships as a second engine
  (see above); next steps are a variational autoencoder (VAE) and a sequence
  model over flow windows, compared on zero-day recall.
- **Federated learning** — train across organisations without sharing raw
  traffic, so rare attacks seen at one site improve everyone's model.
- **Concept drift** — online updating as traffic patterns evolve over time.
- **Modern datasets** — retrain on CIC-IDS2017 / CSE-CIC-IDS2018 (encrypted,
  contemporary traffic) and on real packet captures via CICFlowMeter.
- **Graph learning** — model host-to-host communication as a graph and apply GNNs
  to catch lateral movement the flow-level view misses.

## Data & honest caveats

Trained on **NSL-KDD**, the standard IDS research benchmark. It is a cleaned
1999-era dataset; absolute numbers do not transfer to today's encrypted traffic.
The value here is the *architecture* — layered detection, explainability, and a
deployable serving path — which carries directly to modern data. The live feed
replays real held-out test flows through the models; it is a faithful
demonstration of the engine, not synthetic mock output.
