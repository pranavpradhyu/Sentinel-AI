"""
SentinelAI - model training pipeline
=====================================
Trains two complementary models on the NSL-KDD network-flow dataset:

  1. XGBoost multi-class classifier  -> names KNOWN attack families
     (Normal / DoS / Probe / R2L / U2R)
  2. Isolation Forest anomaly scorer -> flags NOVEL / zero-day behaviour
     that does not resemble the benign traffic it was trained on

The two are combined at serve time: the classifier gives a verdict, the
anomaly scorer gives an independent "how strange is this?" signal. A flow the
classifier calls Normal but the anomaly scorer finds highly abnormal is
surfaced as a suspected zero-day.

Outputs (written to model/artifacts/):
  pipeline.joblib        - preprocessing + XGBoost, ready to predict
  anomaly.joblib         - scaler + Isolation Forest
  meta.json              - label map, feature schema, metrics, feature importance
  demo_flows.json        - a pool of real test flows for the live demo feed
"""

import json
import os
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "artifacts")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- schema
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

# Map the 39 raw attack labels into the 5 canonical NSL-KDD families.
ATTACK_FAMILY = {
    "normal": "Normal",
    # Denial of Service
    "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS",
    "apache2": "DoS", "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    # Probe / surveillance
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # Remote to Local
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # User to Root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}
FAMILY_DESC = {
    "Normal": "Legitimate, expected traffic.",
    "DoS": "Denial of Service - floods a host to exhaust its resources.",
    "Probe": "Reconnaissance - scans ports and services to map targets.",
    "R2L": "Remote-to-Local - unauthorised access from a remote machine.",
    "U2R": "User-to-Root - privilege escalation to superuser.",
}


def load(split):
    df = pd.read_csv(os.path.join(DATA, f"KDD{split}.txt"), names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    df["family"] = df["label"].map(ATTACK_FAMILY).fillna("R2L")
    return df


def main():
    print("Loading NSL-KDD ...")
    train = load("Train")
    test = load("Test")
    print(f"  train {train.shape[0]:>6} flows | test {test.shape[0]:>6} flows")
    print("  family distribution (train):")
    print(train["family"].value_counts().to_string())

    features = [c for c in COLUMNS if c not in ("label", "difficulty")]
    numeric = [c for c in features if c not in CATEGORICAL]

    X_train, y_train = train[features], train["family"]
    X_test, y_test = test[features], test["family"]

    classes = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    y_train_idx = y_train.map({c: i for i, c in enumerate(classes)})
    y_test_idx = y_test.map({c: i for i, c in enumerate(classes)})

    # -------------------------------------------------- classifier
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
         ("num", StandardScaler(), numeric)],
        remainder="drop",
    )
    clf = XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.15,
        subsample=0.9, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(classes),
        eval_metric="mlogloss", tree_method="hist",
        n_jobs=-1, random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    # inverse-frequency weights so rare families (R2L, U2R) are not ignored
    counts = y_train_idx.value_counts().to_dict()
    n = len(y_train_idx)
    weights = y_train_idx.map(lambda c: n / (len(classes) * counts[c])).values

    print("\nTraining XGBoost classifier ...")
    t0 = time.time()
    pipe.fit(X_train, y_train_idx, clf__sample_weight=weights)
    print(f"  done in {time.time() - t0:.1f}s")

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    acc = accuracy_score(y_test_idx, pred)
    macro_f1 = f1_score(y_test_idx, pred, average="macro")
    print(f"  test accuracy  {acc:.4f}")
    print(f"  test macro-F1  {macro_f1:.4f}")
    print(classification_report(y_test_idx, pred, target_names=classes,
                                zero_division=0))

    # binary benign-vs-attack view (what a SOC actually cares about)
    bin_true = (y_test_idx != 0).astype(int)
    bin_score = 1 - proba[:, 0]
    roc = roc_auc_score(bin_true, bin_score)
    bin_pred = (pred != 0).astype(int)
    detection_rate = ((bin_pred == 1) & (bin_true == 1)).sum() / max(bin_true.sum(), 1)
    false_alarm = ((bin_pred == 1) & (bin_true == 0)).sum() / max((bin_true == 0).sum(), 1)
    print(f"  attack ROC-AUC {roc:.4f} | detection {detection_rate:.4f} "
          f"| false-alarm {false_alarm:.4f}")

    cm = confusion_matrix(y_test_idx, pred).tolist()

    # feature importance mapped back to readable names
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL))
    feat_names = cat_names + numeric
    importances = pipe.named_steps["clf"].feature_importances_
    # collapse one-hot columns back to their source feature
    agg = {}
    for name, imp in zip(feat_names, importances):
        base = name.split("_")[0] + "_" + name.split("_")[1] if name in cat_names else name
        src = next((c for c in CATEGORICAL if name.startswith(c + "_")), name)
        agg[src] = agg.get(src, 0.0) + float(imp)
    top_features = sorted(agg.items(), key=lambda x: -x[1])[:15]

    # -------------------------------------------------- anomaly detector
    print("\nTraining Isolation Forest on benign traffic ...")
    benign = X_train[y_train == "Normal"]
    anom_pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
         ("num", StandardScaler(), numeric)],
        remainder="drop",
    )
    iso = IsolationForest(n_estimators=200, contamination=0.02,
                          max_samples="auto", n_jobs=-1, random_state=42)
    anom = Pipeline([("pre", anom_pre), ("iso", iso)])
    anom.fit(benign)

    # calibrate: how often does the anomaly scorer fire on real attacks?
    raw = anom.named_steps["iso"].score_samples(
        anom.named_steps["pre"].transform(X_test))
    thr = np.percentile(
        anom.named_steps["iso"].score_samples(
            anom.named_steps["pre"].transform(benign)), 2)
    flagged = raw < thr
    caught = flagged[(y_test != "Normal").values].mean()
    benign_flagged = flagged[(y_test == "Normal").values].mean()
    print(f"  anomaly threshold {thr:.4f}")
    print(f"  flags {caught:.2%} of real attacks | "
          f"{benign_flagged:.2%} of benign as anomalous")

    # -------------------------------------------------- deep autoencoder
    # A bottleneck MLP trained to reconstruct benign traffic. Reconstruction
    # error (MSE) is the "deep" anomaly score: benign flows rebuild cleanly,
    # novel/attack flows do not. Implemented with sklearn so the whole stack
    # stays light enough for a free-tier deploy (no TensorFlow / PyTorch).
    print("\nTraining deep autoencoder (bottleneck MLP) on benign traffic ...")
    ae_pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
         ("num", StandardScaler(), numeric)],
        remainder="drop",
    )
    Xb = ae_pre.fit_transform(benign)
    Xb = np.asarray(Xb.todense()) if hasattr(Xb, "todense") else np.asarray(Xb)
    ae = MLPRegressor(
        hidden_layer_sizes=(96, 32, 8, 32, 96),   # symmetric encoder/decoder
        activation="relu", solver="adam", alpha=1e-4,
        batch_size=256, learning_rate_init=1e-3, max_iter=60,
        early_stopping=True, n_iter_no_change=6, random_state=42,
    )
    t0 = time.time()
    ae.fit(Xb, Xb)                                 # learn identity through bottleneck
    print(f"  done in {time.time() - t0:.1f}s ({ae.n_iter_} epochs)")

    def recon_error(pre, model, frame):
        z = pre.transform(frame)
        z = np.asarray(z.todense()) if hasattr(z, "todense") else np.asarray(z)
        return np.mean((z - model.predict(z)) ** 2, axis=1)

    benign_err = recon_error(ae_pre, ae, benign)
    ae_thr = float(np.percentile(benign_err, 98))  # 2% benign false-positive budget
    test_err = recon_error(ae_pre, ae, X_test)
    ae_flag = test_err > ae_thr
    ae_caught = ae_flag[(y_test != "Normal").values].mean()
    ae_benign_flagged = ae_flag[(y_test == "Normal").values].mean()
    # how much does the deep model add beyond the isolation forest?
    both = flagged | ae_flag
    union_caught = both[(y_test != "Normal").values].mean()
    print(f"  reconstruction threshold {ae_thr:.4f}")
    print(f"  flags {ae_caught:.2%} of real attacks | "
          f"{ae_benign_flagged:.2%} of benign as anomalous")
    print(f"  isolation-forest OR autoencoder catches {union_caught:.2%} of attacks")

    # -------------------------------------------------- demo flow pool
    rng = np.random.default_rng(7)
    pool = []
    demo_src = test.reset_index(drop=True)
    idx = rng.choice(len(demo_src), size=600, replace=False)
    for i in idx:
        row = demo_src.iloc[int(i)]
        flow = {k: (int(row[k]) if isinstance(row[k], (np.integer,))
                    else float(row[k]) if isinstance(row[k], (np.floating,))
                    else row[k]) for k in features}
        pool.append({"flow": flow, "truth": row["family"], "raw_label": row["label"]})

    # -------------------------------------------------- persist
    joblib.dump(pipe, os.path.join(OUT, "pipeline.joblib"))
    joblib.dump({"pipeline": anom, "threshold": float(thr)},
                os.path.join(OUT, "anomaly.joblib"))
    joblib.dump({"pre": ae_pre, "model": ae, "threshold": ae_thr},
                os.path.join(OUT, "autoencoder.joblib"))

    meta = {
        "classes": classes,
        "family_desc": FAMILY_DESC,
        "features": features,
        "categorical": CATEGORICAL,
        "numeric": numeric,
        "metrics": {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "attack_roc_auc": round(float(roc), 4),
            "detection_rate": round(float(detection_rate), 4),
            "false_alarm_rate": round(float(false_alarm), 4),
            "anomaly_catch_rate": round(float(caught), 4),
            "deep_anomaly_catch_rate": round(float(ae_caught), 4),
            "combined_anomaly_catch_rate": round(float(union_caught), 4),
            "train_size": int(train.shape[0]),
            "test_size": int(test.shape[0]),
        },
        "anomaly_engines": {
            "isolation_forest": {
                "catch_rate": round(float(caught), 4),
                "benign_flag_rate": round(float(benign_flagged), 4),
                "kind": "tree ensemble",
            },
            "autoencoder": {
                "catch_rate": round(float(ae_caught), 4),
                "benign_flag_rate": round(float(ae_benign_flagged), 4),
                "kind": "bottleneck neural net (96-32-8-32-96)",
                "epochs": int(ae.n_iter_),
            },
        },
        "confusion_matrix": cm,
        "top_features": [{"feature": f, "importance": round(v, 4)} for f, v in top_features],
        "trained_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(OUT, "demo_flows.json"), "w") as f:
        json.dump(pool, f)

    print("\nArtifacts written to", OUT)
    for fn in sorted(os.listdir(OUT)):
        size = os.path.getsize(os.path.join(OUT, fn)) / 1024
        print(f"  {fn:<20} {size:8.1f} KB")


if __name__ == "__main__":
    main()
