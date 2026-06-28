"""Run calibrated ExtraTrees and HGB-smaller experiments."""
import sys, json, os, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/research/code"))

from run_experiment import build_match_features, RepeatedStratifiedKFold, cross_validate, log_loss, SEED, N_SPLITS, N_REPEATS, FEATURE_COLS
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import sklearn

def run_calibrated_cv(exp_id, base_model, params, feature_cols=None, calibration_method="isotonic"):
    df = build_match_features()
    fc = feature_cols or FEATURE_COLS
    X = df[fc].fillna(0)
    y = df["label"]
    
    all_cat = ["home_confederation", "away_confederation"]
    cat_cols = [c for c in all_cat if c in fc]
    num_cols = [c for c in fc if c not in cat_cols]
    
    def make_pipe():
        transformers = []
        if cat_cols:
            transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols))
        if num_cols:
            transformers.append(("num", "passthrough", num_cols))
        enc = ColumnTransformer(transformers)
        if base_model == "et":
            clf = ExtraTreesClassifier(random_state=SEED, **params)
        elif base_model == "hgb":
            clf = HistGradientBoostingClassifier(random_state=SEED, **params)
        cal = CalibratedClassifierCV(clf, method=calibration_method, cv=3)
        return Pipeline([("enc", enc), ("cal", cal)])
    
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    
    def neg_logloss(est, X, y):
        proba = est.predict_proba(X)
        classes = list(est.classes_)
        return -log_loss(y, proba, labels=classes)
    
    pipe = make_pipe()
    start = time.time()
    scores = cross_validate(pipe, X, y,
                            cv=cv,
                            scoring={"neg_logloss": neg_logloss, "accuracy": "accuracy"},
                            return_train_score=False)
    elapsed = time.time() - start
    
    ll_mean = -scores["test_neg_logloss"].mean()
    ll_std = scores["test_neg_logloss"].std()
    acc_mean = scores["test_accuracy"].mean()
    acc_std = scores["test_accuracy"].std()
    
    out_dir = os.path.expanduser(f"~/research/artifacts/{exp_id}")
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = {
        "log_loss_mean": round(ll_mean, 5),
        "log_loss_std": round(ll_std, 5),
        "accuracy_mean": round(acc_mean, 4),
        "accuracy_std": round(acc_std, 4),
        "n_samples": int(len(y)),
        "n_features": int(len(fc)),
        "cv": f"{N_SPLITS}fold x {N_REPEATS}repeats",
        "calibration": calibration_method,
        "feature_importance_top10": {},
    }
    run_info = {
        "exp_id": exp_id,
        "model_type": f"{base_model}+calibrated({calibration_method})",
        "params": params,
        "feature_cols": fc,
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(f"{out_dir}/metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    with open(f"{out_dir}/run.json", "w") as f: json.dump(run_info, f, indent=2)
    
    print(f"[{exp_id}] {base_model}+{calibration_method}: log-loss={ll_mean:.4f}±{ll_std:.4f}, acc={acc_mean:.4f}, t={elapsed:.1f}s")

# ET calibrated (isotonic)
run_calibrated_cv(
    "et-calibrated", "et",
    {"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 8, "max_features": 0.5},
    calibration_method="isotonic"
)

# HGB with fewer features (Elo+rank+mv_diff only, smaller and regularised)
sys.path.insert(0, os.path.expanduser("~/research/code"))
FEW_FEATURES = ["elo_diff", "rank_diff", "mv_diff", "mv_top11_diff", "gk_mv_diff", "home_is_host", "away_is_host", "stage_enc", "home_confederation", "away_confederation"]
run_calibrated_cv(
    "hgb-smaller", "hgb",
    {"max_depth": 2, "min_samples_leaf": 15, "l2_regularization": 10.0, "max_iter": 50, "learning_rate": 0.05},
    feature_cols=FEW_FEATURES,
    calibration_method="sigmoid"
)

# ET with more aggressive leaf (leaf=12, depth=2)
run_calibrated_cv(
    "et-deeper", "et",
    {"n_estimators": 300, "max_depth": 2, "min_samples_leaf": 12, "max_features": 0.4},
    calibration_method="sigmoid"
)
