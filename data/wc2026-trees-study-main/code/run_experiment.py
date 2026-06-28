"""
Run one tree ensemble experiment and emit metrics.json + run.json to ~/research/artifacts/<exp_id>/
Usage: python3 run_experiment.py <exp_id> <model_type> [options as JSON string]
"""
import sys, json, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import log_loss, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sklearn

sys.path.insert(0, os.path.expanduser("~/research/code"))
from features import build_match_features

SEED = 0
N_SPLITS = 5
N_REPEATS = 10

LABEL_ORDER = ["A", "D", "H"]

FEATURE_COLS = [
    "elo_diff", "rank_diff",
    "home_elo_rating", "away_elo_rating",
    "home_fifa_ranking_pre_tournament", "away_fifa_ranking_pre_tournament",
    "mv_diff", "mv_top11_diff", "caps_diff", "goals_diff", "age_diff",
    "gk_mv_diff", "att_goals_diff",
    "home_squad_total_mv", "away_squad_total_mv",
    "home_squad_mean_caps", "away_squad_mean_caps",
    "home_squad_mean_age", "away_squad_mean_age",
    "home_squad_total_goals", "away_squad_total_goals",
    "home_squad_mean_height", "away_squad_mean_height",
    "home_gk_mv", "away_gk_mv",
    "home_att_goals", "away_att_goals",
    "home_is_host", "away_is_host",
    "stage_enc",
    "capacity", "elevation_meters",
    "home_confederation", "away_confederation",
]

def build_pipeline(model_type, params, feature_cols=None):
    fc = feature_cols or FEATURE_COLS
    all_cat = ["home_confederation", "away_confederation"]
    cat_cols = [c for c in all_cat if c in fc]
    num_cols = [c for c in fc if c not in cat_cols]

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    transformers = []
    if cat_cols:
        transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    enc = ColumnTransformer(transformers)

    if model_type == "rf":
        clf = RandomForestClassifier(random_state=SEED, **params)
    elif model_type == "et":
        clf = ExtraTreesClassifier(random_state=SEED, **params)
    elif model_type == "hgb":
        clf = HistGradientBoostingClassifier(random_state=SEED, **params)
    elif model_type == "elo_logistic":
        clf = LogisticRegression(max_iter=1000, random_state=SEED, **params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("enc", enc), ("clf", clf)])

def run_cv(model_type, params, feature_cols=None):
    df = build_match_features()
    fc = feature_cols or FEATURE_COLS
    X = df[fc].fillna(0)
    y = df["label"]

    pipe = build_pipeline(model_type, params, feature_cols=fc)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    
    def neg_logloss(est, X, y):
        proba = est.predict_proba(X)
        classes = list(est.classes_)
        return -log_loss(y, proba, labels=classes)
    
    scores = cross_validate(pipe, X, y,
                            cv=cv,
                            scoring={"neg_logloss": neg_logloss, "accuracy": "accuracy"},
                            return_train_score=False)
    
    ll_mean = -scores["test_neg_logloss"].mean()
    ll_std = scores["test_neg_logloss"].std()
    acc_mean = scores["test_accuracy"].mean()
    acc_std = scores["test_accuracy"].std()
    
    return ll_mean, ll_std, acc_mean, acc_std, X, y, pipe

def get_feature_importance(model_type, params, fc):
    df = build_match_features()
    X = df[fc].fillna(0)
    y = df["label"]
    from sklearn.compose import ColumnTransformer
    all_cat = ["home_confederation", "away_confederation"]
    cat_cols = [c for c in all_cat if c in fc]
    num_cols = [c for c in fc if c not in cat_cols]
    from sklearn.preprocessing import OrdinalEncoder
    transformers = []
    if cat_cols:
        transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    enc = ColumnTransformer(transformers)
    from sklearn.pipeline import Pipeline
    if model_type == "rf":
        clf = RandomForestClassifier(random_state=SEED, **params)
    elif model_type == "et":
        clf = ExtraTreesClassifier(random_state=SEED, **params)
    elif model_type == "hgb":
        clf = HistGradientBoostingClassifier(random_state=SEED, **params)
    else:
        return {}
    
    pipe = Pipeline([("enc", enc), ("clf", clf)])
    Xt = enc.fit_transform(X)
    col_names = list(cat_cols) + list(num_cols)
    clf.fit(Xt, y)
    if hasattr(clf, "feature_importances_"):
        imp = dict(sorted(zip(col_names, clf.feature_importances_.tolist()), key=lambda x: -x[1]))
        return imp
    return {}

def main():
    exp_id = sys.argv[1]
    model_type = sys.argv[2]
    params = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    feature_cols = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    
    out_dir = os.path.expanduser(f"~/research/artifacts/{exp_id}")
    os.makedirs(out_dir, exist_ok=True)
    
    start = time.time()
    ll_mean, ll_std, acc_mean, acc_std, X, y, pipe = run_cv(model_type, params, feature_cols)
    elapsed = time.time() - start
    
    fc = feature_cols or FEATURE_COLS
    fi = get_feature_importance(model_type, params, fc)
    
    metrics = {
        "log_loss_mean": round(ll_mean, 5),
        "log_loss_std": round(ll_std, 5),
        "accuracy_mean": round(acc_mean, 4),
        "accuracy_std": round(acc_std, 4),
        "n_samples": int(len(y)),
        "n_features": int(len(fc)),
        "cv": f"{N_SPLITS}fold x {N_REPEATS}repeats",
        "feature_importance_top10": dict(list(fi.items())[:10]),
    }
    
    run_info = {
        "exp_id": exp_id,
        "model_type": model_type,
        "params": params,
        "feature_cols": fc,
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "elapsed_seconds": round(elapsed, 2),
        "command": f"python3 run_experiment.py {exp_id} {model_type} '{json.dumps(params)}'",
    }
    
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{out_dir}/run.json", "w") as f:
        json.dump(run_info, f, indent=2)
    
    print(f"[{exp_id}] {model_type} params={params}")
    print(f"  log-loss: {ll_mean:.4f} ± {ll_std:.4f}")
    print(f"  accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  top features: {list(fi.keys())[:5]}")

if __name__ == "__main__":
    main()
