"""Model selection, tuning, and final refit for the crop-yield task.

Design
------
Rule 6 of the assignment says everything must be fitted on train only. Every
candidate here is a single scikit-learn ``Pipeline`` combining

    categorical encoder  ->  {"onehot", "ordinal"}
    numeric scaler       ->  {"standard", "minmax", "robust"}
    regressor            ->  one of the 16 models in MODEL_SPECS

so the exact same fitted steps run on the test file and in the demo app.

A full sweep of every model in ``MODEL_SPECS`` (18 by default, more when XGBoost /
LightGBM / CatBoost are importable) x 2 encoders x 3 scalers is scored on one
shared held-out validation split, the winner is tuned with RandomizedSearchCV,
optionally blended with the runner-up families, and only then refitted on 100% of
the training data.
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor,
                              VotingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   RobustScaler, StandardScaler)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone

RANDOM_STATE = 42

# Optional gradient-boosting libraries: used when importable, skipped otherwise
# so the notebook still runs end to end in a minimal environment.
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:  # pragma: no cover
    HAS_CATBOOST = False


ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
}

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def model_specs():
    """name -> (estimator factory, log_target, note)."""
    specs = {
        "mean_predictor": (lambda: DummyRegressor(strategy="mean"), False,
                           "baseline: always predicts the training mean"),
        "linear_regression": (lambda: LinearRegression(), False, "ordinary least squares"),
        "ridge": (lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE), False, "L2-regularised linear"),
        "lasso": (lambda: Lasso(alpha=0.01, max_iter=8000, random_state=RANDOM_STATE), False, "L1-regularised linear"),
        "elastic_net": (lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=8000, random_state=RANDOM_STATE), False, "L1+L2 linear"),
        "knn": (lambda: KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1), False, "distance-weighted nearest neighbours"),
        "decision_tree": (lambda: DecisionTreeRegressor(max_depth=14, random_state=RANDOM_STATE), False, "single tree"),
        "random_forest": (lambda: RandomForestRegressor(n_estimators=120, max_depth=24, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1), False, "bagged trees"),
        "extra_trees": (lambda: ExtraTreesRegressor(n_estimators=120, max_depth=24, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1), False, "extremely randomised trees"),
        "gradient_boosting": (lambda: GradientBoostingRegressor(n_estimators=120, learning_rate=0.07, max_depth=4, random_state=RANDOM_STATE), False, "classic boosted trees"),
        "hist_gradient_boosting": (lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=RANDOM_STATE), False, "sklearn histogram boosting"),
        "mlp": (lambda: MLPRegressor(hidden_layer_sizes=(96, 48), batch_size=256, max_iter=200, early_stopping=True, random_state=RANDOM_STATE), False, "neural network"),
        "svr": (lambda: SVR(C=3.0, epsilon=0.1, kernel="rbf", cache_size=1000), False, "support-vector regression"),
    }
    if HAS_XGB:
        specs["xgboost"] = (lambda: XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=7,
                                                 subsample=0.8, colsample_bytree=0.8,
                                                 random_state=RANDOM_STATE, n_jobs=-1), False, "boosted trees (XGBoost)")
    if HAS_LGBM:
        specs["lightgbm"] = (lambda: LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                                   min_child_samples=10, verbose=-1,
                                                   random_state=RANDOM_STATE, n_jobs=-1), False,
                             "boosted trees (LightGBM)")
        # Variant in the same family but fitting log(yield) and inverting the
        # prediction: the yield process looks multiplicative (crop base x soil x
        # fertilizer x weather penalty), so log space helps the trees.
        specs["lightgbm_log_target"] = (
            lambda: LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                  min_child_samples=10, verbose=-1,
                                  random_state=RANDOM_STATE, n_jobs=-1),
            True, "boosted trees (LightGBM), log target")
    if HAS_CATBOOST:
        specs["catboost"] = (lambda: CatBoostRegressor(iterations=500, learning_rate=0.06, depth=7,
                                                       verbose=0, random_state=RANDOM_STATE,
                                                       allow_writing_files=False, thread_count=-1), False,
                             "boosted trees (CatBoost)")
        specs["catboost_log_target"] = (
            lambda: CatBoostRegressor(iterations=500, learning_rate=0.06, depth=7,
                                      verbose=0, random_state=RANDOM_STATE,
                                      allow_writing_files=False, thread_count=-1),
            True, "boosted trees (CatBoost), log target")
    return specs


def build_preprocessor(categorical, numeric, encoder_name, scaler_name):
    return ColumnTransformer([
        ("cat", ENCODERS[encoder_name](), categorical),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", SCALERS[scaler_name]()),
        ]), numeric),
    ])


def build_pipeline(factory, categorical, numeric, encoder_name, scaler_name, log_target=False):
    pipe = Pipeline([
        ("prep", build_preprocessor(categorical, numeric, encoder_name, scaler_name)),
        ("model", factory()),
    ])
    if log_target:
        pipe = TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


def rmse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def score(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_model_selection(X_train, y_train, X_val, y_val, categorical, numeric,
                        encoder_names=None, scaler_names=None, progress=True):
    """Score every (model, encoder, scaler) combination on the shared validation split.

    Returns (results DataFrame sorted by RMSE, dict of fitted pipelines).
    """
    encoder_names = list(encoder_names or ENCODERS)
    scaler_names = list(scaler_names or SCALERS)
    specs = model_specs()
    total = len(specs) * len(encoder_names) * len(scaler_names)

    rows, fitted = [], {}
    done = 0
    for model_name, (factory, log_target, _note) in specs.items():
        for enc in encoder_names:
            for sca in scaler_names:
                done += 1
                key = (model_name, enc, sca)
                pipe = build_pipeline(factory, categorical, numeric, enc, sca, log_target)
                start = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X_train, y_train)
                    preds = pipe.predict(X_val)
                elapsed = time.time() - start
                rows.append({
                    "model": model_name,
                    "encoder": enc,
                    "scaler": sca,
                    **score(y_val, preds),
                    "train_time_sec": round(elapsed, 2),
                })
                fitted[key] = pipe
                if progress and (done % 12 == 0 or done == total):
                    print(f"  {done}/{total} combinations scored...")

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results, fitted


def stratified_subsample(X, y, n, stratify_col="crop_type", random_state=RANDOM_STATE):
    """Smaller, class-proportional copy of (X, y) used to screen the full grid cheaply.

    Screening 18 models x 2 encoders x 3 scalers on the whole 12k-row training
    split takes far longer than it is worth: the ranking is already stable at
    this size, and every finalist is re-scored on the full split afterwards.
    """
    if n is None or n >= len(X):
        return X, y
    frame = X.copy()
    frame["__target__"] = np.asarray(y)
    if stratify_col in frame.columns:
        frac = n / len(frame)
        sampled = frame.groupby(stratify_col, group_keys=False).sample(
            frac=frac, random_state=random_state)
    else:
        sampled = frame.sample(n, random_state=random_state)
    return sampled.drop(columns=["__target__"]), sampled["__target__"]


def confirm_on_full_split(screen_results, X_train, y_train, X_val, y_val,
                          categorical, numeric):
    """Refit the best encoder/scaler combo of every model family on the FULL
    training split and re-score it on the same validation split."""
    specs = model_specs()
    best_per_family = (screen_results.sort_values("RMSE")
                       .groupby("model", as_index=False).first())
    rows = []
    for _, r in best_per_family.iterrows():
        factory, log_target, note = specs[r["model"]]
        pipe = build_pipeline(factory, categorical, numeric, r["encoder"], r["scaler"], log_target)
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)
        rows.append({
            "model": r["model"],
            "encoder": r["encoder"],
            "scaler": r["scaler"],
            "note": note,
            **score(y_val, preds),
            "train_time_sec": round(time.time() - start, 2),
        })
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


TUNING_SPACES = {
    "lightgbm": {
        "model__n_estimators": [300, 500, 800, 1200],
        "model__learning_rate": [0.01, 0.02, 0.05],
        "model__num_leaves": [15, 31, 63, 127],
        "model__min_child_samples": [5, 10, 20, 40],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__subsample": [0.7, 0.9, 1.0],
    },
    "catboost": {
        "model__iterations": [400, 700, 1000],
        "model__learning_rate": [0.02, 0.04, 0.06],
        "model__depth": [6, 7, 8],
        "model__l2_leaf_reg": [1, 3, 6],
    },
    "xgboost": {
        "model__n_estimators": [300, 500, 800],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__max_depth": [5, 7, 9],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
    },
    "random_forest": {
        "model__n_estimators": [200, 400, 700],
        "model__max_depth": [None, 12, 20, 30],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.5, 1.0],
    },
    "hist_gradient_boosting": {
        "model__max_iter": [200, 400, 700],
        "model__learning_rate": [0.02, 0.06, 0.1],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 40],
    },
}

# Single-threaded bases: RandomizedSearchCV parallelises across candidates, so
# letting each estimator also grab every core just causes oversubscription.
_TUNING_BASE = {
    "lightgbm": lambda: LGBMRegressor(verbose=-1, random_state=RANDOM_STATE, n_jobs=1),
    "catboost": lambda: CatBoostRegressor(verbose=0, random_state=RANDOM_STATE,
                                          allow_writing_files=False, thread_count=1),
    "xgboost": lambda: XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, tree_method="hist"),
    "random_forest": lambda: RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}


def tuning_base_name(model_name):
    return model_name.replace("_log_target", "")


def tune_pipeline(model_name, encoder_name, scaler_name, categorical, numeric,
                  X_train, y_train, n_iter=20, cv=3, log_target=False):
    """RandomizedSearchCV over the winner's family. Falls back to the untuned
    pipeline when the family has no documented search space."""
    base = tuning_base_name(model_name)
    if base not in TUNING_SPACES:
        return None, None
    param_dist = dict(TUNING_SPACES[base])
    if log_target:
        # the estimators live one level deeper when wrapped in a
        # TransformedTargetRegressor, so the parameter paths gain a prefix
        param_dist = {f"regressor__{k}": v for k, v in param_dist.items()}
    search = RandomizedSearchCV(
        build_pipeline(_TUNING_BASE[base], categorical, numeric, encoder_name, scaler_name, log_target),
        param_distributions=param_dist,
        n_iter=n_iter, cv=KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def cross_validate(pipeline, X, y, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(pipeline, X, y, cv=cv,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
    return -scores


def save_model_artifacts(model, model_path, metrics, metrics_path):
    """Persist the fitted pipeline and a small JSON model card the README/app read."""
    import joblib
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def build_voting_ensemble(members, weights=None):
    """Equal- or weighted-average blend of already-configured pipelines.

    ``members`` is a list of (name, pipeline). A plain average of a few
    decorrelated strong learners is the cheapest reliable way to shave a little
    more error off the single best model, and unlike stacking it needs no
    extra cross-validated refits.
    """
    if len(members) < 2:
        return None
    return VotingRegressor(estimators=members, weights=weights, n_jobs=-1)


def best_row(results):
    """Lowest-RMSE row of a selection table."""
    return results.sort_values("RMSE").iloc[0]
