import pandas as pd
from sklearn.base import clone 
from sklearn.model_selection import cross_val_score

def feature_ablation_func(model,
    X: pd.DataFrame,
    y: pd.Series,
    features_to_remove: list[str],
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error",
) -> pd.DataFrame:
    """
    Measure the effect of removing individual features.

    Args:
        model: An sklearn-compatible model, e.g. XGBRegressor.
        X: Input feature DataFrame.
        y: Target values.
        features_to_remove: Features to remove one at a time.
        cv: Number of cross-validation folds.
        scoring: Scikit-learn scoring metric.

    Returns:
        DataFrame containing each feature's score and change from baseline.
        Positive score_change means removing the feature improved performance.
    """

    missing_features = set(features_to_remove) - set(X.columns)
    if missing_features:
        raise ValueError(f"Featuer not found {missing_features} cols:{X.columns} , features_to_remove : {features_to_remove}")

    def evaluate(features):
        scores = cross_val_score(
            clone(model),
            X[features],
            y,
            cv = cv,
            scoring = scoring
        )
        return scores, scores.mean()
    ans = []
    baseline_raw,baseline = evaluate(X.columns.to_list())

    ans.append({
        "removed_feature": None,
        "score": baseline,
        "score_change": 0.0,
        "Score Raw": baseline_raw
    })
    for i in features_to_remove:
        remaining = X.columns.drop(i).to_list()
        score_raw, score = evaluate(remaining)

        ans.append({
            "removed_feature": i,
            "score": score,
            "score_change": score - baseline,
            "Score Raw": score_raw
        })

    return pd.DataFrame(ans).sort_values("score_change", ascending=False).reset_index(drop = True)