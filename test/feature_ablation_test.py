import pytest
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder

from feature_ablation import feature_ablation_func


# --------------------
# TEST DATA
# --------------------

@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)

    X = pd.DataFrame({
        "age": np.arange(40),
        "income": rng.normal(50000, 10000, 40),
        "visits": rng.integers(1, 20, 40),
    })

    y = X["age"] * 3 + X["visits"] * 2

    return X, y


# --------------------
# MODEL COMPATIBILITY
# --------------------

def test_works_with_regression(sample_data):
    X, y = sample_data

    result = feature_ablation_func(
        LinearRegression(), X, y, ["age"], cv=5
    )

    assert isinstance(result, pd.DataFrame)


def test_works_with_decision_trees(sample_data):
    X, y = sample_data

    result = feature_ablation_func(
        DecisionTreeRegressor(random_state=42),
        X, y, ["age"], cv=5
    )

    assert isinstance(result, pd.DataFrame)


# --------------------
# MISSING DATA
# --------------------

def test_missing_data_raises_error(sample_data):
    X, y = sample_data
    X["income"] = np.nan

    with pytest.raises(ValueError):
        feature_ablation_func(
            LinearRegression(), X, y, ["age"], cv=5
        )


def test_missing_data_with_imputer_works(sample_data):
    X, y = sample_data
    X.loc[:5, "income"] = np.nan

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        LinearRegression()
    )

    result = feature_ablation_func(
        model, X, y, ["age"], cv=5
    )

    assert isinstance(result, pd.DataFrame)


# --------------------
# CATEGORICAL DATA
# --------------------

def test_regression_with_categorical_data_raises_error(sample_data):
    X, y = sample_data
    X["city"] = ["London", "Manchester"] * 20

    with pytest.raises(ValueError):
        feature_ablation_func(
            LinearRegression(), X, y, ["age"], cv=5
        )


def test_decision_tree_with_categorical_preprocessing(sample_data):
    X, y = sample_data
    X["city"] = ["London", "Manchester"] * 20

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include=["object", "str"]),
            )
        ],
        remainder="passthrough",
    )

    model = make_pipeline(
        preprocessing,
        DecisionTreeRegressor(random_state=42)
    )

    result = feature_ablation_func(
        model, X, y, ["age"], cv=5
    )

    assert isinstance(result, pd.DataFrame)


# --------------------
# REPRODUCIBILITY
# --------------------

def test_same_data_produces_same_results(sample_data):
    X, y = sample_data

    model = LinearRegression()

    result1 = feature_ablation_func(
        model, X, y, ["age", "income"], cv=5
    )

    result2 = feature_ablation_func(
        model, X, y, ["age", "income"], cv=5
    )

    pd.testing.assert_frame_equal(result1, result2)


# --------------------
# INPUT VALIDATION
# --------------------

def test_invalid_feature_raises_error(sample_data):
    X, y = sample_data

    with pytest.raises(ValueError):
        feature_ablation_func(
            LinearRegression(), X, y, ["nonexistent"], cv=5
        )


def test_original_data_is_not_modified(sample_data):
    X, y = sample_data
    original_X = X.copy(deep=True)

    feature_ablation_func(
        LinearRegression(), X, y, ["age"], cv=5
    )

    pd.testing.assert_frame_equal(X, original_X)


def test_original_model_is_not_fitted(sample_data):
    X, y = sample_data
    model = LinearRegression()

    feature_ablation_func(
        model, X, y, ["age"], cv=5
    )

    assert not hasattr(model, "coef_")