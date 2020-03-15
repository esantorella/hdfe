import numpy as np
import pandas as pd
from hdfe import Groupby
import pytest


@pytest.fixture
def df() -> pd.DataFrame:
    np.random.seed(0)
    n_obs = 100
    n_categories = 10
    return pd.DataFrame(
        {
            "first category": np.random.choice(n_categories, n_obs),
            "y": np.random.normal(0, 1, n_obs),
        }
    )


def test_groupby_apply_mean(df: pd.DataFrame) -> None:
    pandas_results = df.groupby("first category")[["y"]].mean()
    groupby_results = Groupby(df["first category"]).apply(
        np.mean, df["y"], broadcast=False, as_dataframe=True
    )
    pd.testing.assert_frame_equal(pandas_results, groupby_results)


def test_groupby_transform_mean(df: pd.DataFrame) -> None:
    pandas_results = df.groupby("first category")["y"].transform("mean")
    groupby_results = Groupby(df["first category"]).apply(
        np.mean, df["y"], broadcast=True, as_dataframe=True
    )
    np.testing.assert_almost_equal(
        pandas_results.values, np.squeeze(groupby_results.values)
    )
