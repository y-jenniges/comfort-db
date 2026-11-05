"""Tests for comfort.scaling."""
import pandas as pd
import pytest

from comfort.scaling import ParamScaler

_has_sklearn = True
try:
    import sklearn
except ImportError:
    _has_sklearn = False


@pytest.mark.skipif(not _has_sklearn, reason="scikit-learn not installed ([scale] extra)")
class TestParamScaler:
    # Default scaler_class is MinMaxScaler
    def test_default_scaler_is_minmax(self):
        scaler = ParamScaler()
        from sklearn.preprocessing import MinMaxScaler
        assert scaler.scaler_class is MinMaxScaler

    # scaler_class can be swapped for any sklearn-compatible scaler
    def test_custom_scaler_class(self):
        from sklearn.preprocessing import StandardScaler
        scaler = ParamScaler(scaler_class=StandardScaler)
        assert scaler.scaler_class is StandardScaler

    # scale() adds a VAL_SCALED column
    def test_scale_adds_val_scaled(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"VAL": [0.0, 5.0, 10.0], "PARAM_NAME": ["NITRATE"] * 3})
        result = scaler.scale(df)
        assert "VAL_SCALED" in result.columns

    # MinMax scaling maps the value range onto [0, 1]
    def test_scale_range_is_0_to_1(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"VAL": [0.0, 5.0, 10.0], "PARAM_NAME": ["NITRATE"] * 3})
        result = scaler.scale(df)
        assert result["VAL_SCALED"].min() == pytest.approx(0.0)
        assert result["VAL_SCALED"].max() == pytest.approx(1.0)

    # scale() returns a copy, the input DataFrame is untouched
    def test_scale_does_not_mutate_input(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"VAL": [0.0, 5.0, 10.0], "PARAM_NAME": ["NITRATE"] * 3})
        scaler.scale(df)
        assert "VAL_SCALED" not in df.columns

    # scale_columns() adds a _SCALED column per requested column
    def test_scale_columns(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"NITRATE": [0.0, 5.0, 10.0], "OXYGEN": [200.0, 250.0, 300.0]})
        result = scaler.scale_columns(["NITRATE", "OXYGEN"], df)
        assert "NITRATE_SCALED" in result.columns
        assert "OXYGEN_SCALED" in result.columns

    # scale_columns() also maps onto [0, 1] by default
    def test_scale_columns_range(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"NITRATE": [0.0, 5.0, 10.0]})
        result = scaler.scale_columns(["NITRATE"], df)
        assert result["NITRATE_SCALED"].min() == pytest.approx(0.0)
        assert result["NITRATE_SCALED"].max() == pytest.approx(1.0)

    # A requested column absent from df logs a warning instead of raising
    def test_scale_columns_missing_column_warns(self, caplog):
        scaler = ParamScaler()
        df = pd.DataFrame({"NITRATE": [0.0, 5.0, 10.0]})
        import logging
        with caplog.at_level(logging.WARNING):
            result = scaler.scale_columns(["MISSING"], df)
        assert "MISSING" in caplog.text
        assert "MISSING_SCALED" not in result.columns

    # scale_columns() returns a copy, the input DataFrame is untouched
    def test_scale_columns_does_not_mutate_input(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"NITRATE": [0.0, 5.0, 10.0]})
        scaler.scale_columns(["NITRATE"], df)
        assert "NITRATE_SCALED" not in df.columns

    # scale_value() applies a previously fitted column's scaler to a single value
    def test_scale_value_after_fit(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"NITRATE": [0.0, 10.0]})
        scaler.scale_columns(["NITRATE"], df)
        assert scaler.scale_value("NITRATE", 5.0) == pytest.approx(0.5)

    # scale_value() raises when the column has never been fitted
    def test_scale_value_unfitted_raises(self):
        scaler = ParamScaler()
        with pytest.raises(KeyError, match="No scaler fitted"):
            scaler.scale_value("NITRATE", 5.0)

    # Extra kwargs are forwarded to the underlying sklearn scaler
    def test_scaler_kwargs_forwarded(self):
        from sklearn.preprocessing import MinMaxScaler
        scaler = ParamScaler(scaler_class=MinMaxScaler, feature_range=(0, 100))
        df = pd.DataFrame({"NITRATE": [0.0, 5.0, 10.0]})
        result = scaler.scale_columns(["NITRATE"], df)
        assert result["NITRATE_SCALED"].max() == pytest.approx(100.0)

    # The fitted scaler persists across calls instead of refitting each time
    def test_scaler_reused_across_calls(self):
        scaler = ParamScaler()
        df = pd.DataFrame({"VAL": [0.0, 10.0], "PARAM_NAME": ["A"] * 2})
        scaler.scale(df)
        # Second call with different values should use the same fitted scaler
        df2 = pd.DataFrame({"VAL": [5.0, 15.0], "PARAM_NAME": ["A"] * 2})
        result = scaler.scale(df2)
        # 5.0 scaled with [0, 10] range -> 0.5; 15.0 -> 1.5 (extrapolated)
        assert result["VAL_SCALED"].iloc[0] == pytest.approx(0.5)
