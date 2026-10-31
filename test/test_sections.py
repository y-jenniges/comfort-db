"""Tests for comfort.sections, i.e. along-track section binning."""
import pandas as pd
import pytest

from comfort.sections import bin_section, section_difference


class TestBinSection:
    @staticmethod
    def _df():
        return pd.DataFrame({
            "ALONG": [0.0, 0.0, 10.0, 10.0],
            "LEV_M": [0.0, 100.0, 0.0, 100.0],
            "VAL": [10.0, 8.0, 6.0, 4.0],
        })

    # Output has one row per non-empty (along, depth) cell
    def test_returns_binned_values(self):
        result = bin_section(self._df(), "VAL", "LEV_M", "ALONG", along_bins=2, depth_bins=2)
        assert set(result.columns) == {"ALONG", "LEV_M", "VAL"}
        assert len(result) == 4

    # Values within a cell are averaged
    def test_aggregates_within_cell(self):
        df = pd.DataFrame({
            "ALONG": [0.0, 0.0],
            "LEV_M": [0.0, 0.0],
            "VAL": [10.0, 20.0],
        })
        result = bin_section(df, "VAL", "LEV_M", "ALONG", along_bins=1, depth_bins=1)
        assert result["VAL"].iloc[0] == pytest.approx(15.0)

    # Empty input returns an empty DataFrame with the expected columns
    def test_empty_input(self):
        df = pd.DataFrame({"ALONG": [], "LEV_M": [], "VAL": []})
        result = bin_section(df, "VAL", "LEV_M", "ALONG")
        assert result.empty
        assert list(result.columns) == ["ALONG", "LEV_M", "VAL"]

    # Rows with NaN values are dropped before binning
    def test_drops_nan_rows(self):
        df = pd.DataFrame({
            "ALONG": [0.0, 10.0],
            "LEV_M": [0.0, 0.0],
            "VAL": [10.0, float("nan")],
        })
        result = bin_section(df, "VAL", "LEV_M", "ALONG", along_bins=1, depth_bins=1)
        assert len(result) == 1
        assert result["VAL"].iloc[0] == pytest.approx(10.0)

    # Explicit bin edges (array-like) are accepted, not just a bin count
    def test_accepts_explicit_bin_edges(self):
        result = bin_section(self._df(), "VAL", "LEV_M", "ALONG",
                             along_bins=[-1, 5, 11], depth_bins=[-1, 50, 101])
        assert len(result) == 4

    # Bins are left-closed like the gridding: Lowest edge kept, interior edge goes up, maximum included
    def test_edge_convention_matches_gridding(self):
        df = pd.DataFrame({
            "ALONG": [0.0, 0.0, 0.0],
            "LEV_M": [0.0, 10.0, 20.0],
            "VAL": [2.0, 4.0, 9.0],
        })
        result = bin_section(df, "VAL", "LEV_M", "ALONG", along_bins=[-1, 1], depth_bins=[0, 10, 20])
        assert result.set_index("LEV_M")["VAL"].to_dict() == {5.0: 2.0, 15.0: 6.5}


class TestSectionDifference:
    @staticmethod
    def _binned(values):
        df = pd.DataFrame({
            "ALONG": [2.0, 2.0, 8.0, 8.0],
            "LEV_M": [10.0, 60.0, 10.0, 60.0],
            "VAL": values,
        })
        return bin_section(df, "VAL", "LEV_M", "ALONG",
                           along_bins=[0, 5, 10], depth_bins=[0, 50, 100])

    # Cell values are subtracted (section_a - section_b) at matching bin centres
    def test_computes_cellwise_difference(self):
        section_a = self._binned([10.0, 8.0, 6.0, 4.0])
        section_b = self._binned([1.0, 1.0, 1.0, 1.0])
        result = section_difference(section_a, section_b, "VAL", "LEV_M", "ALONG")
        assert set(result["VAL"]) == {9.0, 7.0, 5.0, 3.0}

    # Cells only present in one section are dropped (inner merge)
    def test_only_shared_cells_kept(self):
        section_a = self._binned([10.0, 8.0, 6.0, 4.0])
        section_b = self._binned([1.0, 1.0, 1.0, 1.0]).iloc[:2]
        result = section_difference(section_a, section_b, "VAL", "LEV_M", "ALONG")
        assert len(result) == 2


class TestColumnCasing:
    # Column arguments in another casing give the same result
    def test_bin_section_lower_case_arguments(self):
        df = TestBinSection._df()
        expected = bin_section(df, "VAL", "LEV_M", "ALONG", along_bins=2, depth_bins=2)
        result = bin_section(df, "val", "lev_m", "along", along_bins=2, depth_bins=2)
        pd.testing.assert_frame_equal(result, expected)
