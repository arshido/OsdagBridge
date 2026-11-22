"""
Unit tests for Task III: Result Plotting for Transverse Members.
Covers:
  1. Transverse member classification (results_data.py) with and without groups.
  2. Transverse results extraction (forces & displacements across load cases).
  3. Post-processing retention of transverse data structures.
  4. 2D/3D Matplotlib Plot Generation (plot_generator.py) for SFD, BMD, Deflection, Grillage:
     - "All Transverse", "End Diaphragms", "Intermediate Cross Bracings".
     - Graceful handling of zero intermediate cross bracings (short span).
     - Extreme and summary data generation.
     - Dynamic titles and checkbox annotation tags (gid).
  5. Plotly renderer (plots_widget.py) support for transverse members.
  6. Output dock member dropdown population.
"""

from __future__ import annotations
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

import matplotlib
matplotlib.use('Agg')

# -----------------------------------------------------------------------------
# Dependency Mocks (openseespy, plotly, xarray, PySide6)
# -----------------------------------------------------------------------------
mock_ops = MagicMock()
mock_ops.getNodeTags.return_value = []
mock_ops.getEleTags.return_value = []
sys.modules['openseespy'] = MagicMock()
sys.modules['openseespy.opensees'] = mock_ops

mock_plotly = MagicMock()
mock_go = MagicMock()

class MockFigure:
    def __init__(self, *args, **kwargs):
        self.data = []
        self.layout = {}
        self.annotations = []
    def add_trace(self, trace):
        self.data.append(trace)
    def update_layout(self, *args, **kwargs):
        self.layout.update(kwargs)
    def to_json(self):
        return '{"data": [], "layout": {}}'
    def add_annotation(self, **kwargs):
        self.annotations.append(kwargs)

mock_go.Figure = MockFigure
mock_go.Scatter3d = lambda **kw: MagicMock(**kw)
mock_go.Surface = lambda **kw: MagicMock(**kw)
mock_go.Cone = lambda **kw: MagicMock(**kw)
mock_plotly.graph_objects = mock_go
sys.modules['plotly'] = mock_plotly
sys.modules['plotly.graph_objects'] = mock_go

# Lightweight Dataset / DataArray mock for xarray
class MockCoords:
    def __init__(self, d):
        self._d = d
    def __contains__(self, key):
        return key in self._d
    def __getitem__(self, key):
        class _CoordVal:
            def __init__(self, vals):
                self.values = np.array(vals)
        return _CoordVal(self._d.get(key, []))

class MockDataArray:
    def __init__(self, data_fn, coords_dict):
        self._data_fn = data_fn
        self.coords = MockCoords(coords_dict)
        self.values = np.array([])
    def sel(self, **kwargs):
        val = self._data_fn(**kwargs)
        class _Val:
            def __init__(self, v):
                self.values = np.array(v)
            def __float__(self):
                return float(self.values)
        return _Val(val)

class MockDataset:
    def __init__(self, arrays, coords=None):
        self._arrays = arrays
        self.coords = MockCoords(coords or {})
    def __contains__(self, key):
        return key in self._arrays or key in self.coords._d
    def __getitem__(self, key):
        if key in self._arrays:
            return self._arrays[key]
        if key in self.coords._d:
            class _CoordVal:
                def __init__(self, vals):
                    self.values = np.array(vals)
            return _CoordVal(self.coords._d[key])
        raise KeyError(key)
    def sel(self, **kwargs):
        sub_arrays = {}
        for k, arr in self._arrays.items():
            def make_sub_fn(original_fn):
                def sub_fn(**inner_kwargs):
                    merged = dict(kwargs)
                    merged.update(inner_kwargs)
                    return original_fn(**merged)
                return sub_fn
            sub_coords = {ck: cv for ck, cv in arr.coords._d.items() if ck not in kwargs}
            sub_arrays[k] = MockDataArray(make_sub_fn(arr._data_fn), sub_coords)
        return MockDataset(sub_arrays, self.coords._d)

mock_xr = MagicMock()
mock_xr.Dataset = MockDataset
mock_xr.DataArray = MockDataArray
sys.modules['xarray'] = mock_xr

sys.path.insert(0, 'src')

# Now import project modules
from osdagbridge.core.bridge_types.plate_girder.results_data import (
    classify_transverse_members,
    extract_transverse_results,
)
from osdagbridge.core.bridge_types.plate_girder.results_data_post_processing import (
    post_process,
    OUTPUT_KEYS,
)
from osdagbridge.core.bridge_types.plate_girder.plot_generator import (
    TRANSVERSE_OPTIONS,
    _find_transverse_lines,
    _build_transverse_coords,
    build_figure_sfd,
    build_figure_bmd,
    build_figure_deflection,
    build_figure_grillage,
)
from osdagbridge.core.bridge_types.plate_girder.plots_widget import (
    build_figure_sfd as plotly_build_figure_sfd,
    build_figure_bmd as plotly_build_figure_bmd,
)


class TestTransverseClassification(unittest.TestCase):
    """Test classification of transverse members from geometry and groups."""

    def setUp(self):
        # A simple 2-girder, 3-transverse-line bridge model
        # G1 at z=0, G2 at z=3.0
        # Transverse lines at x=0 (start), x=10 (intermediate), x=20 (end)
        self.nodes = {
            1: [0.0, 0.0, 0.0],
            2: [10.0, 0.0, 0.0],
            3: [20.0, 0.0, 0.0],
            4: [0.0, 0.0, 3.0],
            5: [10.0, 0.0, 3.0],
            6: [20.0, 0.0, 3.0],
        }
        # Longitudinal members: 101 (1-2), 102 (2-3), 103 (4-5), 104 (5-6)
        # Transverse members: 201 (1-4), 202 (2-5), 203 (3-6)
        self.members = {
            101: [1, 2],
            102: [2, 3],
            103: [4, 5],
            104: [5, 6],
            201: [1, 4],  # x=0 (start edge diaphragm)
            202: [2, 5],  # x=10 (intermediate cross bracing)
            203: [3, 6],  # x=20 (end edge diaphragm)
        }
        self.groups = {
            "start_edge": [201],
            "transverse_slab": [202],
            "end_edge": [203],
            "main_girder": [101, 102, 103, 104],
        }

    def test_classify_with_groups(self):
        """Classification using ospgrillage groups dict."""
        classified = classify_transverse_members(self.nodes, self.members, self.groups)
        self.assertIn(201, classified["end_diaphragms"])
        self.assertIn(203, classified["end_diaphragms"])
        self.assertIn(202, classified["intermediate_cross_bracings"])
        self.assertEqual(len(classified["end_diaphragms"]), 2)
        self.assertEqual(len(classified["intermediate_cross_bracings"]), 1)
        self.assertEqual(len(classified["all_transverse"]), 3)

    def test_classify_without_groups_geometric_fallback(self):
        """Classification falling back to geometry when groups is None."""
        classified = classify_transverse_members(self.nodes, self.members, groups=None)
        self.assertIn(201, classified["end_diaphragms"])
        self.assertIn(203, classified["end_diaphragms"])
        self.assertIn(202, classified["intermediate_cross_bracings"])
        self.assertEqual(len(classified["end_diaphragms"]), 2)
        self.assertEqual(len(classified["intermediate_cross_bracings"]), 1)

    def test_classify_short_span_zero_intermediate(self):
        """Short span with only end diaphragms has empty intermediate cross bracings."""
        nodes_short = {
            1: [0.0, 0.0, 0.0],
            2: [5.0, 0.0, 0.0],
            3: [0.0, 0.0, 3.0],
            4: [5.0, 0.0, 3.0],
        }
        members_short = {
            101: [1, 2],
            102: [3, 4],
            201: [1, 3],  # Start ED
            202: [2, 4],  # End ED
        }
        classified = classify_transverse_members(nodes_short, members_short, groups=None)
        self.assertEqual(len(classified["end_diaphragms"]), 2)
        self.assertEqual(len(classified["intermediate_cross_bracings"]), 0)


class TestTransverseResultsExtraction(unittest.TestCase):
    """Test extraction of forces and displacements for transverse members."""

    def setUp(self):
        self.nodes = {
            1: [0.0, 0.0, 0.0],
            2: [10.0, 0.0, 0.0],
            3: [0.0, 0.0, 3.0],
            4: [10.0, 0.0, 3.0],
        }
        self.members = {
            101: [1, 2],
            102: [3, 4],
            201: [1, 3],  # start ED
            202: [2, 4],  # end ED
        }
        loadcases = ["LC1", "LC2"]
        elements = [101, 102, 201, 202]
        force_components = ["Vx_i", "Vx_j", "Vy_i", "Vy_j", "Vz_i", "Vz_j",
                            "Mx_i", "Mx_j", "My_i", "My_j", "Mz_i", "Mz_j"]

        def mock_force_fn(**kwargs):
            return 1000.0  # 1000 N / N-mm

        forces_da = MockDataArray(mock_force_fn, {
            "Loadcase": loadcases,
            "Element": elements,
            "Component": force_components,
        })

        nodes_list = [1, 2, 3, 4]
        disp_components = ["x", "y", "z"]

        def mock_disp_fn(**kwargs):
            return 0.005  # 5 mm

        disp_da = MockDataArray(mock_disp_fn, {
            "Loadcase": loadcases,
            "Node": nodes_list,
            "Component": disp_components,
        })

        self.mock_data = {
            "nodes": self.nodes,
            "members": self.members,
            "groups": {"start_edge": [201], "end_edge": [202]},
            "forces": forces_da,
            "displacements": disp_da,
        }

    def test_extract_transverse_results(self):
        """Verify internal forces and displacements extraction across components."""
        t_res = extract_transverse_results(self.mock_data)
        self.assertIn("LC1", t_res)
        self.assertIn("LC2", t_res)
        lc1_res = t_res["LC1"]
        self.assertIn("forces", lc1_res)
        self.assertIn("displacements", lc1_res)

        # Check forces conversion: 1000 N -> 1.0 kN
        self.assertIn(201, lc1_res["forces"])
        f_201 = lc1_res["forces"][201]
        for key in ("Fx", "Vy", "Vz", "Tx", "My", "Mz"):
            self.assertIn(key, f_201)
            self.assertIn("i", f_201[key])
            self.assertIn("j", f_201[key])
            self.assertAlmostEqual(f_201[key]["i"], 1.0, places=3)

        # Check displacements conversion: 0.005 m -> 5.0 mm
        self.assertIn(1, lc1_res["displacements"])
        d_1 = lc1_res["displacements"][1]
        for key in ("Dx", "Dy", "Dz"):
            self.assertIn(key, d_1)
            self.assertAlmostEqual(d_1[key], 5.0, places=3)

    def test_post_process_preserves_transverse_keys(self):
        """Verify results_data_post_processing preserves transverse outputs."""
        self.mock_data["end_diaphragms"] = [201, 202]
        self.mock_data["intermediate_cross_bracings"] = []
        self.mock_data["transverse_results"] = extract_transverse_results(self.mock_data)
        self.mock_data["applied_forces"] = {}

        processed = post_process(self.mock_data)
        self.assertIn("end_diaphragms", processed)
        self.assertIn("intermediate_cross_bracings", processed)
        self.assertIn("transverse_results", processed)
        self.assertIn("groups", processed)


class TestPlotGeneratorTransverse(unittest.TestCase):
    """Test 2D/3D Matplotlib plot generator with transverse member options."""

    def setUp(self):
        self.nodes = {
            1: [0.0, 0.0, 0.0],
            2: [10.0, 0.0, 0.0],
            3: [20.0, 0.0, 0.0],
            4: [0.0, 0.0, 3.0],
            5: [10.0, 0.0, 3.0],
            6: [20.0, 0.0, 3.0],
        }
        self.members = {
            101: [1, 2],
            102: [2, 3],
            103: [4, 5],
            104: [5, 6],
            201: [1, 4],  # start ED
            202: [2, 5],  # intermediate CB
            203: [3, 6],  # end ED
        }
        elements = [101, 102, 103, 104, 201, 202, 203]
        force_components = ["Vx_i", "Vx_j", "Vy_i", "Vy_j", "Vz_i", "Vz_j",
                            "Mx_i", "Mx_j", "My_i", "My_j", "Mz_i", "Mz_j"]

        def mock_force_fn(**kwargs):
            return 25.0

        forces_da = MockDataArray(mock_force_fn, {
            "Element": elements,
            "Component": force_components,
        })

        nodes_list = [1, 2, 3, 4, 5, 6]
        disp_components = ["x", "y", "z"]

        def mock_disp_fn(**kwargs):
            return -0.0125  # in meters -> -12.5 mm

        disp_da = MockDataArray(mock_disp_fn, {
            "Node": nodes_list,
            "Component": disp_components,
        })

        self.ds = MockDataset({
            "forces": forces_da,
            "displacements": disp_da,
        }, coords={"Component": force_components})

    def test_find_transverse_lines(self):
        """Test clustering of transverse elements into lines."""
        t_lines = _find_transverse_lines(self.nodes, self.members)
        self.assertIn("all", t_lines)
        self.assertIn("end_diaphragms", t_lines)
        self.assertIn("intermediate_cross_bracings", t_lines)

        self.assertEqual(len(t_lines["all"]), 3)
        self.assertEqual(len(t_lines["end_diaphragms"]), 2)
        self.assertEqual(len(t_lines["intermediate_cross_bracings"]), 1)

    def test_build_transverse_coords(self):
        """Test coordinate building for transverse line polyline."""
        xs, ys, zs, node_ids = _build_transverse_coords([201], self.members, self.nodes)
        self.assertEqual(len(xs), 2)
        self.assertEqual(xs[0], 0.0)
        self.assertEqual(xs[1], 0.0)
        self.assertAlmostEqual(zs[0], 0.0)
        self.assertAlmostEqual(zs[1], 3.0)

    def test_build_figure_sfd_transverse_options(self):
        """Verify build_figure_sfd handles all transverse selections."""
        for option in ("All Transverse", "End Diaphragms", "Intermediate Cross Bracings"):
            fig, summary = build_figure_sfd(self.ds, "Vy", self.nodes, self.members, selected_girder=option)
            self.assertIsNotNone(fig)
            self.assertGreater(len(summary), 0)
            ax = fig.axes[0]
            self.assertIn(option, ax.get_title())

    def test_build_figure_bmd_transverse_options(self):
        """Verify build_figure_bmd handles all transverse selections."""
        for option in ("All Transverse", "End Diaphragms", "Intermediate Cross Bracings"):
            fig, summary = build_figure_bmd(self.ds, "Mz", self.nodes, self.members, selected_girder=option)
            self.assertIsNotNone(fig)
            self.assertGreater(len(summary), 0)
            ax = fig.axes[0]
            self.assertIn(option, ax.get_title())

    def test_build_figure_deflection_transverse_options(self):
        """Verify build_figure_deflection handles all transverse selections."""
        for option in ("All Transverse", "End Diaphragms", "Intermediate Cross Bracings"):
            fig, summary = build_figure_deflection(self.ds, "Dy", self.nodes, self.members, selected_girder=option)
            self.assertIsNotNone(fig)
            self.assertGreater(len(summary), 0)
            ax = fig.axes[0]
            self.assertIn(option, ax.get_title())

    def test_build_figure_grillage_transverse_options(self):
        """Verify build_figure_grillage highlights transverse members cleanly."""
        for option in ("All Transverse", "End Diaphragms", "Intermediate Cross Bracings"):
            fig = build_figure_grillage(self.nodes, self.members, selected_girder=option)
            self.assertIsNotNone(fig)
            ax = fig.axes[0]
            self.assertIn(option, ax.get_title())

    def test_zero_intermediate_cross_bracings_graceful_handling(self):
        """Gracefully handle bridge models with zero intermediate cross bracings."""
        nodes_no_cb = {
            1: [0.0, 0.0, 0.0],
            2: [10.0, 0.0, 0.0],
            3: [0.0, 0.0, 3.0],
            4: [10.0, 0.0, 3.0],
        }
        members_no_cb = {
            101: [1, 2],
            102: [3, 4],
            201: [1, 3],  # Start ED
            202: [2, 4],  # End ED
        }
        elements = [101, 102, 201, 202]
        force_components = ["Vx_i", "Vx_j", "Vy_i", "Vy_j", "Vz_i", "Vz_j",
                            "Mx_i", "Mx_j", "My_i", "My_j", "Mz_i", "Mz_j"]

        forces_da = MockDataArray(lambda **kw: 10.0, {
            "Element": elements,
            "Component": force_components,
        })
        disp_da = MockDataArray(lambda **kw: -0.005, {
            "Node": [1, 2, 3, 4],
            "Component": ["x", "y", "z"],
        })
        ds = MockDataset({"forces": forces_da, "displacements": disp_da},
                         coords={"Component": force_components})

        # Test SFD with zero intermediate cross bracings
        fig_sfd, summary_sfd = build_figure_sfd(ds, "Vy", nodes_no_cb, members_no_cb,
                                                selected_girder="Intermediate Cross Bracings")
        self.assertIsNotNone(fig_sfd)
        self.assertEqual(len(summary_sfd), 0)
        self.assertIn("Intermediate Cross Bracings (None)", fig_sfd.axes[0].get_title())

        # Test BMD with zero intermediate cross bracings
        fig_bmd, summary_bmd = build_figure_bmd(ds, "Mz", nodes_no_cb, members_no_cb,
                                                selected_girder="Intermediate Cross Bracings")
        self.assertIsNotNone(fig_bmd)
        self.assertEqual(len(summary_bmd), 0)
        self.assertIn("Intermediate Cross Bracings (None)", fig_bmd.axes[0].get_title())

        # Test Deflection with zero intermediate cross bracings
        fig_defl, summary_defl = build_figure_deflection(ds, "Dy", nodes_no_cb, members_no_cb,
                                                         selected_girder="Intermediate Cross Bracings")
        self.assertIsNotNone(fig_defl)
        self.assertEqual(len(summary_defl), 0)
        self.assertIn("Intermediate Cross Bracings (None)", fig_defl.axes[0].get_title())

    def test_checkbox_annotation_gids_present(self):
        """Verify gids for toolbar toggling: max_line, min_line, all_vals, girder_labels."""
        fig, _ = build_figure_bmd(self.ds, "Mz", self.nodes, self.members, selected_girder="All Transverse")
        ax = fig.axes[0]
        gids_found = set()
        for child in ax.get_children():
            gid = child.get_gid()
            if gid:
                gids_found.add(gid)
        for text in ax.texts:
            gid = text.get_gid()
            if gid:
                gids_found.add(gid)

        self.assertIn("max_line", gids_found)
        self.assertIn("min_line", gids_found)
        self.assertIn("all_vals", gids_found)
        self.assertIn("girder_labels", gids_found)


class TestPlotlyRendererTransverse(unittest.TestCase):
    """Test Plotly visualizer in plots_widget.py with transverse options."""

    def setUp(self):
        self.nodes = {
            1: [0.0, 0.0, 0.0],
            2: [10.0, 0.0, 0.0],
            3: [20.0, 0.0, 0.0],
            4: [0.0, 0.0, 3.0],
            5: [10.0, 0.0, 3.0],
            6: [20.0, 0.0, 3.0],
        }
        self.members = {
            101: [1, 2],
            102: [2, 3],
            103: [4, 5],
            104: [5, 6],
            201: [1, 4],  # start ED
            202: [2, 5],  # intermediate CB
            203: [3, 6],  # end ED
        }
        elements = [101, 102, 103, 104, 201, 202, 203]
        force_components = ["Vx_i", "Vx_j", "Vy_i", "Vy_j", "Vz_i", "Vz_j",
                            "Mx_i", "Mx_j", "My_i", "My_j", "Mz_i", "Mz_j"]

        forces_da = MockDataArray(lambda **kw: 15.0, {
            "Element": elements,
            "Component": force_components,
        })
        self.ds = MockDataset({
            "forces": forces_da,
        }, coords={"Component": force_components})

    def test_plotly_sfd_transverse(self):
        """Test build_figure_sfd in plots_widget produces valid JSON for transverse members."""
        json_str = plotly_build_figure_sfd(self.ds, "Fy", self.nodes, self.members,
                                           selected_member="All Transverse")
        self.assertIsInstance(json_str, str)

    def test_plotly_bmd_transverse(self):
        """Test build_figure_bmd in plots_widget produces valid JSON and summary for transverse members."""
        json_str, summary = plotly_build_figure_bmd(self.ds, "Mz", self.nodes, self.members,
                                                    selected_member="End Diaphragms")
        self.assertIsInstance(json_str, str)
        self.assertIn("ED 1 (Start)", summary)
        self.assertIn("ED 2 (End)", summary)


class TestOutputDockMemberDropdown(unittest.TestCase):
    """Test output dock member dropdown options include transverse selections."""

    def test_refresh_member_dropdown_transverse_items(self):
        """Verify OutputDock.refresh_member_dropdown adds transverse options."""
        from tests.unit.mock_pyside6 import MockQWidget, MockSignal
        from osdagbridge.core.utils.common import KEY_ANALYSIS_MEMBER, KEY_OUTPUT_DOCK_MEMBER_ID, KEY_TS_NO_OF_GIRDERS

        class MockComboBox(MockQWidget):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.items = []
                self.currentIndex = 0
            def clear(self):
                self.items = []
            def addItems(self, items):
                self.items.extend(items)
            def setCurrentIndex(self, idx):
                self.currentIndex = idx
            def blockSignals(self, b):
                pass
            def setSizeAdjustPolicy(self, p):
                pass
            def setSizePolicy(self, h, v):
                pass

        class MockBackend:
            def __init__(self):
                self.input_dict = {KEY_TS_NO_OF_GIRDERS: 3}

        # Mock OutputDock widget lookup
        analysis_combo = MockComboBox()
        design_combo = MockComboBox()

        def mock_w(key):
            if key == KEY_ANALYSIS_MEMBER:
                return analysis_combo
            elif key == KEY_OUTPUT_DOCK_MEMBER_ID:
                return design_combo
            return None

        from osdagbridge.desktop.ui.docks.output_dock import OutputDock
        dock = OutputDock.__new__(OutputDock)
        dock.backend = MockBackend()
        dock._w = mock_w

        dock.refresh_member_dropdown()

        # Check design combo has G1, G2, G3
        self.assertEqual(design_combo.items, ["G1", "G2", "G3"])

        # Check analysis combo has All, G1, G2, G3, All Transverse, End Diaphragms, Intermediate Cross Bracings
        expected_analysis = ["All", "G1", "G2", "G3", "All Transverse", "End Diaphragms", "Intermediate Cross Bracings"]
        self.assertEqual(analysis_combo.items, expected_analysis)


if __name__ == "__main__":
    unittest.main()
