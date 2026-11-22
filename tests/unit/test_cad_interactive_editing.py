"""
Unit tests for Interactive 2D CAD Parameter Editing (Task I).
Tests hit-testing, unit-aware parsing/formatting, validation,
double-trigger guarding, escape cancellation, and left-dock flash.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

from tests.unit.mock_pyside6 import (
    MockSignal,
    _BoundMockSignal,
    MockQRectF,
    MockQWidget,
    mock_qt,
)

sys.path.insert(0, 'src')

from osdagbridge.desktop.ui.docks.cad_inline_editor import DimensionLabelZone, InlineCADEditor
from osdagbridge.core.utils.common import (
    KEY_CARRIAGEWAY_WIDTH,
    KEY_SPAN,
    KEY_TS_GIRDER_SPACING,
    KEY_TS_DECK_THICKNESS,
    KEY_TS_DECK_OVERHANG,
    KEY_TS_OVERALL_WIDTH,
    KEY_MP_CB_SPACING,
    KEY_TS_NO_OF_GIRDERS,
)


class TestInteractiveCADEditing(unittest.TestCase):
    def setUp(self):
        # Setup test zones
        self.zone_cw = DimensionLabelZone(
            rect=MockQRectF(100, 100, 120, 20),
            param_key=KEY_CARRIAGEWAY_WIDTH,
            editable=True,
            unit="m",
            current_value=10.5,
            label_name="Carriageway Width"
        )
        self.zone_gs = DimensionLabelZone(
            rect=MockQRectF(200, 300, 100, 20),
            param_key=KEY_TS_GIRDER_SPACING,
            editable=True,
            unit="m",
            current_value=2.75,
            label_name="Girder Spacing"
        )
        self.zone_dt = DimensionLabelZone(
            rect=MockQRectF(300, 150, 110, 20),
            param_key=KEY_TS_DECK_THICKNESS,
            editable=True,
            unit="mm",
            current_value=200.0,
            label_name="Deck Thickness"
        )
        self.zone_span = DimensionLabelZone(
            rect=MockQRectF(150, 400, 100, 20),
            param_key=KEY_SPAN,
            editable=True,
            unit="m",
            current_value=35.0,
            label_name="Span Length"
        )
        self.zone_overhang = DimensionLabelZone(
            rect=MockQRectF(50, 300, 80, 20),
            param_key=KEY_TS_DECK_OVERHANG,
            editable=False,
            unit="m",
            current_value=1.0,
            label_name="Overhang"
        )
        self.zone_overall_width = DimensionLabelZone(
            rect=MockQRectF(100, 50, 150, 20),
            param_key=KEY_TS_OVERALL_WIDTH,
            editable=False,
            unit="m",
            current_value=12.5,
            label_name="Overall Bridge Width"
        )
        self.zone_bracing = DimensionLabelZone(
            rect=MockQRectF(150, 370, 100, 20),
            param_key=KEY_MP_CB_SPACING,
            editable=False,
            unit="m",
            current_value=3.5,
            label_name="Bracing Spacing"
        )

    def test_editable_vs_derived_zones(self):
        """Verify the 4 write-enabled labels are editable and 3 derived labels are strictly locked."""
        self.assertTrue(self.zone_cw.editable)
        self.assertTrue(self.zone_gs.editable)
        self.assertTrue(self.zone_dt.editable)
        self.assertTrue(self.zone_span.editable)

        self.assertFalse(self.zone_overhang.editable)
        self.assertFalse(self.zone_overall_width.editable)
        self.assertFalse(self.zone_bracing.editable)

    def test_hit_testing(self):
        """Verify point hit testing on dimension zones."""
        point_inside = MagicMock(x=lambda: 150, y=lambda: 110)
        point_outside = MagicMock(x=lambda: 50, y=lambda: 50)

        self.assertTrue(self.zone_cw.contains(point_inside))
        self.assertFalse(self.zone_cw.contains(point_outside))

    def test_prefill_formatting(self):
        """Verify unit-aware pre-fill in InlineCADEditor."""
        parent = MockQWidget()
        editor_cw = InlineCADEditor(parent, self.zone_cw)
        self.assertEqual(editor_cw.text(), "10.5")

        editor_dt = InlineCADEditor(parent, self.zone_dt)
        self.assertEqual(editor_dt.text(), "200")

        editor_span = InlineCADEditor(parent, self.zone_span)
        self.assertEqual(editor_span.text(), "35")

        editor_gs = InlineCADEditor(parent, self.zone_gs)
        self.assertEqual(editor_gs.text(), "2.75")

    def test_commit_valid_values(self):
        """Verify successful commits for valid numerical inputs."""
        parent = MockQWidget()
        editor = InlineCADEditor(parent, self.zone_cw)
        editor.setText("12.25")

        committed_args = []
        editor.committed.connect(lambda k, v: committed_args.append((k, v)))

        editor.commit()
        self.assertEqual(len(committed_args), 1)
        self.assertEqual(committed_args[0], (KEY_CARRIAGEWAY_WIDTH, "12.25"))

    def test_commit_deck_thickness_mm(self):
        """Verify deck thickness commits cleanly in mm units."""
        parent = MockQWidget()
        editor = InlineCADEditor(parent, self.zone_dt)
        editor.setText("225")

        committed_args = []
        editor.committed.connect(lambda k, v: committed_args.append((k, v)))

        editor.commit()
        self.assertEqual(len(committed_args), 1)
        self.assertEqual(committed_args[0], (KEY_TS_DECK_THICKNESS, "225"))

    def test_commit_rejection_invalid_values(self):
        """Verify that empty, non-numeric, zero, negative, or NaN values cancel cleanly."""
        for invalid_val in ["", "   ", "abc", "-5", "0", "-0.01", "NaN", "inf"]:
            parent = MockQWidget()
            editor = InlineCADEditor(parent, self.zone_span)
            editor.setText(invalid_val)

            committed_args = []
            cancelled_calls = []
            editor.committed.connect(lambda k, v: committed_args.append((k, v)))
            editor.cancelled.connect(lambda: cancelled_calls.append(True))

            editor.commit()
            self.assertEqual(len(committed_args), 0, f"Expected rejection for '{invalid_val}'")
            self.assertEqual(len(cancelled_calls), 1, f"Expected cancellation for '{invalid_val}'")

    def test_escape_cancellation(self):
        """Verify pressing Escape triggers cancel and does not commit."""
        parent = MockQWidget()
        editor = InlineCADEditor(parent, self.zone_cw)
        editor.setText("99.9")

        committed_args = []
        cancelled_calls = []
        editor.committed.connect(lambda k, v: committed_args.append((k, v)))
        editor.cancelled.connect(lambda: cancelled_calls.append(True))

        mock_event = MagicMock()
        mock_event.key = lambda: mock_qt.Qt.Key_Escape
        editor.keyPressEvent(mock_event)

        self.assertEqual(len(committed_args), 0)
        self.assertEqual(len(cancelled_calls), 1)
        self.assertTrue(mock_event.accept.called)

    def test_double_commit_prevention(self):
        """Verify that double commit/finish triggers only execute once."""
        parent = MockQWidget()
        editor = InlineCADEditor(parent, self.zone_cw)
        editor.setText("11.5")

        committed_args = []
        editor.committed.connect(lambda k, v: committed_args.append((k, v)))

        editor.commit()
        editor.commit()  # Second call (e.g. from editingFinished after Enter)
        editor.cancel()  # Subsequent cancel

        self.assertEqual(len(committed_args), 1)

    def test_left_dock_flash_field(self):
        """Verify flash_field applies highlight stylesheet and restores correctly."""
        dock = MagicMock()
        mock_line_edit = MockQWidget()
        mock_line_edit.setStyleSheet("QLineEdit { background: white; }")
        dock._w = MagicMock(return_value=mock_line_edit)

        # Import the flash_field logic
        from osdagbridge.desktop.ui.docks.input_dock import InputDock
        InputDock.flash_field(dock, KEY_CARRIAGEWAY_WIDTH)

        self.assertIn("#E8F5D0", mock_line_edit.styleSheet())
        self.assertIn("#90AF13", mock_line_edit.styleSheet())

        # Simulate timeout on the created timer
        self.assertTrue(hasattr(mock_line_edit, "_flash_timer"))
        timer = mock_line_edit._flash_timer
        if hasattr(timer.timeout, 'connect') and timer.timeout.connect.call_args:
            callback = timer.timeout.connect.call_args[0][0]
            callback()
        # Verify original stylesheet restored
        self.assertEqual(mock_line_edit.styleSheet(), "QLineEdit { background: white; }")

    def test_template_page_cad_param_edited_handler(self):
        """Verify template_page._on_cad_param_edited_from_canvas syncs input_dict, dock, and CAD."""
        page = MagicMock()
        page.input_dict = {
            KEY_CARRIAGEWAY_WIDTH: "10.5",
            KEY_SPAN: "35.0",
        }
        mock_widget = MockQWidget()
        page.input_dock = MagicMock()
        page.input_dock._w = MagicMock(return_value=mock_widget)
        page.cad_comp_widget = MagicMock()

        from osdagbridge.desktop.ui.template_page import CustomWindow
        CustomWindow._on_cad_param_edited_from_canvas(page, KEY_CARRIAGEWAY_WIDTH, "12.0")

        # Verify input_dict updated
        self.assertEqual(page.input_dict[KEY_CARRIAGEWAY_WIDTH], 12.0)
        # Verify dock widget text updated
        self.assertEqual(mock_widget.text(), "12.0")
        # Verify flash_field called
        page.input_dock.flash_field.assert_called_with(KEY_CARRIAGEWAY_WIDTH)
        # Verify CAD update called
        page.cad_comp_widget.update_from_osdag_inputs.assert_called_with(page.input_dict, force=True)

    def test_escape_strict_cancellation_no_commit_on_focus_loss(self):
        """Verify pressing Escape sets _is_closed and prevents any subsequent focusOutEvent from committing."""
        parent = MockQWidget()
        editor = InlineCADEditor(parent, self.zone_cw)
        editor.setText("25.0")

        committed_args = []
        cancelled_calls = []
        editor.committed.connect(lambda k, v: committed_args.append((k, v)))
        editor.cancelled.connect(lambda: cancelled_calls.append(True))

        # 1. Simulate pressing Escape
        mock_event = MagicMock()
        mock_event.key = lambda: mock_qt.Qt.Key_Escape
        editor.keyPressEvent(mock_event)

        self.assertTrue(editor._is_closed)
        self.assertTrue(editor._is_cancelled)
        self.assertEqual(len(cancelled_calls), 1)
        self.assertEqual(len(committed_args), 0)

        # 2. Simulate subsequent focusOutEvent or editingFinished after Escape
        mock_focus_event = MagicMock()
        editor.focusOutEvent(mock_focus_event)
        editor._on_editing_finished()

        # Verify commit was never invoked
        self.assertEqual(len(committed_args), 0)

    def test_unit_stripping_and_parsing(self):
        """Verify user input with unit suffixes ('mm', 'm') is parsed and committed cleanly."""
        parent = MockQWidget()

        # Test mm input
        editor_dt = InlineCADEditor(parent, self.zone_dt)
        editor_dt.setText("300 mm")
        committed_dt = []
        editor_dt.committed.connect(lambda k, v: committed_dt.append((k, v)))
        editor_dt.commit()
        self.assertEqual(committed_dt, [(KEY_TS_DECK_THICKNESS, "300")])

        # Test m input
        editor_gs = InlineCADEditor(parent, self.zone_gs)
        editor_gs.setText("2.2 m")
        committed_gs = []
        editor_gs.committed.connect(lambda k, v: committed_gs.append((k, v)))
        editor_gs.commit()
        self.assertEqual(committed_gs, [(KEY_TS_GIRDER_SPACING, "2.2")])

    def test_deck_thickness_edit_preservation_in_solve_extend(self):
        """Verify editing Deck Thickness to 300 mm does not revert to 250 mm default on solve_extend."""
        from osdagbridge.core.bridge_types.plate_girder.defaults import solve_extend_basic_input_dict
        from osdagbridge.core.utils.common import KEY_CB_WIDTH

        basic_dict = {
            KEY_CARRIAGEWAY_WIDTH: 7.5,
            KEY_CB_WIDTH: 0.5,
            KEY_SPAN: 25.0,
            KEY_TS_DECK_THICKNESS: 300.0,
        }
        solve_extend_basic_input_dict(basic_dict)
        self.assertEqual(basic_dict[KEY_TS_DECK_THICKNESS], 300.0)

    def test_girder_spacing_edit_preservation_and_overhang_recalc(self):
        """Verify editing Girder Spacing to 2.2 m preserves spacing and recalculates deck overhang."""
        from osdagbridge.core.bridge_types.plate_girder.defaults import solve_extend_basic_input_dict
        from osdagbridge.core.utils.common import KEY_CB_WIDTH, KEY_TS_NO_OF_GIRDERS

        basic_dict = {
            KEY_CARRIAGEWAY_WIDTH: 7.5,
            KEY_CB_WIDTH: 0.5,
            KEY_SPAN: 25.0,
            KEY_TS_NO_OF_GIRDERS: 4,
            KEY_TS_GIRDER_SPACING: 2.2,
        }
        solve_extend_basic_input_dict(basic_dict, changed_field="spacing")
        # Girder spacing preserved
        self.assertEqual(basic_dict[KEY_TS_GIRDER_SPACING], 2.2)
        # Overhang recalculated: overall_width = 7.5 + 2*0.5 = 8.5. (8.5 - 3 * 2.2) / 2 = 0.95 m
        self.assertAlmostEqual(basic_dict[KEY_TS_DECK_OVERHANG], 0.95, places=2)

    def test_cad_dual_view_update_from_osdag_inputs_deck_thickness_and_spacing(self):
        """Verify cad_dual_view maps and pushes deck_thickness and girder_spacing to child widgets."""
        from osdagbridge.desktop.ui.docks.cad_dual_view import BridgeDualCADWidget
        dual = BridgeDualCADWidget()
        dual.cross_section_widget = MagicMock()
        dual.top_view_widget = MagicMock()

        test_inputs = {
            KEY_SPAN: 25.0,
            KEY_CARRIAGEWAY_WIDTH: 7.5,
            KEY_TS_GIRDER_SPACING: 2.2,
            KEY_TS_DECK_THICKNESS: 300.0,
            KEY_TS_DECK_OVERHANG: 0.95,
            KEY_TS_NO_OF_GIRDERS: 4,
        }
        dual.update_from_osdag_inputs(test_inputs, force=True)

        # Verify cross section received girder_spacing (in mm) and deck_thickness (in mm)
        self.assertTrue(dual.cross_section_widget.update_params.called)
        cs_call_args = dual.cross_section_widget.update_params.call_args[0][0]
        self.assertEqual(cs_call_args['deck_thickness'], 300.0)
        self.assertEqual(cs_call_args['girder_spacing'], 2200.0)
        self.assertEqual(cs_call_args['deck_overhang'], 950.0)

        # Verify top view received span_length (in mm) and girder_spacing (in mm)
        self.assertTrue(dual.top_view_widget.update_params.called)
        tv_call_args = dual.top_view_widget.update_params.call_args[0][0]
        self.assertEqual(tv_call_args['span_length'], 25000.0)
        self.assertEqual(tv_call_args['girder_spacing'], 2200.0)


if __name__ == '__main__':
    unittest.main()
