"""
Unit tests for Additional Inputs Dialog Validation & State Management (Task II).
Tests strict key-driven mapping, pre-save custom validation,
empty/whitespace rejection, tab auto-switching, widget focus, border highlight,
modification tracking (is_modified), title bar asterisk, and unsaved changes close confirmation.
"""

from __future__ import annotations
import sys
import unittest
from unittest.mock import MagicMock, patch, call
from copy import deepcopy

from tests.unit.mock_pyside6 import (
    MockSignal,
    _BoundMockSignal,
    MockQRectF,
    MockQWidget,
    MockQTabWidget,
    mock_qt,
)

sys.path.insert(0, 'src')

from osdagbridge.core.utils.common import (
    KEY_CB_TYPE, KEY_CB_WIDTH, KEY_CB_HEIGHT, KEY_CB_LOAD,
    KEY_MD_TYPE, KEY_MD_WIDTH, KEY_MD_HEIGHT, KEY_MD_LOAD,
    KEY_RL_TYPE, KEY_RL_WIDTH, KEY_RL_HEIGHT, KEY_RL_LOAD_MODE, KEY_RL_LOAD_VALUE,
    KEY_WC_MATERIAL, KEY_WC_DENSITY,
    KEY_LL_FOOTPATH_PRESSURE_MODE, KEY_LL_FOOTPATH_PRESSURE_VALUE,
    KEY_SL_DEAD_LOAD_MODE, KEY_SL_DEAD_LOAD_VALUE,
    KEY_WL_GUST_FACTOR_MODE, KEY_WL_GUST_FACTOR_VALUE,
    KEY_DO_CAMBER_MODE, KEY_DO_CAMBER_VALUE,
    KEY_DESIGN_MODE, KEY_MP_GIRDER_DEPTH, KEY_MP_GIRDER_TOP_FLANGE_WIDTH, KEY_MP_GIRDER_BOTTOM_FLANGE_WIDTH,
    KEY_ADD_CUSTOM_LOAD_CASE, KEY_ADD_CUSTOM_LOAD_CASE_NAME,
    ADDITIONAL_INPUTS_PARAM_NAMES,
)
from osdagbridge.desktop.ui.dialogs.additional_input.additional_inputs import AdditionalInputs
from osdagbridge.desktop.ui.dialogs.custom_messagebox import MessageBoxType


class MockCloseEvent:
    def __init__(self):
        self.accepted = False
        self.ignored = False

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


class MockKeyEvent:
    def __init__(self, key):
        self._key = key
        self.accepted = False

    def key(self):
        return self._key

    def accept(self):
        self.accepted = True


class TestAdditionalInputsValidation(unittest.TestCase):
    """Test suite for Task II: Additional Inputs Validation & State Management."""

    def setUp(self):
        # Create a lightweight AdditionalInputs instance without full Qt GUI rendering
        with patch.object(AdditionalInputs, 'setupWrapper'), \
             patch.object(AdditionalInputs, 'init_ui'):
            self.dialog = AdditionalInputs()
            self.dialog.tabs = MockQTabWidget()
            self.dialog.title_bar = MagicMock()
            self.dialog.title_bar.setTitle = MagicMock()
            self.dialog.working_input_dict = {}
            self.dialog.default_input_dict = {}
            self.dialog._is_initializing = False
            self.dialog.is_modified = False
            self.dialog.findChildren = lambda cls: []

    def test_strict_key_driven_architecture(self):
        """Verify that central keys and parameter display names exist in common.py."""
        # Central keys must not be None or empty
        self.assertTrue(bool(KEY_ADD_CUSTOM_LOAD_CASE))
        self.assertTrue(bool(KEY_ADD_CUSTOM_LOAD_CASE_NAME))
        self.assertTrue(bool(KEY_CB_TYPE))
        self.assertTrue(bool(KEY_LL_FOOTPATH_PRESSURE_MODE))
        self.assertTrue(bool(KEY_LL_FOOTPATH_PRESSURE_VALUE))
        self.assertTrue(bool(KEY_SL_DEAD_LOAD_MODE))
        self.assertTrue(bool(KEY_SL_DEAD_LOAD_VALUE))
        self.assertTrue(bool(KEY_WL_GUST_FACTOR_MODE))
        self.assertTrue(bool(KEY_WL_GUST_FACTOR_VALUE))
        self.assertTrue(bool(KEY_DO_CAMBER_MODE))
        self.assertTrue(bool(KEY_DO_CAMBER_VALUE))

        # Check canonical parameter display names in mapping
        self.assertEqual(ADDITIONAL_INPUTS_PARAM_NAMES[KEY_CB_WIDTH], "Crash Barrier Width")
        self.assertEqual(ADDITIONAL_INPUTS_PARAM_NAMES[KEY_CB_HEIGHT], "Crash Barrier Height")
        self.assertEqual(ADDITIONAL_INPUTS_PARAM_NAMES[KEY_CB_LOAD], "Crash Barrier Load")
        self.assertEqual(ADDITIONAL_INPUTS_PARAM_NAMES[KEY_MD_WIDTH], "Median Width")
        self.assertEqual(ADDITIONAL_INPUTS_PARAM_NAMES[KEY_RL_WIDTH], "Railing Width")
        self.assertEqual(ADDITIONAL_INPUTS_PARAM_NAMES[KEY_WC_DENSITY], "Wearing Course Density")
        self.assertEqual(ADDITIONAL_INPUTS_PARAM_NAMES[KEY_LL_FOOTPATH_PRESSURE_VALUE], "Footpath Pressure")
        self.assertEqual(ADDITIONAL_INPUTS_PARAM_NAMES[KEY_SL_DEAD_LOAD_VALUE], "Dead Load for Seismic Force")
        self.assertEqual(ADDITIONAL_INPUTS_PARAM_NAMES[KEY_WL_GUST_FACTOR_VALUE], "Gust Factor")
        self.assertEqual(ADDITIONAL_INPUTS_PARAM_NAMES[KEY_DO_CAMBER_VALUE], "Camber")
        self.assertEqual(ADDITIONAL_INPUTS_PARAM_NAMES[KEY_ADD_CUSTOM_LOAD_CASE_NAME], "Custom Load Case Name")

    def test_is_modified_flag_and_title_asterisk(self):
        """Verify tracking of is_modified flag and title bar asterisk appending/clearing."""
        self.assertFalse(self.dialog.is_modified)

        # Set modified = True
        self.dialog.set_modified(True)
        self.assertTrue(self.dialog.is_modified)
        self.dialog.title_bar.setTitle.assert_called_with("Additional Inputs *")

        # Set modified = False
        self.dialog.set_modified(False)
        self.assertFalse(self.dialog.is_modified)
        self.dialog.title_bar.setTitle.assert_called_with("Additional Inputs")

        # Ignored during programmatic initialization
        self.dialog._is_initializing = True
        self.dialog.set_modified(True)
        self.assertFalse(self.dialog.is_modified)  # should not change
        self.dialog._is_initializing = False

    @patch('osdagbridge.desktop.ui.dialogs.additional_input.additional_inputs.CustomMessageBox')
    def test_pre_save_validation_empty_custom_field(self, mock_msg_box):
        """Verify that an empty custom field halts save and displays exact error message."""
        # Set up a top-level tab
        tab_page = MockQWidget(parent=self.dialog.tabs)
        self.dialog.tabs.addTab(tab_page, "Loading")

        # Create mode combo set to "Custom" and empty value line edit
        combo = MockQWidget(parent=tab_page)
        combo.setObjectName(KEY_LL_FOOTPATH_PRESSURE_MODE)
        combo.setCurrentText("Custom")

        val_input = MockQWidget(parent=tab_page)
        val_input.setObjectName(KEY_LL_FOOTPATH_PRESSURE_VALUE)
        val_input.setText("")

        self.dialog.findChild = lambda cls, name: {
            KEY_LL_FOOTPATH_PRESSURE_MODE: combo,
            KEY_LL_FOOTPATH_PRESSURE_VALUE: val_input,
        }.get(name)

        # Attempt validation
        is_valid = self.dialog._validate_custom_inputs()
        self.assertFalse(is_valid)

        # Verify exact error message: "Error: The custom value for '[Parameter Name]' cannot be empty."
        mock_msg_box.assert_called_once()
        _, kwargs = mock_msg_box.call_args
        self.assertEqual(kwargs.get("title"), "Error")
        self.assertEqual(kwargs.get("text"), "Error: The custom value for 'Footpath Pressure' cannot be empty.")
        self.assertEqual(kwargs.get("dialogType"), MessageBoxType.Critical)

    @patch('osdagbridge.desktop.ui.dialogs.additional_input.additional_inputs.CustomMessageBox')
    def test_pre_save_validation_whitespace_only_field(self, mock_msg_box):
        """Verify that whitespace-only input is rejected cleanly."""
        tab_page = MockQWidget(parent=self.dialog.tabs)
        self.dialog.tabs.addTab(tab_page, "Loading")

        combo = MockQWidget(parent=tab_page)
        combo.setObjectName(KEY_SL_DEAD_LOAD_MODE)
        combo.setCurrentText("Custom")

        val_input = MockQWidget(parent=tab_page)
        val_input.setObjectName(KEY_SL_DEAD_LOAD_VALUE)
        val_input.setText("     ")  # Whitespace only

        self.dialog.findChild = lambda cls, name: {
            KEY_SL_DEAD_LOAD_MODE: combo,
            KEY_SL_DEAD_LOAD_VALUE: val_input,
        }.get(name)

        is_valid = self.dialog._validate_custom_inputs()
        self.assertFalse(is_valid)
        mock_msg_box.assert_called_once()
        _, kwargs = mock_msg_box.call_args
        self.assertEqual(kwargs.get("text"), "Error: The custom value for 'Dead Load for Seismic Force' cannot be empty.")

    @patch('osdagbridge.desktop.ui.dialogs.additional_input.additional_inputs.CustomMessageBox')
    def test_pre_save_validation_success_and_save_commit(self, mock_msg_box):
        """Verify that valid custom values allow saving and reset is_modified."""
        tab_page = MockQWidget(parent=self.dialog.tabs)
        self.dialog.tabs.addTab(tab_page, "Loading")

        combo = MockQWidget(parent=tab_page)
        combo.setObjectName(KEY_LL_FOOTPATH_PRESSURE_MODE)
        combo.setCurrentText("Custom")

        val_input = MockQWidget(parent=tab_page)
        val_input.setObjectName(KEY_LL_FOOTPATH_PRESSURE_VALUE)
        val_input.setText("4.5")  # Valid non-empty value

        self.dialog.findChild = lambda cls, name: {
            KEY_LL_FOOTPATH_PRESSURE_MODE: combo,
            KEY_LL_FOOTPATH_PRESSURE_VALUE: val_input,
        }.get(name)

        self.dialog.working_input_dict = {
            KEY_LL_FOOTPATH_PRESSURE_MODE: "Custom",
            KEY_LL_FOOTPATH_PRESSURE_VALUE: "4.5"
        }
        self.dialog.set_modified(True)

        # Trigger _save_inputs
        with patch.object(self.dialog, '_save_stiffener_member_data'):
            self.dialog._save_inputs()

        # Check committed to default_input_dict
        self.assertEqual(self.dialog.default_input_dict.get(KEY_LL_FOOTPATH_PRESSURE_VALUE), "4.5")
        # is_modified should be reset to False
        self.assertFalse(self.dialog.is_modified)
        self.dialog.title_bar.setTitle.assert_called_with("Additional Inputs")

    def test_non_custom_mode_does_not_block_save(self):
        """Verify that empty fields in non-custom modes (e.g. Automatic) do not block save."""
        tab_page = MockQWidget(parent=self.dialog.tabs)
        self.dialog.tabs.addTab(tab_page, "Loading")

        combo = MockQWidget(parent=tab_page)
        combo.setObjectName(KEY_LL_FOOTPATH_PRESSURE_MODE)
        combo.setCurrentText("As per IRC 6")  # Non-custom

        val_input = MockQWidget(parent=tab_page)
        val_input.setObjectName(KEY_LL_FOOTPATH_PRESSURE_VALUE)
        val_input.setText("")  # Empty, but mode is not Custom

        self.dialog.findChild = lambda cls, name: {
            KEY_LL_FOOTPATH_PRESSURE_MODE: combo,
            KEY_LL_FOOTPATH_PRESSURE_VALUE: val_input,
        }.get(name)

        is_valid = self.dialog._validate_custom_inputs()
        self.assertTrue(is_valid)

    def test_disabled_tab_custom_field_ignored(self):
        """Verify that fields in disabled tabs are ignored during pre-save validation."""
        tab_page = MockQWidget(parent=self.dialog.tabs)
        tab_idx = self.dialog.tabs.addTab(tab_page, "Disabled Tab")
        self.dialog.tabs.setTabEnabled(tab_idx, False)  # Tab is disabled

        combo = MockQWidget(parent=tab_page)
        combo.setObjectName(KEY_WL_GUST_FACTOR_MODE)
        combo.setCurrentText("Custom")

        val_input = MockQWidget(parent=tab_page)
        val_input.setObjectName(KEY_WL_GUST_FACTOR_VALUE)
        val_input.setText("")

        self.dialog.findChild = lambda cls, name: {
            KEY_WL_GUST_FACTOR_MODE: combo,
            KEY_WL_GUST_FACTOR_VALUE: val_input,
        }.get(name)

        is_valid = self.dialog._validate_custom_inputs()
        self.assertTrue(is_valid)

    def test_ux_tab_switching_and_field_focus(self):
        """Verify UX Polish: auto-switch QTabWidget and set focus to the invalid field."""
        # Top-level tab 0
        tab0 = MockQWidget(parent=self.dialog.tabs)
        self.dialog.tabs.addTab(tab0, "Tab 0")

        # Top-level tab 1 containing an inner sub-tab
        tab1 = MockQWidget(parent=self.dialog.tabs)
        self.dialog.tabs.addTab(tab1, "Tab 1")

        inner_tabs = MockQTabWidget(parent=tab1)
        sub_page0 = MockQWidget(parent=inner_tabs)
        inner_tabs.addTab(sub_page0, "Sub 0")

        sub_page1 = MockQWidget(parent=inner_tabs)
        inner_tabs.addTab(sub_page1, "Sub 1")

        # Field in sub_page1
        invalid_field = MockQWidget(parent=sub_page1)
        invalid_field.setFocus = MagicMock()

        # Currently showing top tab 0, inner tab 0
        self.dialog.tabs.setCurrentIndex(0)
        inner_tabs.setCurrentIndex(0)

        self.dialog._activate_and_focus_field(invalid_field)

        # Both top and inner tabs should have switched to tab 1 and sub 1
        self.assertEqual(self.dialog.tabs.currentIndex(), 1)
        self.assertEqual(inner_tabs.currentIndex(), 1)
        invalid_field.setFocus.assert_called_once()
        self.assertIn("#F44336", invalid_field.styleSheet())

    @patch('osdagbridge.desktop.ui.dialogs.additional_input.additional_inputs.CustomMessageBox')
    def test_close_event_unmodified_closes_without_prompt(self, mock_msg_box):
        """Verify that closing an unmodified dialog accepts event without warning."""
        self.dialog.is_modified = False
        event = MockCloseEvent()
        self.dialog.closeEvent(event)

        self.assertTrue(event.accepted)
        self.assertFalse(event.ignored)
        mock_msg_box.assert_not_called()

    @patch('osdagbridge.desktop.ui.dialogs.additional_input.additional_inputs.CustomMessageBox')
    def test_close_event_modified_cancel_aborts_close(self, mock_msg_box):
        """Verify that Cancel on unsaved changes keeps the dialog open."""
        self.dialog.is_modified = True
        mock_box_inst = MagicMock()
        mock_box_inst.exec.return_value = "Cancel"
        mock_msg_box.return_value = mock_box_inst

        event = MockCloseEvent()
        self.dialog.closeEvent(event)

        mock_msg_box.assert_called_once()
        _, kwargs = mock_msg_box.call_args
        self.assertEqual(kwargs.get("title"), "Warning")
        self.assertEqual(kwargs.get("text"), "You have unsaved changes. Closing this window will discard them. Do you want to proceed?")
        self.assertEqual(kwargs.get("buttons"), ["Discard", "Cancel"])

        self.assertFalse(event.accepted)
        self.assertTrue(event.ignored)
        self.assertTrue(self.dialog.is_modified)

    @patch('osdagbridge.desktop.ui.dialogs.additional_input.additional_inputs.CustomMessageBox')
    def test_close_event_modified_discard_reverts_and_closes(self, mock_msg_box):
        """Verify that Discard on unsaved changes reverts defaults and accepts close."""
        self.dialog.is_modified = True
        self.dialog.default_input_dict = {"param": "original_val"}
        self.dialog.working_input_dict = {"param": "modified_val"}

        mock_box_inst = MagicMock()
        mock_box_inst.exec.return_value = "Discard"
        mock_msg_box.return_value = mock_box_inst

        with patch.object(self.dialog, 'set_defaults') as mock_set_defaults:
            event = MockCloseEvent()
            self.dialog.closeEvent(event)

            self.assertTrue(event.accepted)
            self.assertFalse(event.ignored)
            self.assertFalse(self.dialog.is_modified)
            self.assertEqual(self.dialog.working_input_dict.get("param"), "original_val")
            mock_set_defaults.assert_called_once()

    def test_escape_key_and_reject_route_through_close(self):
        """Verify that Escape key and reject() call self.close()."""
        self.dialog.close = MagicMock()
        key_event = MockKeyEvent(mock_qt.Qt.Key_Escape)

        self.dialog.keyPressEvent(key_event)
        self.dialog.close.assert_called_once()
        self.assertTrue(key_event.accepted)

        self.dialog.close.reset_mock()
        self.dialog.reject()
        self.dialog.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
