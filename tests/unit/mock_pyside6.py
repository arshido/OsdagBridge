"""
Shared PySide6 and Qt mock utilities for unit tests.
Provides complete mock implementations of QWidget, QLineEdit, QComboBox,
QTabWidget, QDialog, QRectF, and Qt signals when PySide6 is not present in the environment.
"""

from __future__ import annotations
import sys
from unittest.mock import MagicMock


class MockSignal:
    def __init__(self, *types):
        self._name = None

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        attr = f"__signal_{self._name}"
        if not hasattr(instance, attr):
            setattr(instance, attr, _BoundMockSignal())
        return getattr(instance, attr)


class _BoundMockSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, cb):
        self._callbacks.append(cb)

    def emit(self, *args):
        for cb in list(self._callbacks):
            cb(*args)


class MockQRectF:
    def __init__(self, x=0, y=0, w=100, h=30):
        self._x, self._y, self._w, self._h = x, y, w, h

    def x(self): return self._x
    def y(self): return self._y
    def width(self): return self._w
    def height(self): return self._h
    def center(self):
        return MagicMock(x=lambda: self._x + self._w / 2, y=lambda: self._y + self._h / 2)

    def contains(self, point):
        px = point.x() if hasattr(point, 'x') else point[0]
        py = point.y() if hasattr(point, 'y') else point[1]
        return self._x <= px <= self._x + self._w and self._y <= py <= self._y + self._h

    def adjusted(self, dx1, dy1, dx2, dy2):
        return MockQRectF(self._x + dx1, self._y + dy1, self._w + dx2 - dx1, self._h + dy2 - dy1)


class MockQWidget:
    AdjustToMinimumContentsLengthWithIcon = 1

    def __init__(self, parent=None):
        self._parent = parent
        self._children = []
        if parent is not None and hasattr(parent, '_children'):
            parent._children.append(self)
        self._object_name = ""
        self._stylesheet = ""
        self._text = ""
        self._enabled = True
        self._visible = True
        self._hidden = False
        self._properties = {}
        self.editingFinished = _BoundMockSignal()
        self.textEdited = _BoundMockSignal()
        self.textChanged = _BoundMockSignal()
        self.currentTextChanged = _BoundMockSignal()
        self.activated = _BoundMockSignal()
        self.clicked = _BoundMockSignal()

    def setObjectName(self, name): self._object_name = name
    def objectName(self): return self._object_name
    def parent(self): return self._parent
    def parentWidget(self): return self._parent
    def setStyleSheet(self, s): self._stylesheet = s
    def styleSheet(self): return self._stylesheet
    def isEnabled(self): return self._enabled
    def setEnabled(self, e): self._enabled = bool(e)
    def isVisible(self): return self._visible
    def setVisible(self, v):
        self._visible = bool(v)
        self._hidden = not bool(v)
    def isHidden(self): return self._hidden
    def show(self): self.setVisible(True)
    def hide(self): self.setVisible(False)
    def text(self): return self._text
    def setText(self, t): self._text = str(t)
    def currentText(self): return self._text
    def setCurrentText(self, t): self._text = str(t)
    def setFocus(self, reason=None): pass
    def selectAll(self): pass
    def deleteLater(self): pass
    def update(self): pass
    def setMouseTracking(self, b): pass
    def hasMouseTracking(self): return True
    def width(self): return 800
    def height(self): return 600
    def blockSignals(self, b): pass
    def setGeometry(self, x, y, w, h): pass
    def setAlignment(self, align): pass
    def setWindowFlags(self, f): pass
    def setAttribute(self, a, v=True): pass
    def setMinimumSize(self, w, h): pass
    def resize(self, w, h): pass
    def setSizeGripEnabled(self, b): pass
    def property(self, name): return self._properties.get(name)
    def setProperty(self, name, val): self._properties[name] = val
    def screen(self):
        m = MagicMock()
        m.availableGeometry.return_value.height.return_value = 1080
        return m

    def findChildren(self, cls, name=None):
        results = []
        def walk(w):
            for child in getattr(w, '_children', []):
                if name is None or getattr(child, '_object_name', '') == name:
                    results.append(child)
                walk(child)
        walk(self)
        return results

    def findChild(self, cls, name=None):
        matches = self.findChildren(cls, name)
        return matches[0] if matches else None


class MockQTabWidget(MockQWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tabs = []
        self._current_index = 0
        self._tab_enabled = {}

    def addTab(self, widget, label):
        self._tabs.append((widget, label))
        widget._parent = self
        idx = len(self._tabs) - 1
        self._tab_enabled[idx] = True
        return idx

    def count(self): return len(self._tabs)
    def widget(self, idx):
        if 0 <= idx < len(self._tabs):
            return self._tabs[idx][0]
        return None

    def indexOf(self, widget):
        for i, (w, _) in enumerate(self._tabs):
            if w is widget:
                return i
        return -1

    def currentIndex(self): return self._current_index
    def setCurrentIndex(self, idx): self._current_index = idx
    def isTabEnabled(self, idx): return self._tab_enabled.get(idx, True)
    def setTabEnabled(self, idx, enabled): self._tab_enabled[idx] = bool(enabled)
    def tabText(self, idx):
        if 0 <= idx < len(self._tabs):
            return self._tabs[idx][1]
        return ""


mock_qt = MagicMock()
mock_qt.Signal = MockSignal
mock_qt.QRectF = MockQRectF
mock_qt.QWidget = MockQWidget
mock_qt.QLineEdit = MockQWidget
mock_qt.QComboBox = MockQWidget
mock_qt.QCheckBox = MockQWidget
mock_qt.QRadioButton = MockQWidget
mock_qt.QTabWidget = MockQTabWidget
mock_qt.QDialog = MockQWidget
mock_qt.Qt.AlignCenter = 0
mock_qt.Qt.OtherFocusReason = 0
mock_qt.Qt.Key_Escape = 0x01000000
mock_qt.Qt.Key_Return = 0x01000004
mock_qt.Qt.Key_Enter = 0x01000005
mock_qt.__path__ = []
mock_qt.__file__ = 'mock_pyside6'

sys.modules['PySide6'] = mock_qt
sys.modules['PySide6.QtWidgets'] = mock_qt
sys.modules['PySide6.QtCore'] = mock_qt
sys.modules['PySide6.QtGui'] = mock_qt
sys.modules['PySide6.QtSvgWidgets'] = mock_qt
sys.modules['PySide6.QtNetwork'] = mock_qt

if 'openpyxl' not in sys.modules:
    sys.modules['openpyxl'] = MagicMock()
    sys.modules['openpyxl.styles'] = MagicMock()

sys.modules['osdagbridge.desktop.ui.cad_3d'] = MagicMock()
