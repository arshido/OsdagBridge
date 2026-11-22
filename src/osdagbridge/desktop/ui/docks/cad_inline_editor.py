"""
Inline CAD Parameter Editor for OsdagBridge CAD canvases.
Allows inline double-click editing of dimension labels on CAD widgets.
"""

import math
from dataclasses import dataclass
from typing import Optional

from PySide6.QtWidgets import QLineEdit, QWidget, QToolTip
from PySide6.QtCore import Qt, QRectF, QPoint, QPointF, Signal
from PySide6.QtGui import QCursor, QFont, QPen, QColor, QPainter


@dataclass
class DimensionLabelZone:
    rect: QRectF
    param_key: str
    editable: bool
    unit: str  # "m" or "mm"
    current_value: float  # Display/input unit numeric value
    label_name: str  # e.g., "Carriageway Width"

    def contains(self, point: QPointF | QPoint) -> bool:
        return self.rect.contains(point)


class InlineCADEditor(QLineEdit):
    """
    Temporary QLineEdit overlay positioned on top of an editable dimension label.
    Supports:
      - Unit-aware pre-fill and parsing (meters vs millimeters).
      - Enter to commit.
      - Escape to cancel and restore original label.
      - Click outside (focus loss) to commit.
      - Input validation rejecting empty, non-numeric, or <= 0 values.
      - Guard against double-firing editingFinished / commit.
    """
    committed = Signal(str, str)  # (param_key, new_value_str)
    cancelled = Signal()

    def __init__(self, parent: QWidget, zone: DimensionLabelZone):
        super().__init__(parent)
        self.zone = zone
        self._is_committed = False
        self._is_cancelled = False
        self._is_closed = False

        # Format initial value cleanly
        val = zone.current_value
        if zone.unit == "mm":
            initial_text = f"{val:.0f}" if val.is_integer() else f"{val:.2f}".rstrip('0').rstrip('.')
        else:
            initial_text = f"{val:.3f}".rstrip('0').rstrip('.')
            if initial_text.endswith('.'):
                initial_text = initial_text[:-1]

        self.setText(initial_text)

        # High-contrast, clean CAD styling matching Osdag's primary color (#90AF13)
        self.setStyleSheet("""
            QLineEdit {
                background-color: #FFFFFF;
                color: #111111;
                border: 2px solid #90AF13;
                border-radius: 3px;
                font-family: Arial;
                font-size: 11px;
                font-weight: bold;
                padding: 1px 4px;
                selection-background-color: #90AF13;
                selection-color: #FFFFFF;
            }
        """)

        # Calculate geometry: overlay on top of zone.rect with comfortable min size
        rect = zone.rect
        min_width = max(int(rect.width()) + 16, 80)
        min_height = max(int(rect.height()) + 8, 24)
        center_x = int(rect.center().x())
        center_y = int(rect.center().y())

        x = center_x - min_width // 2
        y = center_y - min_height // 2

        # Clamp to parent widget bounds
        x = max(2, min(x, max(2, parent.width() - min_width - 2)))
        y = max(2, min(y, max(2, parent.height() - min_height - 2)))

        self.setGeometry(x, y, min_width, min_height)
        self.setAlignment(Qt.AlignCenter)

        # Show, focus and select all text
        self.show()
        self.setFocus(Qt.OtherFocusReason)
        self.selectAll()

        # Connect signals
        self.editingFinished.connect(self._on_editing_finished)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._is_closed = True
            self._is_cancelled = True
            self.cancel()
            event.accept()
            return
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.commit()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        if (
            getattr(self, "_is_closed", False)
            or getattr(self, "_is_cancelled", False)
            or getattr(self, "_is_committed", False)
        ):
            event.accept()
            return
        self.commit()
        super().focusOutEvent(event)

    def _on_editing_finished(self):
        if (
            getattr(self, "_is_closed", False)
            or getattr(self, "_is_cancelled", False)
            or getattr(self, "_is_committed", False)
        ):
            return
        self.commit()

    def commit(self):
        if (
            getattr(self, "_is_committed", False)
            or getattr(self, "_is_cancelled", False)
            or getattr(self, "_is_closed", False)
        ):
            return

        raw_text = self.text().strip()
        cleaned_text = raw_text.lower()
        if cleaned_text.endswith("mm"):
            cleaned_text = cleaned_text[:-2].strip()
        elif cleaned_text.endswith("m"):
            cleaned_text = cleaned_text[:-1].strip()

        try:
            val = float(cleaned_text)
            if val <= 0 or math.isnan(val) or math.isinf(val):
                self.cancel()
                return

            self._is_committed = True
            self._is_closed = True

            if self.zone.unit == "mm":
                formatted_value = f"{val:.0f}" if val.is_integer() else f"{val:.2f}".rstrip('0').rstrip('.')
            else:
                formatted_value = f"{val:.3f}".rstrip('0').rstrip('.')
                if formatted_value.endswith('.'):
                    formatted_value = formatted_value[:-1]

            self.committed.emit(self.zone.param_key, formatted_value)
            self._cleanup()
        except (ValueError, TypeError):
            self.cancel()
            return

    def cancel(self):
        if getattr(self, "_is_committed", False):
            return
        self._is_closed = True
        self._is_cancelled = True
        self.cancelled.emit()
        self._cleanup()

    def _cleanup(self):
        self._is_closed = True
        self.hide()
        parent = self.parentWidget() or (self.parent() if hasattr(self, 'parent') else None)
        self.deleteLater()
        if parent:
            if hasattr(parent, "setFocus"):
                parent.setFocus(Qt.OtherFocusReason)
            if hasattr(parent, "update"):
                parent.update()
