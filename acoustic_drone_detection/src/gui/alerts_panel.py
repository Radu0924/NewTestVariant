"""
Alerts Panel Widget

Detection alerts and event log display:
- Real-time detection alerts
- Event history log
- Alert filtering and search
- Export functionality
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QComboBox, QLineEdit, QLabel,
    QGroupBox, QFrame, QScrollArea, QSplitter, QMenu
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont, QAction
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class AlertEvent:
    """Detection alert event."""
    timestamp: float
    track_id: Optional[int]
    azimuth: float
    elevation: float
    distance: float
    confidence: float
    classification: str
    threat_level: str
    snr: float = 0.0


class AlertCard(QFrame):
    """Individual alert card widget."""

    clicked = pyqtSignal(int)  # track_id

    # Colors for threat levels
    COLORS = {
        'high': '#ff4444',
        'medium': '#ffaa00',
        'low': '#44ff88'
    }

    def __init__(self, alert: AlertEvent, parent=None):
        super().__init__(parent)

        self._alert = alert
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)

        # Set border color based on threat level
        color = self.COLORS.get(self._alert.threat_level, '#888888')
        self.setStyleSheet(f"""
            AlertCard {{
                border: 2px solid {color};
                border-radius: 5px;
                background-color: #2a2a3e;
                padding: 5px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()

        # Threat level indicator
        threat_label = QLabel(self._alert.threat_level.upper())
        threat_label.setStyleSheet(f"""
            color: {color};
            font-weight: bold;
            font-size: 11px;
        """)
        header.addWidget(threat_label)

        # Classification
        class_label = QLabel(self._alert.classification)
        class_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        header.addWidget(class_label)

        header.addStretch()

        # Time
        time_str = datetime.fromtimestamp(self._alert.timestamp).strftime("%H:%M:%S")
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: #888888; font-size: 10px;")
        header.addWidget(time_label)

        layout.addLayout(header)

        # Details
        details = QHBoxLayout()

        # Position
        pos_text = f"Az: {self._alert.azimuth:.1f}deg  El: {self._alert.elevation:.1f}deg"
        pos_label = QLabel(pos_text)
        pos_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        details.addWidget(pos_label)

        # Distance
        dist_label = QLabel(f"{self._alert.distance:.1f}m")
        dist_label.setStyleSheet("color: #88ff88; font-size: 10px;")
        details.addWidget(dist_label)

        # Confidence
        conf_label = QLabel(f"{self._alert.confidence:.0%}")
        conf_label.setStyleSheet("color: #8888ff; font-size: 10px;")
        details.addWidget(conf_label)

        details.addStretch()

        layout.addLayout(details)

        # Track ID if available
        if self._alert.track_id is not None:
            track_label = QLabel(f"Track #{self._alert.track_id}")
            track_label.setStyleSheet("color: #666666; font-size: 9px;")
            layout.addWidget(track_label)

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._alert.track_id is not None:
                self.clicked.emit(self._alert.track_id)
        super().mousePressEvent(event)


class AlertsPanel(QWidget):
    """
    Alerts and detection log panel.

    Shows recent detection alerts and maintains event history.
    """

    # Signals
    alert_selected = pyqtSignal(int)  # track_id
    export_requested = pyqtSignal()

    def __init__(self, parent=None, max_alerts: int = 100):
        super().__init__(parent)

        self._max_alerts = max_alerts
        self._alerts: deque = deque(maxlen=max_alerts)
        self._filtered_alerts: List[AlertEvent] = []
        self._filter_threat: str = "all"
        self._filter_class: str = "all"
        self._filter_text: str = ""

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with filters
        header = QHBoxLayout()

        header.addWidget(QLabel("Alerts"))

        # Threat filter
        self._threat_combo = QComboBox()
        self._threat_combo.addItems(["All", "High", "Medium", "Low"])
        self._threat_combo.currentIndexChanged.connect(self._apply_filter)
        header.addWidget(self._threat_combo)

        # Class filter
        self._class_combo = QComboBox()
        self._class_combo.addItems(["All Classes"])
        self._class_combo.currentIndexChanged.connect(self._apply_filter)
        header.addWidget(self._class_combo)

        # Search
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search...")
        self._search_edit.textChanged.connect(self._apply_filter)
        header.addWidget(self._search_edit)

        layout.addLayout(header)

        # Splitter for cards and table
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Alert cards (recent)
        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)

        cards_label = QLabel("Recent Detections")
        cards_label.setStyleSheet("font-weight: bold; color: #00ff88;")
        cards_layout.addWidget(cards_label)

        # Scroll area for cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(5)
        self._cards_layout.addStretch()

        scroll.setWidget(self._cards_container)
        cards_layout.addWidget(scroll)

        splitter.addWidget(cards_widget)

        # Event log table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)

        table_label = QLabel("Event Log")
        table_label.setStyleSheet("font-weight: bold; color: #00ff88;")
        table_layout.addWidget(table_label)

        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels([
            "Time", "Track", "Class", "Threat",
            "Azimuth", "Distance", "Confidence", "SNR"
        ])

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)

        table_layout.addWidget(self._table)

        splitter.addWidget(table_widget)

        # Set splitter sizes
        splitter.setSizes([200, 300])

        layout.addWidget(splitter)

        # Footer with stats and actions
        footer = QHBoxLayout()

        self._stats_label = QLabel("0 alerts")
        self._stats_label.setStyleSheet("color: #888888;")
        footer.addWidget(self._stats_label)

        footer.addStretch()

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self.clear_alerts)
        footer.addWidget(self._clear_btn)

        self._export_btn = QPushButton("Export...")
        self._export_btn.clicked.connect(self.export_requested.emit)
        footer.addWidget(self._export_btn)

        layout.addLayout(footer)

    def add_alert(self, alert: AlertEvent) -> None:
        """Add a new alert."""
        self._alerts.appendleft(alert)

        # Update class filter options
        current_classes = {a.classification for a in self._alerts}
        self._update_class_filter(current_classes)

        # Apply filter and update display
        self._apply_filter()

    def _update_class_filter(self, classes: set) -> None:
        """Update classification filter options."""
        current = self._class_combo.currentText()
        self._class_combo.clear()
        self._class_combo.addItem("All Classes")
        for cls in sorted(classes):
            self._class_combo.addItem(cls)

        # Restore selection
        idx = self._class_combo.findText(current)
        if idx >= 0:
            self._class_combo.setCurrentIndex(idx)

    def _apply_filter(self) -> None:
        """Apply current filters."""
        # Get filter values
        threat = self._threat_combo.currentText().lower()
        classification = self._class_combo.currentText()
        search = self._search_edit.text().lower()

        # Filter alerts
        self._filtered_alerts = []
        for alert in self._alerts:
            # Threat filter
            if threat != "all" and alert.threat_level != threat:
                continue

            # Class filter
            if classification != "All Classes" and alert.classification != classification:
                continue

            # Search filter
            if search:
                search_text = f"{alert.classification} {alert.threat_level}".lower()
                if search not in search_text:
                    continue

            self._filtered_alerts.append(alert)

        # Update displays
        self._update_cards()
        self._update_table()
        self._update_stats()

    def _update_cards(self) -> None:
        """Update alert cards display."""
        # Clear existing cards
        while self._cards_layout.count() > 1:
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add cards for recent alerts (top 10)
        for alert in self._filtered_alerts[:10]:
            card = AlertCard(alert)
            card.clicked.connect(self.alert_selected.emit)
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)

    def _update_table(self) -> None:
        """Update event log table."""
        self._table.setRowCount(len(self._filtered_alerts))

        for row, alert in enumerate(self._filtered_alerts):
            # Time
            time_str = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S.%f")[:-3]
            self._table.setItem(row, 0, QTableWidgetItem(time_str))

            # Track ID
            track_str = str(alert.track_id) if alert.track_id else "-"
            self._table.setItem(row, 1, QTableWidgetItem(track_str))

            # Classification
            self._table.setItem(row, 2, QTableWidgetItem(alert.classification))

            # Threat level
            threat_item = QTableWidgetItem(alert.threat_level.upper())
            color = AlertCard.COLORS.get(alert.threat_level, '#888888')
            threat_item.setForeground(QBrush(QColor(color)))
            self._table.setItem(row, 3, threat_item)

            # Azimuth
            self._table.setItem(row, 4, QTableWidgetItem(f"{alert.azimuth:.1f}deg"))

            # Distance
            self._table.setItem(row, 5, QTableWidgetItem(f"{alert.distance:.1f}m"))

            # Confidence
            self._table.setItem(row, 6, QTableWidgetItem(f"{alert.confidence:.1%}"))

            # SNR
            self._table.setItem(row, 7, QTableWidgetItem(f"{alert.snr:.1f}dB"))

    def _update_stats(self) -> None:
        """Update statistics display."""
        total = len(self._alerts)
        filtered = len(self._filtered_alerts)

        if total == filtered:
            self._stats_label.setText(f"{total} alerts")
        else:
            self._stats_label.setText(f"{filtered} / {total} alerts")

        # Count by threat level
        high = sum(1 for a in self._alerts if a.threat_level == "high")
        medium = sum(1 for a in self._alerts if a.threat_level == "medium")
        low = sum(1 for a in self._alerts if a.threat_level == "low")

        if high > 0:
            self._stats_label.setText(
                f"{total} alerts ({high} high, {medium} medium, {low} low)"
            )

    def _show_context_menu(self, pos) -> None:
        """Show context menu for table."""
        menu = QMenu(self)

        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self._copy_selected)
        menu.addAction(copy_action)

        menu.addSeparator()

        export_action = QAction("Export All...", self)
        export_action.triggered.connect(self.export_requested.emit)
        menu.addAction(export_action)

        menu.exec(self._table.mapToGlobal(pos))

    def _copy_selected(self) -> None:
        """Copy selected rows to clipboard."""
        from PyQt6.QtWidgets import QApplication

        selected = self._table.selectedItems()
        if not selected:
            return

        rows = set(item.row() for item in selected)
        text_lines = []

        for row in sorted(rows):
            row_data = []
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                if item:
                    row_data.append(item.text())
            text_lines.append("\t".join(row_data))

        QApplication.clipboard().setText("\n".join(text_lines))

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()
        self._filtered_alerts.clear()
        self._update_cards()
        self._update_table()
        self._update_stats()

    def get_alerts(self) -> List[AlertEvent]:
        """Get all alerts."""
        return list(self._alerts)

    def get_filtered_alerts(self) -> List[AlertEvent]:
        """Get filtered alerts."""
        return self._filtered_alerts.copy()


class MiniAlertsWidget(QWidget):
    """Compact alerts widget for sidebar."""

    alert_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._alerts: deque = deque(maxlen=5)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        # Header
        header = QLabel("Recent Alerts")
        header.setStyleSheet("font-weight: bold; color: #00ff88;")
        layout.addWidget(header)

        # Alert labels
        self._labels: List[QLabel] = []
        for _ in range(5):
            label = QLabel("-")
            label.setStyleSheet("""
                padding: 3px;
                background-color: #2a2a3e;
                border-radius: 3px;
                font-size: 10px;
            """)
            self._labels.append(label)
            layout.addWidget(label)

        layout.addStretch()

    def add_alert(self, alert: AlertEvent) -> None:
        """Add alert."""
        self._alerts.appendleft(alert)
        self._update_display()

    def _update_display(self) -> None:
        """Update display."""
        for i, label in enumerate(self._labels):
            if i < len(self._alerts):
                alert = self._alerts[i]
                color = AlertCard.COLORS.get(alert.threat_level, '#888888')
                label.setText(
                    f"{alert.classification} - {alert.distance:.0f}m"
                )
                label.setStyleSheet(f"""
                    padding: 3px;
                    background-color: #2a2a3e;
                    border-left: 3px solid {color};
                    border-radius: 3px;
                    font-size: 10px;
                """)
            else:
                label.setText("-")
                label.setStyleSheet("""
                    padding: 3px;
                    background-color: #2a2a3e;
                    border-radius: 3px;
                    font-size: 10px;
                    color: #666666;
                """)
