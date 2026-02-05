"""
Data Export Module

Handles exporting detection data to various formats:
- JSON (streaming and batch)
- CSV
- SQLite database
- WebSocket streaming
"""

import json
import csv
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator
from dataclasses import dataclass, asdict
import threading
from queue import Queue
import numpy as np


@dataclass
class DetectionRecord:
    """Standard detection record for export."""
    timestamp: float
    datetime_str: str
    azimuth: float
    elevation: float
    distance: float
    confidence: float
    snr: float
    classification: str
    threat_level: str
    dominant_frequencies: List[float]
    track_id: Optional[int] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class JSONExporter:
    """
    JSON data exporter with streaming and batch support.

    Provides both single-file export and real-time streaming capabilities.
    """

    def __init__(self, output_path: Optional[str] = None, pretty: bool = True):
        """
        Initialize the JSON exporter.

        Args:
            output_path: Path for output file.
            pretty: Use pretty formatting with indentation.
        """
        self.output_path = output_path
        self.pretty = pretty
        self._records: List[Dict] = []
        self._lock = threading.Lock()

    def add_record(self, record: DetectionRecord) -> None:
        """
        Add a detection record.

        Args:
            record: Detection record to add.
        """
        with self._lock:
            self._records.append(asdict(record))

    def export(self, output_path: Optional[str] = None) -> bool:
        """
        Export all records to JSON file.

        Args:
            output_path: Output file path (uses default if not specified).

        Returns:
            True if export successful.
        """
        path = output_path or self.output_path
        if not path:
            return False

        try:
            with self._lock:
                data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'record_count': len(self._records),
                    'detections': self._records.copy()
                }

            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                if self.pretty:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(data, f, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"JSON export error: {e}")
            return False

    def export_streaming(self, output_path: str) -> Generator[str, None, None]:
        """
        Stream records as JSON lines.

        Args:
            output_path: Output file path.

        Yields:
            JSON string for each record.
        """
        with self._lock:
            records = self._records.copy()

        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                line = json.dumps(record, ensure_ascii=False)
                f.write(line + '\n')
                yield line

    def clear(self) -> None:
        """Clear all stored records."""
        with self._lock:
            self._records.clear()

    def get_json_string(self, record: DetectionRecord) -> str:
        """
        Get JSON string for a single record.

        Args:
            record: Detection record.

        Returns:
            JSON string representation.
        """
        return json.dumps(asdict(record), ensure_ascii=False)


class CSVExporter:
    """
    CSV data exporter.

    Exports detection data in CSV format with configurable columns.
    """

    DEFAULT_COLUMNS = [
        'timestamp', 'datetime_str', 'azimuth', 'elevation', 'distance',
        'confidence', 'snr', 'classification', 'threat_level',
        'dominant_frequencies', 'track_id', 'x', 'y', 'z'
    ]

    def __init__(
        self,
        output_path: Optional[str] = None,
        columns: Optional[List[str]] = None
    ):
        """
        Initialize the CSV exporter.

        Args:
            output_path: Path for output file.
            columns: List of columns to export.
        """
        self.output_path = output_path
        self.columns = columns or self.DEFAULT_COLUMNS
        self._records: List[Dict] = []
        self._lock = threading.Lock()

    def add_record(self, record: DetectionRecord) -> None:
        """Add a detection record."""
        with self._lock:
            self._records.append(asdict(record))

    def export(self, output_path: Optional[str] = None) -> bool:
        """
        Export all records to CSV file.

        Args:
            output_path: Output file path.

        Returns:
            True if export successful.
        """
        path = output_path or self.output_path
        if not path:
            return False

        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

            with self._lock:
                records = self._records.copy()

            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.columns, extrasaction='ignore')
                writer.writeheader()

                for record in records:
                    # Convert lists to strings for CSV
                    row = record.copy()
                    if 'dominant_frequencies' in row:
                        row['dominant_frequencies'] = ';'.join(
                            map(str, row['dominant_frequencies'])
                        )
                    writer.writerow(row)

            return True

        except Exception as e:
            print(f"CSV export error: {e}")
            return False

    def clear(self) -> None:
        """Clear all stored records."""
        with self._lock:
            self._records.clear()


class SQLiteExporter:
    """
    SQLite database exporter.

    Provides persistent storage with query capabilities for historical analysis.
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        datetime_str TEXT NOT NULL,
        azimuth REAL NOT NULL,
        elevation REAL NOT NULL,
        distance REAL NOT NULL,
        confidence REAL NOT NULL,
        snr REAL,
        classification TEXT NOT NULL,
        threat_level TEXT NOT NULL,
        dominant_frequencies TEXT,
        track_id INTEGER,
        x REAL,
        y REAL,
        z REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """

    CREATE_INDEX_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_classification ON detections(classification)",
        "CREATE INDEX IF NOT EXISTS idx_threat_level ON detections(threat_level)",
        "CREATE INDEX IF NOT EXISTS idx_track_id ON detections(track_id)"
    ]

    def __init__(self, db_path: str = "data/detections.db"):
        """
        Initialize the SQLite exporter.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the database and create tables."""
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(self.CREATE_TABLE_SQL)
            for idx_sql in self.CREATE_INDEX_SQL:
                conn.execute(idx_sql)
            conn.commit()

    def add_record(self, record: DetectionRecord) -> int:
        """
        Add a detection record to the database.

        Args:
            record: Detection record to add.

        Returns:
            ID of the inserted record.
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO detections (
                        timestamp, datetime_str, azimuth, elevation, distance,
                        confidence, snr, classification, threat_level,
                        dominant_frequencies, track_id, x, y, z
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.timestamp,
                    record.datetime_str,
                    record.azimuth,
                    record.elevation,
                    record.distance,
                    record.confidence,
                    record.snr,
                    record.classification,
                    record.threat_level,
                    json.dumps(record.dominant_frequencies),
                    record.track_id,
                    record.x,
                    record.y,
                    record.z
                ))
                conn.commit()
                return cursor.lastrowid

    def query(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        classification: Optional[str] = None,
        threat_level: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Query detection records from the database.

        Args:
            start_time: Start timestamp filter.
            end_time: End timestamp filter.
            classification: Classification filter.
            threat_level: Threat level filter.
            min_confidence: Minimum confidence filter.
            limit: Maximum number of records to return.

        Returns:
            List of matching detection records.
        """
        conditions = []
        params = []

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time)

        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        if classification is not None:
            conditions.append("classification = ?")
            params.append(classification)

        if threat_level is not None:
            conditions.append("threat_level = ?")
            params.append(threat_level)

        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT * FROM detections
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params + [limit])

            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical summary of stored detections.

        Returns:
            Dictionary with statistics.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total count
            total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

            # Counts by classification
            classifications = dict(conn.execute("""
                SELECT classification, COUNT(*) FROM detections
                GROUP BY classification
            """).fetchall())

            # Counts by threat level
            threat_levels = dict(conn.execute("""
                SELECT threat_level, COUNT(*) FROM detections
                GROUP BY threat_level
            """).fetchall())

            # Average confidence
            avg_confidence = conn.execute("""
                SELECT AVG(confidence) FROM detections
            """).fetchone()[0]

            # Time range
            time_range = conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp) FROM detections
            """).fetchone()

            return {
                'total_detections': total,
                'by_classification': classifications,
                'by_threat_level': threat_levels,
                'average_confidence': avg_confidence,
                'earliest_detection': time_range[0],
                'latest_detection': time_range[1]
            }

    def delete_old_records(self, days: int = 30) -> int:
        """
        Delete records older than specified days.

        Args:
            days: Number of days to keep.

        Returns:
            Number of deleted records.
        """
        import time
        cutoff = time.time() - (days * 24 * 60 * 60)

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM detections WHERE timestamp < ?",
                    (cutoff,)
                )
                conn.commit()
                return cursor.rowcount


class DataExportManager:
    """
    Unified data export manager.

    Manages multiple export formats and provides a unified interface.
    """

    def __init__(
        self,
        export_dir: str = "data/exports",
        enable_json: bool = True,
        enable_csv: bool = True,
        enable_sqlite: bool = True
    ):
        """
        Initialize the export manager.

        Args:
            export_dir: Directory for export files.
            enable_json: Enable JSON export.
            enable_csv: Enable CSV export.
            enable_sqlite: Enable SQLite export.
        """
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

        self.json_exporter = JSONExporter() if enable_json else None
        self.csv_exporter = CSVExporter() if enable_csv else None
        self.sqlite_exporter = SQLiteExporter(
            os.path.join(export_dir, "detections.db")
        ) if enable_sqlite else None

    def add_detection(self, record: DetectionRecord) -> None:
        """
        Add a detection to all enabled exporters.

        Args:
            record: Detection record to add.
        """
        if self.json_exporter:
            self.json_exporter.add_record(record)

        if self.csv_exporter:
            self.csv_exporter.add_record(record)

        if self.sqlite_exporter:
            self.sqlite_exporter.add_record(record)

    def export_all(self, prefix: str = "") -> Dict[str, str]:
        """
        Export data to all enabled formats.

        Args:
            prefix: Filename prefix.

        Returns:
            Dictionary mapping format to file path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{prefix}_" if prefix else ""
        exported = {}

        if self.json_exporter:
            path = os.path.join(self.export_dir, f"{prefix}{timestamp}.json")
            if self.json_exporter.export(path):
                exported['json'] = path

        if self.csv_exporter:
            path = os.path.join(self.export_dir, f"{prefix}{timestamp}.csv")
            if self.csv_exporter.export(path):
                exported['csv'] = path

        if self.sqlite_exporter:
            exported['sqlite'] = self.sqlite_exporter.db_path

        return exported

    def clear_buffers(self) -> None:
        """Clear all in-memory buffers."""
        if self.json_exporter:
            self.json_exporter.clear()
        if self.csv_exporter:
            self.csv_exporter.clear()


def create_detection_record(
    timestamp: float,
    azimuth: float,
    elevation: float,
    distance: float,
    confidence: float,
    snr: float,
    classification: str,
    threat_level: str,
    dominant_frequencies: List[float],
    track_id: Optional[int] = None,
    cartesian: Optional[tuple] = None
) -> DetectionRecord:
    """
    Create a detection record from detection data.

    Args:
        timestamp: Unix timestamp of detection.
        azimuth: Detection azimuth in degrees.
        elevation: Detection elevation in degrees.
        distance: Estimated distance in meters.
        confidence: Detection confidence (0-1).
        snr: Signal-to-noise ratio in dB.
        classification: Drone classification.
        threat_level: Threat level assessment.
        dominant_frequencies: List of dominant frequencies.
        track_id: Associated track ID.
        cartesian: Optional (x, y, z) coordinates.

    Returns:
        DetectionRecord instance.
    """
    x, y, z = cartesian if cartesian else (None, None, None)

    return DetectionRecord(
        timestamp=timestamp,
        datetime_str=datetime.fromtimestamp(timestamp).isoformat(),
        azimuth=azimuth,
        elevation=elevation,
        distance=distance,
        confidence=confidence,
        snr=snr,
        classification=classification,
        threat_level=threat_level,
        dominant_frequencies=list(dominant_frequencies),
        track_id=track_id,
        x=x,
        y=y,
        z=z
    )
