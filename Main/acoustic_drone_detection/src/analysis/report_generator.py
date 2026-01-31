"""
Report Generator Module

Generates analysis reports in various formats:
- JSON reports
- CSV exports
- HTML reports
- PDF reports (optional)
"""

import json
import csv
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from .batch_processor import BatchResult, FileAnalysisResult, DetectionEvent


@dataclass
class ReportConfig:
    """Report generation configuration."""
    include_summary: bool = True
    include_timeline: bool = True
    include_statistics: bool = True
    include_file_details: bool = True
    include_detections: bool = True
    max_detections_per_file: int = 100


class ReportGenerator:
    """
    Generates analysis reports from batch processing results.

    Supports JSON, CSV, and HTML output formats.
    """

    def __init__(self, output_directory: str, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.

        Args:
            output_directory: Directory for output files.
            config: Report configuration.
        """
        self._output_dir = Path(output_directory)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or ReportConfig()

    def generate_json_report(
        self,
        batch_result: BatchResult,
        filename: str = "report.json"
    ) -> str:
        """
        Generate JSON report.

        Args:
            batch_result: Batch processing result.
            filename: Output filename.

        Returns:
            Path to generated report.
        """
        report_data = self._build_report_data(batch_result)

        output_path = self._output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def generate_csv_report(
        self,
        batch_result: BatchResult,
        filename: str = "detections.csv"
    ) -> str:
        """
        Generate CSV report of detections.

        Args:
            batch_result: Batch processing result.
            filename: Output filename.

        Returns:
            Path to generated report.
        """
        output_path = self._output_dir / filename

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'File', 'Time (s)', 'Azimuth (deg)', 'Elevation (deg)',
                'Distance (m)', 'Confidence', 'Classification',
                'Threat Level', 'Dominant Frequencies'
            ])

            # Data rows
            for result in batch_result.results:
                for det in result.detections[:self._config.max_detections_per_file]:
                    writer.writerow([
                        result.filename,
                        f"{det.file_time:.3f}",
                        f"{det.azimuth:.1f}",
                        f"{det.elevation:.1f}",
                        f"{det.distance:.1f}",
                        f"{det.confidence:.3f}",
                        det.classification,
                        det.threat_level,
                        ';'.join(f"{f:.1f}" for f in det.dominant_frequencies)
                    ])

        return str(output_path)

    def generate_html_report(
        self,
        batch_result: BatchResult,
        filename: str = "report.html",
        title: str = "Drone Detection Analysis Report"
    ) -> str:
        """
        Generate HTML report.

        Args:
            batch_result: Batch processing result.
            filename: Output filename.
            title: Report title.

        Returns:
            Path to generated report.
        """
        summary = batch_result.get_summary()

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a2e;
            color: #e0e0e0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #00ff88;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: #2a2a4e;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #00ff88;
        }}
        .summary-card .label {{
            color: #888;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background-color: #2a2a4e;
            color: #00ff88;
        }}
        tr:hover {{
            background-color: #2a2a4e;
        }}
        .threat-high {{
            color: #ff4444;
            font-weight: bold;
        }}
        .threat-medium {{
            color: #ffaa00;
        }}
        .threat-low {{
            color: #00ff88;
        }}
        .chart-container {{
            background: #2a2a4e;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{batch_result.files_processed}</div>
                <div class="label">Files Processed</div>
            </div>
            <div class="summary-card">
                <div class="value">{batch_result.total_detections}</div>
                <div class="label">Total Detections</div>
            </div>
            <div class="summary-card">
                <div class="value">{batch_result.files_with_detections}</div>
                <div class="label">Files with Detections</div>
            </div>
            <div class="summary-card">
                <div class="value">{batch_result.total_processing_time:.1f}s</div>
                <div class="label">Processing Time</div>
            </div>
        </div>

        <h2>Threat Level Distribution</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value threat-high">{summary['threat_levels'].get('high', 0)}</div>
                <div class="label">High Threat</div>
            </div>
            <div class="summary-card">
                <div class="value threat-medium">{summary['threat_levels'].get('medium', 0)}</div>
                <div class="label">Medium Threat</div>
            </div>
            <div class="summary-card">
                <div class="value threat-low">{summary['threat_levels'].get('low', 0)}</div>
                <div class="label">Low Threat</div>
            </div>
        </div>

        <h2>Classifications</h2>
        <table>
            <thead>
                <tr>
                    <th>Classification</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""
        total_class = sum(summary['classifications'].values()) or 1
        for cls, count in sorted(summary['classifications'].items(), key=lambda x: -x[1]):
            pct = count / total_class * 100
            html_content += f"""
                <tr>
                    <td>{cls}</td>
                    <td>{count}</td>
                    <td>{pct:.1f}%</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <h2>Detection Details</h2>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Time</th>
                    <th>Position</th>
                    <th>Distance</th>
                    <th>Classification</th>
                    <th>Confidence</th>
                    <th>Threat</th>
                </tr>
            </thead>
            <tbody>
"""
        for result in batch_result.results:
            for det in result.detections[:self._config.max_detections_per_file]:
                threat_class = f"threat-{det.threat_level}"
                html_content += f"""
                <tr>
                    <td>{result.filename}</td>
                    <td>{det.file_time:.2f}s</td>
                    <td>Az: {det.azimuth:.1f}, El: {det.elevation:.1f}</td>
                    <td>{det.distance:.1f}m</td>
                    <td>{det.classification}</td>
                    <td>{det.confidence:.1%}</td>
                    <td class="{threat_class}">{det.threat_level.upper()}</td>
                </tr>
"""

        html_content += f"""
            </tbody>
        </table>

        <div class="footer">
            <p>Acoustic Drone Detection System - Analysis Report</p>
            <p>Processing period: {batch_result.start_time} to {batch_result.end_time}</p>
        </div>
    </div>
</body>
</html>
"""

        output_path = self._output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(output_path)

    def _build_report_data(self, batch_result: BatchResult) -> Dict[str, Any]:
        """Build report data structure."""
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0',
                'processing_period': {
                    'start': batch_result.start_time,
                    'end': batch_result.end_time
                }
            }
        }

        if self._config.include_summary:
            report['summary'] = batch_result.get_summary()

        if self._config.include_statistics:
            report['statistics'] = self._compute_statistics(batch_result)

        if self._config.include_file_details:
            report['files'] = [
                {
                    'filepath': r.filepath,
                    'filename': r.filename,
                    'status': r.status,
                    'detection_count': r.detection_count,
                    'processing_time': r.processing_time_seconds,
                    'metadata': {
                        'format': r.metadata.format,
                        'sample_rate': r.metadata.sample_rate,
                        'channels': r.metadata.channels,
                        'duration': r.metadata.duration_seconds
                    } if r.metadata else None
                }
                for r in batch_result.results
            ]

        if self._config.include_detections:
            report['detections'] = []
            for result in batch_result.results:
                for det in result.detections[:self._config.max_detections_per_file]:
                    report['detections'].append({
                        'file': result.filename,
                        'file_time': det.file_time,
                        'azimuth': det.azimuth,
                        'elevation': det.elevation,
                        'distance': det.distance,
                        'confidence': det.confidence,
                        'classification': det.classification,
                        'threat_level': det.threat_level,
                        'dominant_frequencies': det.dominant_frequencies
                    })

        if self._config.include_timeline:
            report['timeline'] = self._build_timeline(batch_result)

        return report

    def _compute_statistics(self, batch_result: BatchResult) -> Dict[str, Any]:
        """Compute statistical metrics."""
        all_detections = []
        for result in batch_result.results:
            all_detections.extend(result.detections)

        if not all_detections:
            return {
                'detection_count': 0,
                'confidence': {'mean': 0, 'min': 0, 'max': 0},
                'distance': {'mean': 0, 'min': 0, 'max': 0}
            }

        confidences = [d.confidence for d in all_detections]
        distances = [d.distance for d in all_detections]

        return {
            'detection_count': len(all_detections),
            'confidence': {
                'mean': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences)
            },
            'distance': {
                'mean': sum(distances) / len(distances),
                'min': min(distances),
                'max': max(distances)
            }
        }

    def _build_timeline(self, batch_result: BatchResult) -> List[Dict]:
        """Build detection timeline."""
        timeline = []

        for result in batch_result.results:
            for det in result.detections:
                timeline.append({
                    'file': result.filename,
                    'time': det.file_time,
                    'classification': det.classification,
                    'threat_level': det.threat_level
                })

        # Sort by file then time
        timeline.sort(key=lambda x: (x['file'], x['time']))

        return timeline

    def generate_all_reports(
        self,
        batch_result: BatchResult,
        base_name: str = "analysis"
    ) -> Dict[str, str]:
        """
        Generate all report formats.

        Args:
            batch_result: Batch processing result.
            base_name: Base filename for reports.

        Returns:
            Dictionary mapping format to file path.
        """
        reports = {}

        reports['json'] = self.generate_json_report(
            batch_result, f"{base_name}.json"
        )
        reports['csv'] = self.generate_csv_report(
            batch_result, f"{base_name}_detections.csv"
        )
        reports['html'] = self.generate_html_report(
            batch_result, f"{base_name}.html"
        )

        return reports
