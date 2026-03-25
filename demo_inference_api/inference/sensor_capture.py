from __future__ import annotations

import csv
import io
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensor_reader import SensorReader


def capture_imu(duration_sec: float, output_csv: str | None = None) -> str:
    """
    用非 root 的 AppleSPUHIDDevice 直接 service matching 路径采集 IMU。
    返回 CSV 字符串（timestamp_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z）。
    如果指定 output_csv，同时写入文件。
    """
    reader = SensorReader(force_macimu=False)
    reader.start()
    try:
        time.sleep(max(float(duration_sec), 0.0))
        samples = reader.drain()
    finally:
        reader.stop()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'timestamp_ns', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'
    ])
    for s in samples:
        writer.writerow([
            int(s.timestamp_ns),
            float(s.accel_x),
            float(s.accel_y),
            float(s.accel_z),
            float(s.gyro_x),
            float(s.gyro_y),
            float(s.gyro_z),
        ])
    csv_string = buf.getvalue()

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(csv_string, encoding='utf-8')

    return csv_string
