"""
Istanbul ADS-B Trajectory Processor 
"""

import json
import gzip
import re
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from dataclasses import dataclass, field
import os

# ============================================================
# CONFIGURATION
# ============================================================

IST_LAT, IST_LON = 41.275278, 28.751944
RADIUS_KM = 120
R_EARTH = 6371.0
BATCH_SIZE = 100_000

BASE_PATH = Path("/Users/YGT/1 airline project/traj")
OUT_DIR = Path("/Users/YGT/1 airline project/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Debug mode: set DEBUG=1 environment variable for verbose logging
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# DATA QUALITY BOUNDS
# ============================================================

BOUNDS = {
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "altitude_baro": (-1500.0, 65000.0),      # Feet (Dead Sea to max altitude)
    "altitude_geom": (-1500.0, 65000.0),
    "ground_speed": (0.0, 850.0),             # Knots (max commercial ~600kt)
    "track": (0.0, 360.0),                    # Degrees
    "ias": (0.0, 500.0),                      # Indicated airspeed (knots)
    "tas": (0.0, 850.0),                      # True airspeed (knots)
    "baro_rate": (-10000.0, 10000.0),         # Feet per minute
    "geom_rate": (-10000.0, 10000.0),
}

# ============================================================
# SCHEMA
# ============================================================

SCHEMA = pa.schema([
    ("hex", pa.string()),
    ("icao", pa.string()),
    ("date", pa.string()),
    ("trajectory_timestamp", pa.timestamp("us")),
    ("point_timestamp", pa.timestamp("us")),
    ("timestamp_offset", pa.float32()),
    ("latitude", pa.float32()),
    ("longitude", pa.float32()),
    ("distance_km", pa.float32()),
    ("altitude_baro", pa.float32()),
    ("altitude_geom", pa.float32()),
    ("on_ground", pa.bool_()),
    ("ground_speed", pa.float32()),
    ("track", pa.float32()),
    ("ias", pa.float32()),
    ("tas", pa.float32()),
    ("baro_rate", pa.float32()),
    ("geom_rate", pa.float32()),
    ("flags", pa.int32()),
    ("operational", pa.float32()),
    ("squawk", pa.string()),
    ("source_type", pa.string()),
    ("nav_altitude_mcp", pa.float32()),
    ("nav_altitude_fms", pa.float32()),
    ("nic", pa.int8()),
    ("rc", pa.int32()),
    ("field_12", pa.float32()),
    ("field_13", pa.float32()),
])

# ============================================================
# ENHANCED STATISTICS WITH DETAILED TRACKING
# ============================================================

@dataclass
class DetailedStats:
    """Comprehensive statistics tracking for debugging and quality monitoring."""

    # File-level stats
    files_total: int = 0
    files_valid: int = 0
    files_parse_failed: int = 0
    files_no_trace: int = 0
    files_no_timestamp: int = 0

    # Point-level stats
    points_examined: int = 0
    points_written: int = 0

    # Skip reasons (detailed)
    skipped_invalid_structure: int = 0      # Not a list or too short
    skipped_missing_coords: int = 0          # Lat or lon is None
    skipped_invalid_coord_range: int = 0     # Lat/lon outside valid range
    skipped_out_of_radius: int = 0           # Beyond 120km from Istanbul

    # Data quality issues
    missing_dict8: int = 0                   # Index [8] missing or not a dict
    ground_altitude_conversions: int = 0     # "ground" string → 0.0

    # Field-level null counts
    null_counts: dict = field(default_factory=lambda: {
        "altitude_geom": 0,
        "ground_speed": 0,
        "track": 0,
        "ias": 0,
        "tas": 0,
        "baro_rate": 0,
        "geom_rate": 0,
        "squawk": 0,
        "source_type": 0,
        "nav_altitude_mcp": 0,
        "nav_altitude_fms": 0,
        "nic": 0,
        "rc": 0,
    })

    # Validation warnings (out of expected range)
    validation_warnings: dict = field(default_factory=lambda: {
        "altitude_baro": 0,
        "altitude_geom": 0,
        "ground_speed": 0,
        "track": 0,
        "ias": 0,
        "tas": 0,
        "baro_rate": 0,
        "geom_rate": 0,
    })

    # Sample problematic records (for debugging)
    sample_parse_errors: List[Tuple[str, str]] = field(default_factory=list)
    sample_invalid_coords: List[Tuple[str, float, float]] = field(default_factory=list)
    sample_out_of_range: List[Tuple[str, str, float]] = field(default_factory=list)

    MAX_SAMPLES = 5  # Keep first N samples of each issue type

    def track_null(self, field_name: str):
        """Track a NULL value in a field."""
        if field_name in self.null_counts:
            self.null_counts[field_name] += 1

    def track_validation_warning(self, field_name: str, value: float, hex_code: str):
        """Track a value outside expected range."""
        if field_name in self.validation_warnings:
            self.validation_warnings[field_name] += 1

            # Sample the first few occurrences
            if len(self.sample_out_of_range) < self.MAX_SAMPLES:
                self.sample_out_of_range.append((hex_code, field_name, value))
                logger.debug(
                    f"Out-of-range {field_name}: {value} for {hex_code} "
                    f"(expected {BOUNDS.get(field_name, 'N/A')})"
                )

    def log_summary(self):
        """Print comprehensive statistics report."""
        logger.info(f"\n{'='*80}")
        logger.info("PROCESSING SUMMARY")
        logger.info(f"{'='*80}")

        # Files
        logger.info(f"\nFILES:")
        logger.info(f"  Total examined:     {self.files_total:>12,}")
        logger.info(f"  Successfully read:  {self.files_valid:>12,} ({100*self.files_valid/max(1,self.files_total):>5.1f}%)")
        logger.info(f"  Parse failures:     {self.files_parse_failed:>12,}")
        logger.info(f"  Missing trace:      {self.files_no_trace:>12,}")
        logger.info(f"  Missing timestamp:  {self.files_no_timestamp:>12,}")

        # Points
        total_skipped = (self.skipped_invalid_structure + self.skipped_missing_coords +
                        self.skipped_invalid_coord_range + self.skipped_out_of_radius)

        logger.info(f"\nTRACE POINTS:")
        logger.info(f"  Examined:           {self.points_examined:>12,}")
        logger.info(f"  Written to Parquet: {self.points_written:>12,} ({100*self.points_written/max(1,self.points_examined):>5.1f}%)")
        logger.info(f"  Total filtered:     {total_skipped:>12,}")

        # Skip reasons breakdown
        logger.info(f"\nFILTER REASONS:")
        logger.info(f"  Invalid structure:  {self.skipped_invalid_structure:>12,} (not list or len<3)")
        logger.info(f"  Missing coords:     {self.skipped_missing_coords:>12,} (lat/lon is None)")
        logger.info(f"  Invalid coord range:{self.skipped_invalid_coord_range:>12,} (lat/lon bounds)")
        logger.info(f"  Out of radius:      {self.skipped_out_of_radius:>12,} (>{RADIUS_KM}km from Istanbul)")

        # Data quality
        logger.info(f"\nDATA QUALITY:")
        logger.info(f"  Missing dict[8]:    {self.missing_dict8:>12,}")
        logger.info(f"  'ground' → 0.0 alt: {self.ground_altitude_conversions:>12,}")

        # Null counts for optional fields
        logger.info(f"\nNULL VALUE COUNTS (optional fields):")
        for field_name, count in sorted(self.null_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = 100 * count / max(1, self.points_written)
                logger.info(f"  {field_name:20s}: {count:>12,} ({pct:>5.1f}% of written points)")

        # Validation warnings
        total_warnings = sum(self.validation_warnings.values())
        if total_warnings > 0:
            logger.info(f"\nVALIDATION WARNINGS (out-of-range values):")
            for field_name, count in sorted(self.validation_warnings.items(), key=lambda x: -x[1]):
                if count > 0:
                    logger.info(f"  {field_name:20s}: {count:>12,} warnings")

        # Sample errors
        if self.sample_parse_errors:
            logger.info(f"\nSAMPLE PARSE ERRORS (first {len(self.sample_parse_errors)}):")
            for filename, error in self.sample_parse_errors:
                logger.info(f"  {filename}: {error}")

        if self.sample_invalid_coords:
            logger.info(f"\nSAMPLE INVALID COORDINATES (first {len(self.sample_invalid_coords)}):")
            for hex_code, lat, lon in self.sample_invalid_coords:
                logger.info(f"  {hex_code}: lat={lat}, lon={lon}")

        if self.sample_out_of_range:
            logger.info(f"\nSAMPLE OUT-OF-RANGE VALUES (first {len(self.sample_out_of_range)}):")
            for hex_code, field, value in self.sample_out_of_range:
                logger.info(f"  {hex_code}: {field}={value} (expected {BOUNDS[field]})")

        logger.info(f"\n{'='*80}\n")

# Global stats instance
stats = DetailedStats()

# ============================================================
# UTILS
# ============================================================

def haversine_vectorized(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Calculate distance from Istanbul using vectorized haversine formula."""
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)
    dlat = lat_rad - np.radians(IST_LAT)
    dlon = lon_rad - np.radians(IST_LON)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(IST_LAT))*np.cos(lat_rad)*np.sin(dlon/2)**2
    return 2 * R_EARTH * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def read_json_smart(path: Path) -> Optional[dict]:
    """
    Read JSON with automatic gzip detection via magic bytes.
    Tracks parse failures for debugging.
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(2)

        if magic == b'\x1f\x8b':  # gzip magic bytes
            with gzip.open(path, "rt", encoding="utf-8") as gz:
                data = json.load(gz)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Handle list wrapper
        if isinstance(data, list):
            return data[0] if data else None
        return data

    except Exception as e:
        stats.files_parse_failed += 1

        # Sample first few parse errors
        if len(stats.sample_parse_errors) < stats.MAX_SAMPLES:
            stats.sample_parse_errors.append((path.name, str(e)))

        logger.debug(f"Parse failed: {path.name} - {e}")
        return None

def extract_hex(path: Path) -> str:
    """Extract hex code from trace_full_<HEX>.json filename."""
    m = re.search(r"trace_full_([a-f0-9]+)", path.name)
    return m.group(1) if m else "unknown"

def extract_date(dir_name: str) -> str:
    """Extract date from directory name (format: vYYYY.MM.DD-...)."""
    m = re.search(r'(\d{4})\.(\d{2})\.(\d{2})', dir_name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "unknown"

def normalize_timestamp(ts) -> Optional[float]:
    """
    Normalize timestamp to UNIX seconds.
    Handles microseconds (>1e12), milliseconds (>1e9), or seconds.
    """
    if not isinstance(ts, (int, float)):
        return None

    ts = float(ts)

    if ts > 1e12:      # Microseconds
        return ts / 1e6
    elif ts > 1e9:     # Milliseconds
        return ts / 1e3
    else:              # Seconds
        return ts

def safe_float(v, default=None):
    """Safely convert to float, returning default if conversion fails."""
    try:
        if v is None:
            return default
        return float(v)
    except (ValueError, TypeError):
        return default

def safe_int(v, default=None):
    """Safely convert to int, returning default if conversion fails."""
    try:
        if v is None:
            return default
        return int(v)
    except (ValueError, TypeError):
        return default

def safe_string(v, default=None):
    """Safely convert to string."""
    if v is None:
        return default
    return str(v)

def validate_range(value: Optional[float], field_name: str, hex_code: str) -> Optional[float]:
    """
    Validate that a value is within expected bounds.
    Returns the value if valid, logs warning if out of range.
    """
    if value is None or field_name not in BOUNDS:
        return value

    min_val, max_val = BOUNDS[field_name]

    if not (min_val <= value <= max_val):
        stats.track_validation_warning(field_name, value, hex_code)

    return value

def extract_dict_fields(point: list, hex_code: str) -> dict:
    """
    Extract fields from index [8] dictionary.
    Tracks if dict is missing for quality monitoring.
    """
    result = {
        "alt_geom": None,
        "baro_rate": None,
        "geom_rate": None,
        "ias": None,
        "tas": None,
        "squawk": None,
        "nic": None,
        "rc": None,
    }

    if len(point) <= 8 or point[8] is None:
        stats.missing_dict8 += 1
        return result

    if not isinstance(point[8], dict):
        stats.missing_dict8 += 1
        logger.debug(f"Index [8] not a dict for {hex_code}: {type(point[8])}")
        return result

    d = point[8]

    # Extract with validation
    result["alt_geom"] = validate_range(safe_float(d.get("alt_geom")), "altitude_geom", hex_code)
    result["baro_rate"] = validate_range(safe_float(d.get("baro_rate")), "baro_rate", hex_code)
    result["geom_rate"] = validate_range(safe_float(d.get("geom_rate")), "geom_rate", hex_code)
    result["ias"] = validate_range(safe_float(d.get("ias")), "ias", hex_code)
    result["tas"] = validate_range(safe_float(d.get("tas")), "tas", hex_code)
    result["squawk"] = safe_string(d.get("squawk"))
    result["nic"] = safe_int(d.get("nic"))
    result["rc"] = safe_int(d.get("rc"))

    return result

# ============================================================
# TRACE EXPLODER WITH QUALITY TRACKING
# ============================================================

def explode_trace(
    trace: list,
    hex_code: str,
    icao: str,
    date: str,
    base_ts: float
) -> list[dict]:
    """
    Explode trace into individual point dictionaries.
    
    ENHANCEMENTS:
    - Tracks all skip/filter reasons
    - Validates coordinate ranges
    - Validates data ranges for key fields
    - Tracks null counts per field
    - Samples problematic data for debugging
    """
    if not trace:
        return []

    # Phase 1: Validate structure and coordinates
    valid_pts, lats, lons = [], [], []

    for p in trace:
        # Check structure
        if not isinstance(p, list) or len(p) < 3:
            stats.skipped_invalid_structure += 1
            continue

        # Extract coordinates
        lat = safe_float(p[1])
        lon = safe_float(p[2])

        # Check for None
        if lat is None or lon is None:
            stats.skipped_missing_coords += 1
            continue

        # Validate coordinate ranges
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            stats.skipped_invalid_coord_range += 1

            # Sample first few invalid coords
            if len(stats.sample_invalid_coords) < stats.MAX_SAMPLES:
                stats.sample_invalid_coords.append((hex_code, lat, lon))

            continue

        valid_pts.append(p)
        lats.append(lat)
        lons.append(lon)

    if not valid_pts:
        return []

    stats.points_examined += len(valid_pts)

    # Phase 2: Vectorized distance filtering
    dists = haversine_vectorized(
        np.array(lats, dtype=np.float32),
        np.array(lons, dtype=np.float32)
    )

    mask = dists <= RADIUS_KM
    n_inside = np.sum(mask)
    stats.skipped_out_of_radius += len(valid_pts) - n_inside

    if n_inside == 0:
        return []

    # Phase 3: Build rows for points within radius
    rows = []

    for i, (p, inside) in enumerate(zip(valid_pts, mask)):
        if not inside:
            continue

        plen = len(p)

        # Time
        ts_offset = safe_float(p[0], 0.0)

        # Position
        lat, lon, dist = lats[i], lons[i], float(dists[i])

        # Altitude (barometric)
        alt_baro_raw = p[3] if plen > 3 else None
        on_ground = False
        alt_baro = None

        if alt_baro_raw is not None:
            if isinstance(alt_baro_raw, str) and alt_baro_raw.lower() == "ground":
                on_ground = True
                alt_baro = 0.0
                stats.ground_altitude_conversions += 1
            else:
                alt_baro = safe_float(alt_baro_raw)
                alt_baro = validate_range(alt_baro, "altitude_baro", hex_code)

        # Velocity
        gs = safe_float(p[4] if plen > 4 else None)
        gs = validate_range(gs, "ground_speed", hex_code)

        trk = safe_float(p[5] if plen > 5 else None)
        trk = validate_range(trk, "track", hex_code)

        # Flags and operational
        flags = safe_int(p[6] if plen > 6 else None)
        operational = safe_float(p[7] if plen > 7 else None)

        # Dict fields (index 8)
        meta = extract_dict_fields(p, hex_code)

        # Source and navigation
        src = safe_string(p[9] if plen > 9 else None)
        nav_mcp = safe_float(p[10] if plen > 10 else None)
        nav_fms = safe_float(p[11] if plen > 11 else None)

        # Additional fields
        f12 = safe_float(p[12] if plen > 12 else None)
        f13 = safe_float(p[13] if plen > 13 else None)

        # Track null counts for optional fields
        if meta["alt_geom"] is None:
            stats.track_null("altitude_geom")
        if gs is None:
            stats.track_null("ground_speed")
        if trk is None:
            stats.track_null("track")
        if meta["ias"] is None:
            stats.track_null("ias")
        if meta["tas"] is None:
            stats.track_null("tas")
        if meta["baro_rate"] is None:
            stats.track_null("baro_rate")
        if meta["geom_rate"] is None:
            stats.track_null("geom_rate")
        if meta["squawk"] is None:
            stats.track_null("squawk")
        if src is None:
            stats.track_null("source_type")
        if nav_mcp is None:
            stats.track_null("nav_altitude_mcp")
        if nav_fms is None:
            stats.track_null("nav_altitude_fms")
        if meta["nic"] is None:
            stats.track_null("nic")
        if meta["rc"] is None:
            stats.track_null("rc")

        # Calculate timestamps with microsecond precision
        # CRITICAL: Use integer microseconds to avoid float drift
        pt_ts = pd.Timestamp(base_ts * 1e6 + ts_offset * 1e6, unit='us', tz='UTC')
        traj_ts = pd.Timestamp(base_ts * 1e6, unit='us', tz='UTC')

        # Build row
        rows.append({
            "hex": hex_code,
            "icao": icao,
            "date": date,
            "trajectory_timestamp": traj_ts,
            "point_timestamp": pt_ts,
            "timestamp_offset": ts_offset,
            "latitude": lat,
            "longitude": lon,
            "distance_km": dist,
            "altitude_baro": alt_baro,
            "altitude_geom": meta["alt_geom"],
            "on_ground": on_ground,
            "ground_speed": gs,
            "track": trk,
            "ias": meta["ias"],
            "tas": meta["tas"],
            "baro_rate": meta["baro_rate"],
            "geom_rate": meta["geom_rate"],
            "flags": flags,
            "operational": operational,
            "squawk": meta["squawk"],
            "source_type": src,
            "nav_altitude_mcp": nav_mcp,
            "nav_altitude_fms": nav_fms,
            "nic": meta["nic"],
            "rc": meta["rc"],
            "field_12": f12,
            "field_13": f13,
        })

    stats.points_written += len(rows)

    return rows

# ============================================================
# BATCH WRITER
# ============================================================

class BatchWriter:
    """Write trace points to Parquet in batches for performance."""

    def __init__(self, out_file: Path):
        self.out_file = out_file
        self.buffer = []
        self.writer: Optional[pq.ParquetWriter] = None
        self.total_rows = 0

    def add_rows(self, rows: list):
        """Add rows to buffer and flush if batch size reached."""
        self.buffer.extend(rows)

        if len(self.buffer) >= BATCH_SIZE:
            self.flush()

    def flush(self):
        """Write buffer to Parquet file."""
        if not self.buffer:
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.buffer)

        # Type enforcement for string columns
        df["hex"] = df["hex"].astype("string")
        df["icao"] = df["icao"].astype("string")
        df["date"] = df["date"].astype("string")
        df["squawk"] = df["squawk"].astype("string")
        df["source_type"] = df["source_type"].astype("string")

        # Convert to Arrow Table
        table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)

        # Initialize writer on first flush
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.out_file,
                schema=SCHEMA,
                compression="zstd",
                compression_level=3
            )

        self.writer.write_table(table)
        self.total_rows += len(df)
        self.buffer.clear()

        logger.info(f"  ✓ Flushed batch → {self.total_rows:,} points total")

    def close(self):
        """Flush remaining buffer and close writer."""
        self.flush()

        if self.writer:
            self.writer.close()

        logger.info(f"  ✓ File complete: {self.total_rows:,} points written")

# ============================================================
# DAY PROCESSOR
# ============================================================

def process_day(day_dir: Path):
    """Process all trajectory files for one day."""
    date = extract_date(day_dir.name)
    out_file = OUT_DIR / f"istanbul_{date}_adsb.parquet"

    logger.info(f"\n{'='*70}")
    logger.info(f" Processing: {date}")
    logger.info(f" Directory: {day_dir.name}")
    logger.info(f" Output: {out_file.name}")
    logger.info(f"{'='*70}")

    # Find all trace files recursively (searches through all nested subdirectories)
    # Pattern: v2025.03.12-*/traces/61/trace_full_*.json
    logger.info(f"🔍 Searching recursively in {day_dir.name}...")
    files = list(day_dir.rglob("trace_full_*.json*"))

    if not files:
        logger.warning(f"  No trace files found in {day_dir.name}")
        return

    stats.files_total += len(files)

    # Show directory structure depth
    if files:
        sample_path = files[0]
        depth = len(sample_path.relative_to(day_dir).parts)
        logger.info(f"✓ Found {len(files):,} trajectory files")
        logger.info(f"  Structure depth: {depth} levels (e.g., traces/61/trace_full_*.json)")
    else:
        logger.info(f"✓ Found {len(files):,} trajectory files")

    writer = BatchWriter(out_file)
    valid_trajectories = 0

    for idx, file_path in enumerate(files, 1):
        # Read JSON
        data = read_json_smart(file_path)
        if not data:
            continue

        # Extract metadata
        hex_code = extract_hex(file_path)
        icao = data.get("icao", "")

        # Validate timestamp
        base_ts = normalize_timestamp(data.get("timestamp"))
        if base_ts is None:
            stats.files_no_timestamp += 1
            logger.debug(f"No timestamp in {file_path.name}")
            continue

        # Validate trace
        trace = data.get("trace")
        if not trace:
            stats.files_no_trace += 1
            logger.debug(f"No trace data in {file_path.name}")
            continue

        # Explode trace to rows
        rows = explode_trace(trace, hex_code, icao, date, base_ts)

        if not rows:
            continue

        valid_trajectories += 1
        stats.files_valid += 1

        writer.add_rows(rows)

        # Progress logging every 1000 files
        if idx % 1000 == 0:
            logger.info(
                f"  📊 Progress: {idx:,}/{len(files):,} files | "
                f"Valid: {valid_trajectories:,} | "
                f"Points written: {stats.points_written:,} | "
                f"Points examined: {stats.points_examined:,}"
            )

    writer.close()

    logger.info(f"✓ Day complete: {valid_trajectories:,} valid trajectories")

# ============================================================
# MAIN
# ============================================================

def main():
    """Run the Istanbul ADS-B processing pipeline."""
    logger.info("\n" + "="*80)
    logger.info("🛫 ISTANBUL ADS-B TRAJECTORY PROCESSOR - PRODUCTION VERSION")
    logger.info("="*80)
    logger.info(f" Source directory: {BASE_PATH}")
    logger.info(f" Output directory: {OUT_DIR}")
    logger.info(f" Radius filter: {RADIUS_KM} km from Istanbul")
    logger.info(f" Batch size: {BATCH_SIZE:,} rows")
    logger.info(f" Debug mode: {'ENABLED' if DEBUG_MODE else 'DISABLED'}")
    logger.info("="*80)

    # OPTION 1: Process by day directories (if structure is /traj/vYYYY.MM.DD/...)
    day_dirs = sorted([d for d in BASE_PATH.iterdir() if d.is_dir()])

    if day_dirs:
        logger.info(f"\n✓ Found {len(day_dirs)} top-level directory(ies)")

        # Quick scan to show total scope
        logger.info(f"\n🔍 Quick scan of directory structure:")
        total_files_estimate = 0
        for i, d in enumerate(day_dirs[:3]):  # Show first 3
            sample_files = list(d.rglob("trace_full_*.json*"))
            total_files_estimate += len(sample_files)
            logger.info(f"  {d.name}: ~{len(sample_files):,} files")

        if len(day_dirs) > 3:
            logger.info(f"  ... and {len(day_dirs) - 3} more directories")

        logger.info(f"\n🚀 Starting processing...\n")

        # Process each day
        for day_dir in day_dirs:
            try:
                process_day(day_dir)
            except KeyboardInterrupt:
                logger.warning("\n  Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"✗ Failed processing {day_dir.name}: {e}", exc_info=True)
    else:
        # OPTION 2: No day directories found - process all files directly from BASE_PATH
        logger.info(f"\nNo subdirectories found. Processing all files from {BASE_PATH}")

        try:
            # Create a single output file for all data
            out_file = OUT_DIR / "istanbul_all_adsb.parquet"

            logger.info(f"\n{'='*70}")
            logger.info(f"Processing: All trajectory files")
            logger.info(f"Output: {out_file.name}")
            logger.info(f"{'='*70}")

            # Find all trace files recursively
            files = list(BASE_PATH.rglob("trace_full_*.json*"))
            stats.files_total = len(files)

            logger.info(f"Found {len(files):,} trajectory files (searched recursively)")

            writer = BatchWriter(out_file)
            valid_trajectories = 0

            for idx, file_path in enumerate(files, 1):
                # Read JSON
                data = read_json_smart(file_path)
                if not data:
                    continue

                # Extract metadata
                hex_code = extract_hex(file_path)
                icao = data.get("icao", "")

                # Try to extract date from parent directory structure
                date = extract_date(file_path.parent.name)
                if date == "unknown":
                    # Try grandparent
                    date = extract_date(file_path.parent.parent.name)
                if date == "unknown":
                    # Fallback to "unknown"
                    date = "unknown"

                # Validate timestamp
                base_ts = normalize_timestamp(data.get("timestamp"))
                if base_ts is None:
                    stats.files_no_timestamp += 1
                    logger.debug(f"No timestamp in {file_path.name}")
                    continue

                # Validate trace
                trace = data.get("trace")
                if not trace:
                    stats.files_no_trace += 1
                    logger.debug(f"No trace data in {file_path.name}")
                    continue

                # Explode trace to rows
                rows = explode_trace(trace, hex_code, icao, date, base_ts)

                if not rows:
                    continue

                valid_trajectories += 1
                stats.files_valid += 1

                writer.add_rows(rows)

                # Progress logging every 1000 files
                if idx % 1000 == 0:
                    logger.info(
                        f"  Progress: {idx:,}/{len(files):,} files | "
                        f"Valid: {valid_trajectories:,} | "
                        f"Points written: {stats.points_written:,} | "
                        f"Points examined: {stats.points_examined:,}"
                    )

            writer.close()
            logger.info(f" Processing complete: {valid_trajectories:,} valid trajectories")

        except KeyboardInterrupt:
            logger.warning("\n  Processing interrupted by user")
        except Exception as e:
            logger.error(f" Failed processing: {e}", exc_info=True)

    # Final summary
    stats.log_summary()

    logger.info("✓ Pipeline complete\n")

if __name__ == "__main__":
    main()