#!/usr/bin/env python3
"""
generate_combined_dataset.py
============================
Universal multi-sensor Chandrayaan-2 (PRADAN) data fusion pipeline.

Constructs a co-registered, multi-modal CSV dataset for YOLOv8 lunar
hazard detection from raw ISRO PDS4 archives (DFSAR, OHRC, TMC-2, IIRS).

Author :Team LAEP / developed by Abbas shaikh 
Version: 2.0.0
"""

from __future__ import annotations

import sys
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CRATER_NAME: str = "F2"                # Change per-crater or accept via CLI
N_SAMPLES: int = 1000                  # Number of output sample rows
RANDOM_SEED: int = 42                  # Reproducibility seed
OHRC_OOB_MAX_DIST_DEG: float = 0.05   # Max KDTree distance (degrees) for
                                       # OHRC spatial lookup before marking OOB
IIRS_OOB_MAX_DIST_DEG: float = 0.50   # Max KDTree distance (degrees) for
                                       # IIRS spatial lookup before marking OOB

# IIRS band-index constants (0-indexed into 256-band cube)
# Approximate mapping based on IIRS spectral range ~0.8-5.0 µm / 256 bands
IIRS_BAND_IDX_1500NM: int = 103        # Placeholder for ~1.5 µm
IIRS_BAND_IDX_2000NM: int = 155        # Placeholder for ~2.0 µm
IIRS_BAND_IDX_CONTINUUM: int = 130     # Continuum reference between 1.5 & 2.0 µm

# Derived paths
CRATER_DIR = BASE_DIR / "Dataset" / "data" / CRATER_NAME
OUTPUT_CSV = BASE_DIR / f"{CRATER_NAME}_combined_dataset.csv"

np.random.seed(RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DYNAMIC FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def _find_one(root: Path, pattern: str, label: str) -> Path:
    """Return the first file matching *pattern* under *root*, or raise."""
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"[FATAL] {label}: no file matching '{pattern}' found under {root}"
        )
    chosen = matches[0]
    log.info("Found %-18s: %s", label, chosen)
    return chosen


def _find_optional(root: Path, pattern: str, label: str) -> Optional[Path]:
    """Return the first match or None (no error)."""
    matches = sorted(root.rglob(pattern))
    if matches:
        log.info("Found %-18s: %s", label, matches[0])
        return matches[0]
    return None


def discover_files(crater_dir: Path) -> Dict[str, Optional[Path]]:
    """
    Locate all required and optional input files inside *crater_dir*.

    Returns a dict keyed by logical name -> resolved Path (or None).
    Raises FileNotFoundError for any mandatory file that is missing.
    """
    log.info("=" * 70)
    log.info("FILE DISCOVERY — scanning: %s", crater_dir)
    log.info("=" * 70)

    if not crater_dir.is_dir():
        raise FileNotFoundError(
            f"[FATAL] Crater directory does not exist: {crater_dir}"
        )

    files: Dict[str, Optional[Path]] = {}

    # ── Mandatory DFSAR geometry CSVs ─────────────────────────────────────
    files["dfsar_gri"] = _find_one(crater_dir, "*_g_gri_*.csv",  "DFSAR GRI")
    files["dfsar_sli"] = _find_one(crater_dir, "*_g_sli_*.csv",  "DFSAR SLI")
    files["dfsar_oat"] = _find_one(crater_dir, "*_g_oat_*.csv",  "DFSAR OAT")

    # ── OHRC raster: prefer GeoTIFF, fallback to raw .img ────────────────
    ohrc_tif = _find_optional(crater_dir, "ohrc_output.tif", "OHRC GeoTIFF")
    if ohrc_tif is None:
        ohrc_tif = _find_optional(crater_dir, "ch2_ohr_*.tif", "OHRC TIF alt")

    if ohrc_tif:
        files["ohrc_tif"] = ohrc_tif
        files["ohrc_img"] = None
        files["ohrc_xml"] = None
        files["ohrc_grd"] = None
    else:
        # Raw PDS4 binary path
        files["ohrc_tif"] = None
        files["ohrc_img"] = _find_one(
            crater_dir, "ch2_ohr_*_d_img_*.img", "OHRC Raw Image"
        )
        files["ohrc_xml"] = _find_one(
            crater_dir, "ch2_ohr_*_d_img_*.xml", "OHRC XML Label"
        )
        files["ohrc_grd"] = _find_one(
            crater_dir, "ch2_ohr_*_g_grd_*.csv", "OHRC Grid CSV"
        )

    # ── Optional TMC-2 DEM rasters ────────────────────────────────────────
    files["tmc2_dem"]   = _find_optional(crater_dir, "tmc2_dem.tif",   "TMC-2 DEM")
    files["tmc2_slope"] = _find_optional(crater_dir, "tmc2_slope.tif", "TMC-2 Slope")

    if files["tmc2_dem"] is None or files["tmc2_slope"] is None:
        log.warning(
            "TMC-2 DEM rasters not found. "
            "Utilizing baseline synthetic values for Elevation & Slope."
        )

    # ── Optional IIRS hyperspectral data ──────────────────────────────────
    files["iirs_hdr"] = _find_optional(crater_dir, "ch2_iir_*.hdr",       "IIRS HDR")
    files["iirs_qub"] = _find_optional(crater_dir, "ch2_iir_*.qub",       "IIRS QUB")
    files["iirs_grd"] = _find_optional(crater_dir, "ch2_iir_*_g_grd_*.csv", "IIRS Grid CSV")

    if files["iirs_hdr"] is None or files["iirs_qub"] is None or files["iirs_grd"] is None:
        log.warning(
            "IIRS data files not fully found. "
            "IIRS spectral features will be set to NaN."
        )

    log.info("=" * 70)
    return files


# ═══════════════════════════════════════════════════════════════════════════
# 2. MASTER SPATIAL GRID CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_master_grid(gri_path: Path, n_samples: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load the DFSAR ground-range-image geometry CSV, uniformly subsample
    to *n_samples* rows, and produce the master coordinate grid with
    Sample_ID primary keys.
    """
    log.info("Building master grid from GRI (%d target samples)...", n_samples)

    df_gri = pd.read_csv(gri_path)
    total_rows = len(df_gri)
    log.info("  GRI total rows: %d", total_rows)

    # Uniform subsample indices
    indices = np.linspace(0, total_rows - 1, n_samples, dtype=int)
    df_sub = df_gri.iloc[indices].reset_index(drop=True)

    # Rename to output schema
    df = pd.DataFrame({
        "Sample_ID": [f"sample_{i+1:04d}" for i in range(n_samples)],
        "Latitude_deg_SAR":       df_sub["Latitude(deg.)"].values,
        "Longitude_deg_SAR":      df_sub["Longitude(deg.)"].values,
        "Incidence_Angle_deg_SAR": df_sub["Incidence_Angle(deg.)"].values,
        "Ground_Range_m_SAR":     df_sub["Range(m)"].values,
    })

    log.info("  Master grid shape: %s", df.shape)
    return df, indices


# ═══════════════════════════════════════════════════════════════════════════
# 3. DFSAR GEOMETRY & ORBIT JOIN
# ═══════════════════════════════════════════════════════════════════════════

def join_geometry(
    df: pd.DataFrame,
    master_indices: np.ndarray,
    sli_path: Path,
    oat_path: Path,
) -> pd.DataFrame:
    """
    Align slant-range and orbit-state-vector data to the master grid.

    - SLI has ~269K rows vs GRI 82K -> proportional index mapping.
    - OAT has ~205 rows -> evenly distribute across the master grid.
    """
    n_samples = len(df)

    # ── Slant Range ───────────────────────────────────────────────────────
    log.info("Joining DFSAR slant-range geometry (SLI)...")
    df_sli = pd.read_csv(sli_path)
    sli_total = len(df_sli)

    # Map each master GRI index -> proportional SLI index
    gri_total = master_indices.max() + 1  # original GRI row count
    sli_mapped_indices = np.clip(
        (master_indices.astype(np.float64) / gri_total * sli_total).astype(int),
        0,
        sli_total - 1,
    )
    df["Slant_Range_m_SAR"] = df_sli["Slant_Range(m)"].iloc[
        sli_mapped_indices
    ].values

    log.info("  SLI total rows: %d -> mapped %d samples", sli_total, n_samples)

    # ── Orbit State Vectors ───────────────────────────────────────────────
    log.info("Joining DFSAR orbit state vectors (OAT)...")
    df_oat = pd.read_csv(oat_path)
    oat_total = len(df_oat)

    # Evenly map each of the n_samples to the nearest OAT record
    oat_mapped_indices = np.clip(
        np.linspace(0, oat_total - 1, n_samples, dtype=int),
        0,
        oat_total - 1,
    )
    df["Sat_Pos_X_m_SAR_OAT"] = df_oat["x(m)"].iloc[oat_mapped_indices].values
    df["Sat_Pos_Y_m_SAR_OAT"] = df_oat["y(m)"].iloc[oat_mapped_indices].values
    df["Sat_Pos_Z_m_SAR_OAT"] = df_oat["z(m)"].iloc[oat_mapped_indices].values

    log.info("  OAT total rows: %d -> mapped %d samples", oat_total, n_samples)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. OHRC OPTICAL REFLECTANCE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _parse_ohrc_xml(xml_path: Path) -> Tuple[int, int, str]:
    """
    Parse the PDS4 XML label to extract image dimensions and data type.

    Returns (n_lines, n_samples_per_line, numpy_dtype_string).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # PDS4 namespaces
    ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}

    array_2d = root.find(".//pds:Array_2D_Image", ns)
    if array_2d is None:
        raise ValueError(f"No Array_2D_Image element found in {xml_path}")

    axes = {}
    for ax in array_2d.findall("pds:Axis_Array", ns):
        name = ax.find("pds:axis_name", ns).text.strip()
        elems = int(ax.find("pds:elements", ns).text.strip())
        axes[name] = elems

    n_lines = axes.get("Line", 0)
    n_samples = axes.get("Sample", 0)

    dt_text = array_2d.find("pds:Element_Array/pds:data_type", ns).text.strip()
    dtype_map = {
        "UnsignedByte":       "uint8",
        "SignedByte":         "int8",
        "UnsignedMSB2":      ">u2",
        "SignedMSB2":         ">i2",
        "UnsignedMSB4":      ">u4",
        "SignedMSB4":         ">i4",
        "IEEE754MSBSingle":   ">f4",
        "IEEE754MSBDouble":   ">f8",
    }
    np_dtype = dtype_map.get(dt_text, "uint8")

    log.info(
        "  OHRC XML parsed -> %d lines x %d samples, dtype=%s (%s)",
        n_lines, n_samples, np_dtype, dt_text,
    )
    return n_lines, n_samples, np_dtype


def extract_ohrc_reflectance(
    df: pd.DataFrame,
    files: Dict[str, Optional[Path]],
    max_dist_deg: float = OHRC_OOB_MAX_DIST_DEG,
) -> pd.DataFrame:
    """
    Sample OHRC optical reflectance for each (Lat, Lon) in the master grid.

    Path A -- GeoTIFF via rasterio.
    Path B -- Raw PDS4 .img via KDTree + numpy.memmap.
    """
    lats = df["Latitude_deg_SAR"].values
    lons = df["Longitude_deg_SAR"].values
    n = len(df)

    # ── Path A: GeoTIFF ──────────────────────────────────────────────────
    if files.get("ohrc_tif") is not None:
        import rasterio  # conditional import

        log.info("Extracting OHRC reflectance via GeoTIFF...")
        reflectance = np.full(n, 0.0)
        with rasterio.open(files["ohrc_tif"]) as src:
            coords = list(zip(lons, lats))
            for i, val in enumerate(src.sample(coords)):
                try:
                    reflectance[i] = float(val[0])
                except (IndexError, ValueError):
                    reflectance[i] = 0.0

        df["Reflectance_OHRC"] = reflectance
        log.info("  GeoTIFF sampling complete.")
        return df

    # ── Path B: Raw PDS4 .img + KDTree ───────────────────────────────────
    from scipy.spatial import cKDTree

    log.info("Extracting OHRC reflectance via raw PDS4 .img + KDTree...")

    # 4a. Parse XML for image dimensions
    n_lines, n_line_samples, np_dtype = _parse_ohrc_xml(files["ohrc_xml"])

    # 4b. Load OHRC coordinate grid
    log.info("  Loading OHRC coordinate grid CSV...")
    df_grd = pd.read_csv(files["ohrc_grd"])
    log.info("  OHRC grid rows: %d", len(df_grd))

    grd_lats  = df_grd["Latitude"].values
    grd_lons  = df_grd["Longitude"].values   # 0-360 degree convention
    grd_pixel = df_grd["Pixel"].values.astype(int)
    grd_scan  = df_grd["Scan"].values.astype(int)

    # 4c. Build KDTree over (Lat, Lon_360)
    log.info("  Building KDTree over OHRC grid (%d points)...", len(grd_lats))
    tree = cKDTree(np.column_stack([grd_lats, grd_lons]))

    # 4d. Normalize query longitudes to 0-360
    query_lons_360 = lons % 360
    query_points = np.column_stack([lats, query_lons_360])

    # 4e. Query nearest neighbors
    distances, nn_indices = tree.query(query_points, k=1)

    # 4f. Memory-map the .img file
    log.info("  Memory-mapping OHRC .img (%d x %d)...", n_lines, n_line_samples)
    img_mmap = np.memmap(
        files["ohrc_img"],
        dtype=np_dtype,
        mode="r",
        shape=(n_lines, n_line_samples),
    )

    # Determine max pixel value for normalization
    dtype_info = np.iinfo(np.dtype(np_dtype)) if np.issubdtype(
        np.dtype(np_dtype), np.integer
    ) else None
    max_val = float(dtype_info.max) if dtype_info else 1.0

    # 4g. Extract reflectance with OOB safety guard
    reflectance = np.full(n, 0.0)
    oob_count = 0

    for i in range(n):
        if distances[i] > max_dist_deg:
            # Point falls outside OHRC coverage strip
            reflectance[i] = 0.0
            oob_count += 1
            continue

        px = grd_pixel[nn_indices[i]]
        sc = grd_scan[nn_indices[i]]

        # Bounds-check against image dimensions
        if 0 <= sc < n_lines and 0 <= px < n_line_samples:
            raw_val = float(img_mmap[sc, px])
            reflectance[i] = raw_val / max_val
        else:
            reflectance[i] = 0.0
            oob_count += 1

    df["Reflectance_OHRC"] = reflectance

    if oob_count > 0:
        log.warning(
            "  %d / %d samples fell outside OHRC coverage "
            "(dist > %.4f deg) -> Reflectance set to 0.0",
            oob_count, n, max_dist_deg,
        )
    else:
        log.info("  All %d samples within OHRC coverage.", n)

    log.info("  Raw .img reflectance extraction complete.")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. TMC-2 DEM & SLOPE FEATURE HOOKS
# ═══════════════════════════════════════════════════════════════════════════

def extract_tmc2_metrics(
    df: pd.DataFrame,
    files: Dict[str, Optional[Path]],
) -> pd.DataFrame:
    """
    Extract terrain elevation and slope.

    Primary path  -- sample from real GeoTIFFs (tmc2_dem.tif, tmc2_slope.tif).
    Fallback path -- generate synthetic uniform-random placeholders.
    """
    n = len(df)
    lats = df["Latitude_deg_SAR"].values
    lons = df["Longitude_deg_SAR"].values

    dem_path   = files.get("tmc2_dem")
    slope_path = files.get("tmc2_slope")

    if dem_path is not None and slope_path is not None:
        # ── Real raster extraction ────────────────────────────────────────
        import rasterio

        log.info("Extracting TMC-2 elevation & slope from GeoTIFFs...")
        coords = list(zip(lons, lats))

        elevation = np.full(n, np.nan)
        with rasterio.open(dem_path) as src:
            for i, val in enumerate(src.sample(coords)):
                try:
                    elevation[i] = float(val[0])
                except (IndexError, ValueError):
                    pass

        slope = np.full(n, np.nan)
        with rasterio.open(slope_path) as src:
            for i, val in enumerate(src.sample(coords)):
                try:
                    slope[i] = float(val[0])
                except (IndexError, ValueError):
                    pass

        df["Elevation_m_TMC2"] = elevation
        df["Slope_deg_TMC2"]   = slope

        # Fill any remaining NaN with synthetic fallback
        nan_elev  = df["Elevation_m_TMC2"].isna().sum()
        nan_slope = df["Slope_deg_TMC2"].isna().sum()
        if nan_elev > 0 or nan_slope > 0:
            log.warning(
                "  %d elevation / %d slope NaN values filled with synthetic data.",
                nan_elev, nan_slope,
            )
            df["Elevation_m_TMC2"] = df["Elevation_m_TMC2"].fillna(
                pd.Series(np.random.uniform(-1980.0, -1800.0, n))
            )
            df["Slope_deg_TMC2"] = df["Slope_deg_TMC2"].fillna(
                pd.Series(np.random.uniform(2.0, 25.0, n))
            )

        log.info("  TMC-2 real raster extraction complete.")
    else:
        # ── Synthetic fallback ────────────────────────────────────────────
        log.info("Generating synthetic TMC-2 elevation & slope placeholders...")
        df["Elevation_m_TMC2"] = np.random.uniform(-1980.0, -1800.0, n)
        df["Slope_deg_TMC2"]   = np.random.uniform(2.0, 25.0, n)
        log.info("  Synthetic TMC-2 data generated.")

    # CPR is always synthetic in the current pipeline phase
    log.info("Generating synthetic CPR_DFSAR values...")
    df["CPR_DFSAR"] = np.random.uniform(0.20, 1.50, n)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 6. IIRS HYPERSPECTRAL FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_iirs_metrics(
    df: pd.DataFrame,
    files: Dict[str, Optional[Path]],
    max_dist_deg: float = IIRS_OOB_MAX_DIST_DEG,
) -> pd.DataFrame:
    """
    Extract IIRS hyperspectral features for each (Lat, Lon) in the master grid.

    Uses the `spectral` library to read individual pixels from the ENVI
    datacube on disk (no full 4 GB load) and calculates ice-detection
    relevant spectral indices.

    Columns added:
      - Reflectance_IIRS_1500nm
      - Reflectance_IIRS_2000nm
      - IIRS_Band_Ratio
      - IIRS_H2O_Absorption_Depth
    """
    n = len(df)
    iirs_cols = [
        "Reflectance_IIRS_1500nm",
        "Reflectance_IIRS_2000nm",
        "IIRS_Band_Ratio",
        "IIRS_H2O_Absorption_Depth",
    ]

    # ── Guard: if any IIRS file is missing, fill NaN and return ───────────
    hdr_path = files.get("iirs_hdr")
    qub_path = files.get("iirs_qub")
    grd_path = files.get("iirs_grd")

    if hdr_path is None or qub_path is None or grd_path is None:
        log.warning("IIRS files incomplete — all IIRS columns set to NaN.")
        for col in iirs_cols:
            df[col] = np.nan
        return df

    from scipy.spatial import cKDTree
    import spectral.io.envi as envi

    log.info("Extracting IIRS spectral metrics...")

    # ── 6a. Load IIRS geometry grid ───────────────────────────────────────
    log.info("  Loading IIRS geometry grid CSV...")
    df_grd = pd.read_csv(grd_path)
    log.info("  IIRS grid rows: %d", len(df_grd))

    grd_lats  = df_grd["Latitude"].values
    grd_lons  = df_grd["Longitude"].values   # 0-360 degree convention
    grd_pixel = df_grd["Pixel"].values.astype(int)
    grd_scan  = df_grd["Scan"].values.astype(int)

    # ── 6b. Build KDTree over IIRS (Lat, Lon) ────────────────────────────
    log.info("  Building KDTree over IIRS grid (%d points)...", len(grd_lats))
    tree = cKDTree(np.column_stack([grd_lats, grd_lons]))

    # ── 6c. Normalize query longitudes to 0-360 ─────────────────────────
    lats = df["Latitude_deg_SAR"].values
    lons = df["Longitude_deg_SAR"].values
    query_lons_360 = lons % 360
    query_points = np.column_stack([lats, query_lons_360])

    # ── 6d. Query nearest neighbors ──────────────────────────────────────
    distances, nn_indices = tree.query(query_points, k=1)

    # ── 6e. Open ENVI datacube (header + binary) ─────────────────────────
    log.info("  Opening IIRS ENVI datacube via spectral library...")
    img = envi.open(str(hdr_path), str(qub_path))
    n_bands = img.shape[2]  # expected 256
    log.info(
        "  IIRS cube shape: %d lines x %d samples x %d bands",
        img.shape[0], img.shape[1], n_bands,
    )

    # Validate band indices against actual band count
    band_1500 = min(IIRS_BAND_IDX_1500NM, n_bands - 1)
    band_2000 = min(IIRS_BAND_IDX_2000NM, n_bands - 1)
    band_cont = min(IIRS_BAND_IDX_CONTINUUM, n_bands - 1)

    # ── 6f. Extract per-pixel spectra and compute features ───────────────
    refl_1500  = np.full(n, np.nan)
    refl_2000  = np.full(n, np.nan)
    band_ratio = np.full(n, np.nan)
    h2o_depth  = np.full(n, np.nan)

    oob_count = 0
    n_lines   = img.shape[0]  # scan dimension
    n_samples = img.shape[1]  # pixel dimension

    for i in range(n):
        if distances[i] > max_dist_deg:
            # Point falls outside IIRS coverage
            oob_count += 1
            continue

        px = grd_pixel[nn_indices[i]]
        sc = grd_scan[nn_indices[i]]

        # Bounds-check against datacube dimensions
        if not (0 <= sc < n_lines and 0 <= px < n_samples):
            oob_count += 1
            continue

        # Read single pixel spectrum from disk (no full cube load)
        try:
            spectrum = img.read_pixel(sc, px)  # shape: (n_bands,)
        except Exception as exc:
            log.debug("  read_pixel(%d, %d) failed: %s", sc, px, exc)
            continue

        val_1500 = float(spectrum[band_1500])
        val_2000 = float(spectrum[band_2000])
        val_cont = float(spectrum[band_cont])

        refl_1500[i] = val_1500
        refl_2000[i] = val_2000

        # Band ratio (division-by-zero guard)
        if val_2000 != 0.0:
            band_ratio[i] = val_1500 / val_2000

        # Continuum-removed absorption depth proxy:
        #   depth = 1 - (2 * R_absorption) / (R_left_shoulder + R_right_shoulder)
        # Using 1.5 µm as left shoulder, continuum band as right shoulder,
        # and 2.0 µm as the absorption band.
        denom = val_1500 + val_cont
        if denom != 0.0:
            h2o_depth[i] = 1.0 - (2.0 * val_2000) / denom

    df["Reflectance_IIRS_1500nm"]    = refl_1500
    df["Reflectance_IIRS_2000nm"]    = refl_2000
    df["IIRS_Band_Ratio"]            = band_ratio
    df["IIRS_H2O_Absorption_Depth"]  = h2o_depth

    if oob_count > 0:
        log.warning(
            "  %d / %d samples fell outside IIRS coverage "
            "(dist > %.4f deg or pixel OOB) -> values set to NaN",
            oob_count, n, max_dist_deg,
        )
    else:
        log.info("  All %d samples matched to IIRS pixels.", n)

    valid_count = np.count_nonzero(~np.isnan(refl_1500))
    log.info("  IIRS extraction complete: %d / %d valid pixels.", valid_count, n)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 7. RULE-BASED HAZARD CLASSIFICATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def classify_hazards(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign Hazard_Class_Label using vectorized conditional rules.

    Priority order (first match wins):
      1. Slope > 15 deg          -> steep_slope_hazard
      2. Reflectance > 0.75      -> boulder
      3. Reflectance < 0.05      -> shadowing_obscuration
      4. CPR > 1.0               -> crater_rim
      5. else                    -> safe_flat
    """
    log.info("Running hazard classification engine...")

    conditions = [
        df["Slope_deg_TMC2"]   > 15.0,
        df["Reflectance_OHRC"] > 0.75,
        df["Reflectance_OHRC"] < 0.05,
        df["CPR_DFSAR"]        > 1.0,
    ]
    choices = [
        "steep_slope_hazard",
        "boulder",
        "shadowing_obscuration",
        "crater_rim",
    ]

    df["Hazard_Class_Label"] = np.select(conditions, choices, default="safe_flat")

    log.info("  Classification complete.")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 8. FORMATTING, VALIDATION & EXPORT
# ═══════════════════════════════════════════════════════════════════════════

# Column-specific decimal precision map
PRECISION_MAP = {
    "Latitude_deg_SAR":          4,
    "Longitude_deg_SAR":         4,
    "Incidence_Angle_deg_SAR":   2,
    "Ground_Range_m_SAR":        2,
    "Slant_Range_m_SAR":         2,
    "Sat_Pos_X_m_SAR_OAT":      2,
    "Sat_Pos_Y_m_SAR_OAT":      2,
    "Sat_Pos_Z_m_SAR_OAT":      2,
    "Reflectance_OHRC":          4,
    "Elevation_m_TMC2":          1,
    "Slope_deg_TMC2":            1,
    "CPR_DFSAR":                 2,
    "Reflectance_IIRS_1500nm":   4,
    "Reflectance_IIRS_2000nm":   4,
    "IIRS_Band_Ratio":           4,
    "IIRS_H2O_Absorption_Depth": 4,
}

# Canonical column order
COLUMN_ORDER = [
    "Sample_ID",
    "Latitude_deg_SAR",
    "Longitude_deg_SAR",
    "Incidence_Angle_deg_SAR",
    "Ground_Range_m_SAR",
    "Slant_Range_m_SAR",
    "Sat_Pos_X_m_SAR_OAT",
    "Sat_Pos_Y_m_SAR_OAT",
    "Sat_Pos_Z_m_SAR_OAT",
    "Reflectance_OHRC",
    "Elevation_m_TMC2",
    "Slope_deg_TMC2",
    "CPR_DFSAR",
    "Reflectance_IIRS_1500nm",
    "Reflectance_IIRS_2000nm",
    "IIRS_Band_Ratio",
    "IIRS_H2O_Absorption_Depth",
    "Hazard_Class_Label",
]


def format_and_export(df: pd.DataFrame, output_path: Path) -> None:
    """Apply precision formatting, reorder columns, and save to CSV."""
    log.info("Formatting & exporting dataset...")

    # Apply rounding
    for col, decimals in PRECISION_MAP.items():
        if col in df.columns:
            df[col] = df[col].round(decimals)

    # Enforce canonical column order
    df = df[COLUMN_ORDER]

    # Export
    df.to_csv(output_path, index=False)
    log.info("  Output saved -> %s", output_path)

    # ── Summary Statistics ────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 70)
    log.info("  Output file      : %s", output_path.resolve())
    log.info("  Total rows       : %d", len(df))
    log.info("  Total columns    : %d", len(df.columns))
    log.info("")
    log.info("  Hazard Class Distribution:")
    class_counts = df["Hazard_Class_Label"].value_counts()
    for label, count in class_counts.items():
        pct = count / len(df) * 100
        log.info("    %-28s  %5d  (%5.1f%%)", label, count, pct)
    log.info("")
    log.info("  Dataset Preview (head):")
    print(df.head().to_string(index=False))
    log.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """End-to-end pipeline execution."""
    log.info("=" * 64)
    log.info("  LAEP / ISRO -- Multi-Sensor Lunar Hazard Dataset Builder")
    log.info("  Crater: %s", CRATER_NAME)
    log.info("=" * 64)

    # Step 1 -- Discover files
    files = discover_files(CRATER_DIR)

    # Step 2 -- Build master spatial grid
    df, master_indices = build_master_grid(files["dfsar_gri"], N_SAMPLES)

    # Step 3 -- Join DFSAR geometry & orbit data
    df = join_geometry(df, master_indices, files["dfsar_sli"], files["dfsar_oat"])

    # Step 4 -- Extract OHRC reflectance
    df = extract_ohrc_reflectance(df, files)

    # Step 5 -- Extract TMC-2 DEM metrics (or fallback)
    df = extract_tmc2_metrics(df, files)

    # Step 6 -- Extract IIRS spectral metrics
    df = extract_iirs_metrics(df, files)

    # Step 7 -- Classify hazards
    df = classify_hazards(df)

    # Step 8 -- Format & export
    format_and_export(df, OUTPUT_CSV)

    log.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
