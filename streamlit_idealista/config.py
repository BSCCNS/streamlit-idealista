import json
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from upath import UPath

# Load environment variables from .env file if it exists
load_dotenv()

# cloud
DATA_DIR_FSSPEC_BASE_URL = os.getenv("DATA_DIR_FSSPEC_BASE_URL", "")
# DATA_DIR_FSSPEC_USER
# DATA_DIR_FSSPEC_PASS  # (user, password) is not stored to avoid exposure.

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = UPath(os.getenv("DATA_DIR_FSSPEC_URI", f"""file://{PROJ_ROOT / "data"}"""),
                 base_url=DATA_DIR_FSSPEC_BASE_URL,
                 auth=(os.getenv("DATA_DIR_FSSPEC_USER", ""),
                       os.getenv("DATA_DIR_FSSPEC_PASS", "")
                      ),
                )

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Custom paths based on your specific project structure

INPUT_DATA_PATH = PROCESSED_DATA_DIR / "full/02-metricas-de-mercado-extended-ad-2010-q2-2024_utf8_pivot.csv"
INPUT_DTYPES_COUPLED_JSON_PATH = PROCESSED_DATA_DIR / "dtypes-coupled.json"
INPUT_SUPERILLES_INTERVENTIONS_GEOJSON = PROCESSED_DATA_DIR / "full/CENSUSTRACT_superilles.geojson"
INPUT_OPERATION_TYPES_PATH = PROCESSED_DATA_DIR / "full/dimension-table_data-t-adoperations_utf8.csv"
INPUT_TYPOLOGY_TYPES_PATH = PROCESSED_DATA_DIR / "full/dimension-table_data-t-adtypologies_utf8.csv"

INPUT_INE_CENSUSTRACT_GEOJSON = PROCESSED_DATA_DIR / "censustracts_geometries.geojson"


SAVE_OUTPUT = False
OUTPUT_DATA_PATH = PROCESSED_DATA_DIR / "full/"

# Log the important paths
logger.info(f"Input data path: {INPUT_DATA_PATH}")
logger.info(f"Input JSON path: {INPUT_DTYPES_COUPLED_JSON_PATH}")
logger.info(f"Input superilles interventions GeoJSON: {INPUT_SUPERILLES_INTERVENTIONS_GEOJSON}")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
