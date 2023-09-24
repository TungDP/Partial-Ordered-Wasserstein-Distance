import logging
import sys
from pathlib import Path
import logging.config

from rich.logging import RichHandler
import gdown
import zipfile
import tempfile
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

WEIDATA_URL = "https://drive.google.com/uc?id=1W_g3NKVirJCYms0JX7fEjXd4dhPc5w30"
MULTIWEIDATA_URL = "https://drive.google.com/file/d/15bxa94ZMZS0ffxTFgFBsMDBnYY5UtDyM"
DIGITMOVING_URL = ""
UCRDATA_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"

BASE_DIR = Path(__file__).parent.parent.absolute()
PROJECT_PATH = BASE_DIR
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "Datasets")
COIN_PATH = Path(DATA_DIR, "COIN")
LOGS_DIR = Path(BASE_DIR, "logs")
CT_PATH = Path(DATA_DIR, "CrossTask")
YC_PATH = Path(DATA_DIR, "YouCook2")
WEIGHTS_PATH = Path(BASE_DIR, "weights")
WEI_PATH = Path(DATA_DIR, "wei")
MULTI_WEI_PATH = Path(DATA_DIR,"wei_dataset_feature")
UCR_PATH = Path(DATA_DIR, "UCR")

LOGS_DIR.mkdir(parents=True, exist_ok=True)

def download_data_gdown(url, dest_path):
    """Download data from url, extract it and save it to dest_path"""
    logger.info(f"Downloading data from {url} to {dest_path}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_path = tmp_path / "data.zip"
        gdown.download(url, str(zip_path), quiet=False)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_path)

def download_data_ucr():
    """Download UCR data, extract it and save it to dest_path"""
    import urllib.request
    from tqdm import tqdm
    if len(list(UCR_PATH.glob("*"))) > 0:
        logger.info("UCR data already downloaded")
        return
    logger.info(f"Downloading data from {UCRDATA_URL} to {UCR_PATH}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_path = tmp_path / "data.zip"
        with tqdm(unit='B', unit_scale=True, miniters=1, desc='Downloading') as t:
            urllib.request.urlretrieve(UCRDATA_URL, zip_path, reporthook=lambda b, bsize, tsize: t.update(bsize))
        password = 'someone'
        # Extract the contents of the downloaded zip file to the UCR data directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=UCR_PATH, pwd=bytes(password.encode('utf-8')))

#logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)
