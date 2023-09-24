from config.config import logger, WEI_PATH, download_data_gdown, WEIDATA_URL, download_data_ucr, MULTI_WEI_PATH, MULTIWEIDATA_URL
import gdown
import zipfile
from pathlib import Path
import tempfile
import argparse

def download_wei():
    # check if wei data is already downloaded
    if len(list(WEI_PATH.glob("*"))) > 0:
        logger.info("Wei data already downloaded")
        return
    logger.info("Downloading wei data")
    download_data_gdown(
        WEIDATA_URL,
        WEI_PATH,
    )

def download_multi_wei():
    # check if wei data is already downloaded
    if len(list(MULTI_WEI_PATH.glob("*"))) > 0:
        logger.info("Wei data already downloaded")
        return
    logger.info("Downloading wei data")
    download_data_gdown(
        MULTIWEIDATA_URL,
        MULTI_WEI_PATH,
    )


def download_ucr():
    download_data_ucr()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="wei",
        help="Which data to download,'all','wei', 'ucr', ....",
    )
    args = parser.parse_args()
    if args.data == "wei":
        download_wei()
    elif args.data == "multi_wei":
        download_multi_wei()
    elif args.data == "ucr":
        download_ucr()
    elif args.data == "all":
        download_wei()
        download_ucr()
    else:
        raise ValueError(f"Unknown data {args.data}")
