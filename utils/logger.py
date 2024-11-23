import logging
from config.settings import LOG_FILE


def setup_logger():
    logging.basicConfig(
        filename=LOG_FILE,
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    return logging.getLogger()
