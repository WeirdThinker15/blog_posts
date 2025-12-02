import logging
import logging.config
import os

def setup_logging(default_level=logging.INFO):
    """
    Loads logging configuration from file or environment variables
    """

    if os.path.exists("logging.conf"):
        logging.config.fileConfig("logging.conf")
    else:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", default_level),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    return logging.getLogger("PGP-AUDIT")
