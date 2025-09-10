import configparser
import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(log_level=logging.INFO, name="root"):
    logger = logging.getLogger(name)

    # Avoid printing multiple logs
    logger.propagate = False

    root_dir = get_log_root_dir()
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not logger.handlers:
        log_handler = RotatingFileHandler(
            os.path.join(root_dir, name + ".txt"),
            maxBytes=100 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        logger.setLevel(log_level)
        log_format = logging.Formatter(
            "[%(asctime)-15s] [%(levelname)8s] %(filename)s:%(lineno)s - %(message)s"
        )
        log_handler.setFormatter(log_format)
        logger.addHandler(log_handler)
    else:
        logger.setLevel(log_level)
    return logger


def get_log_root_dir():
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../config/common.ini"
    )
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    root_dir = config["log"]["root_dir"]
    return root_dir
