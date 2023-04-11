import logging


LOG_FORMAT = "%(asctime)s: %(levelname)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_basic_logger(
    console=True,
    log_file=None,
    log_level=logging.DEBUG,
    log_format=LOG_FORMAT,
    date_format=DATE_FORMAT,
    logger_name="root"
):
    """
    Create a basic logger to log messages in console and/or file.

    :param console: bool output log in python console.
    :param log_file: str path of the file where to write the log.
    :param log_level: minimal level of log to show (logging.DEBUG, INFO, WARNING...)
    :param log_format: str format of log messages
    :param date_format: str format of datetime
    :param logger_name: str name of the logger. Default is root.
    :return: the logger
    """
    logger = logging.getLogger(logger_name)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)

    logger.setLevel(log_level)

    return logger
