import logging
import os
from logging import handlers

# LOG_LEVEL = logging.INFO
LOG_LEVEL = logging.DEBUG
CURRENT_PATH = os.getcwd()
LOG_PATH = os.path.join(CURRENT_PATH, "logs")
LOG_FILE = "agent_run.log"


class Logger:
    """
    Set up a logger with specified settings.
    This logger class ensures that only one instance of logger is created per name, avoiding multiple handler attachments.
    """
    loggers = {}  # Class variable to store logger instances, ensuring singleton pattern per logger name.
    
    def __new__(cls, *args, **kwargs):
        # Retrieve the 'name' parameter, if it exists, else default to None
        name = kwargs.get('name', None)

        # Check if the logger with the given name already exists to avoid creating multiple instances of logger handlers.
        if name in cls.loggers:
            return cls.loggers[name]

        # If logger does not exist, create a new instance and store it in the class dictionary.
        instance = super(Logger, cls).__new__(cls)
        cls.loggers[name] = instance
        return instance

    def __init__(self, name=None, level=logging.DEBUG,
                 log_path="./logs", log_file="my_logs.log",
                 when="d", interval=1, backup_count=7, encoding='utf-8'):
        # Avoid reinitializing if the instance has already been initialized.
        if hasattr(self, 'is_initialized'):
            return
        self.is_initialized = True

        # Create the logger and set its level.
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Ensure log directory exists, create if necessary.
        if not os.path.isdir(log_path):
            try:
                os.makedirs(log_path)
            except Exception as e:
                # Handle exceptions if unable to create directory and raise an informative error.
                raise RuntimeError(f"Failed to create log directory: {log_path}") from e
        # ---
        # Setting up the StreamHandler for console logging.
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter(self.set_format()))
        # sh.stream.encoding = encoding  # Explicitly set encoding
        self.logger.addHandler(sh)
        # ---
        # Setting up the TimedRotatingFileHandler for file logging.
        rh = handlers.TimedRotatingFileHandler(
            os.path.join(log_path, log_file), when, interval, backup_count, encoding=encoding
        )
        rh.setLevel(level)
        rh.setFormatter(logging.Formatter(self.set_format()))
        self.logger.addHandler(rh)

    @staticmethod
    def set_format():
        """
        Sets the log format without color codes, suitable for both console and file logs.
        Format includes log level, timestamp, logger name, module, and function.
        """
        # return "[%(levelname)s][%(asctime)s][%(name)s/%(module)s/%(funcName)s] %(message)s"
        return "[%(levelname)s][%(asctime)s][%(module)s/%(funcName)s] %(message)s"

def main():
    current_path = os.getcwd()
    save_path = os.path.join(current_path, "logs")
    print("> Save Path:",save_path)
    logger = Logger(name="MyLogger", level=logging.DEBUG, log_path=save_path, log_file="MyApp.log")
    logger.logger.info("This is a info message.")
    logger.logger.debug("This is a debug message.")
    logger.logger.error("This is a error message.")

if __name__ == "__main__":
    main()
