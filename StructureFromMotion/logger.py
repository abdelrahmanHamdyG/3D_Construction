import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", log_name=None, log_level=logging.INFO):
        """
        Initializes the logger.

        :param log_dir: Directory to store log files.
        :param log_name: Name of the log file. Defaults to 'app_<date>.log'.
        :param log_level: Logging level (default: logging.INFO).
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)  # Ensure log directory exists
        
        if not log_name:
            log_name = f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
        
        self.log_file = os.path.join(self.log_dir, log_name)
        
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)
        
        # Prevent adding duplicate handlers
        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(self._get_formatter())
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._get_formatter())
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def _get_formatter(self):
        """
        Returns a logging formatter.
        """
        return logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    def get_logger(self):
        """
        Returns the logger instance.
        """
        return self.logger


# Example usage
if __name__ == "__main__":
    logger = Logger(log_dir="my_logs", log_name="example.log").get_logger()
    
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
