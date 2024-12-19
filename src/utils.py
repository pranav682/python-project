import logging
import re

class LoggerFactory:
    @staticmethod
    def create_logger(level=logging.INFO):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        return logging.getLogger()

class TextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        text = text.strip()
        return text

