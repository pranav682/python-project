import pandas as pd
import os
from .config import Config

class DataIngestion:
    def __init__(self, raw_data_file: str = Config.RAW_DATA_FILE):
        self.raw_data_file = raw_data_file

    def load_raw_data(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_data_file):
            raise FileNotFoundError(f"Data file not found at {self.raw_data_file}")
        df = pd.read_json(self.raw_data_file, lines=True)
        return df

    @staticmethod
    def save_data(df: pd.DataFrame, filepath: str):
        df.to_pickle(filepath)

