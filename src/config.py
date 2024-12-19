import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, "preprocessed")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

    RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "renttherunway_final_data.json.gz")
    CLEANED_DATA_FILE = os.path.join(PREPROCESSED_DATA_DIR, "cleaned_data.pkl")
    USER_INTERACTIONS_FILE = os.path.join(PREPROCESSED_DATA_DIR, "user_item_interactions.pkl")
    RECOMMENDATION_FILE = os.path.join(PROCESSED_DATA_DIR, "user_recommendation.pkl")

    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    NUM_RECOMMENDATIONS = 5

