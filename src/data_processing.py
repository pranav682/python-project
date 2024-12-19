import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from .config import Config
from .utils import TextCleaner

class DataProcessor:
    def __init__(self, cleaned_data_file: str = Config.CLEANED_DATA_FILE, user_interactions_file: str = Config.USER_INTERACTIONS_FILE):
        self.cleaned_data_file = cleaned_data_file
        self.user_interactions_file = user_interactions_file

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop columns with >50% missing data
        df.dropna(thresh=len(df)*0.5, axis=1, inplace=True)

        # Categorical imputation
        cat_cols = df.select_dtypes(include='object').columns
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

        # Numeric imputation
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            df[num_cols] = num_imputer.fit_transform(df[num_cols])

        # Text cleaning
        if "review_text" in df.columns:
            df["review_text"] = df["review_text"].fillna("").apply(TextCleaner.clean_text)

        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create full_review
        if "review_text" in df.columns and "review_summary" in df.columns:
            df["full_review"] = (df["review_summary"].fillna("") + " " + df["review_text"].fillna("")).str.strip()

        # Encode fit
        if "fit" in df.columns:
            fit_mapping = {"small": 0, "fit": 1, "large": 2}
            df["fit_encoded"] = df["fit"].map(fit_mapping)


        # Parse weight
        if "weight" in df.columns:
            df["weight"] = df["weight"].astype(str).str.replace("lbs", "", regex=False).str.strip()
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

        # Parse height
        if "height" in df.columns:
            def convert_height(h):
                if pd.isna(h):
                    return np.nan
                parts = h.split("'")
                if len(parts) == 2:
                    feet = parts[0]
                    inches = parts[1].replace('"', '').strip()
                    if feet.isdigit() and inches.isdigit():
                        return int(feet)*12 + int(inches)
                return np.nan
            df["height"] = df["height"].apply(convert_height)

        # Compute BMI
        if "height" in df.columns and "weight" in df.columns:
            df["BMI"] = np.where((df["height"]>0) & (df["weight"]>0), (df["weight"] / (df["height"]**2))*703, np.nan)

        # Parse bust size
        if "bust size" in df.columns:
            df["bust_band"] = pd.to_numeric(df["bust size"].str.extract(r"(\d+)", expand=False), errors='coerce')
            df["bust_cup"] = df["bust size"].str.extract(r"([A-Za-z]+)[^A-Za-z]*$", expand=False).str.lower()

            cup_map = {"aa":0.5,"a":1,"b":2,"c":3,"d":4,"dd":5,"ddd":6,"f":7}
            df["bust_cup_encoded"] = df["bust_cup"].map(cup_map)
      
        # Fill missing values for encoded columns
        if "bust_cup_encoded" in df.columns:
            df["bust_cup_encoded"] = df["bust_cup_encoded"].fillna(0)  # Default to 0 if no cup size
  
        # Label encode categorical features if present
        categorical_features = []
        if "body type" in df.columns:
            categorical_features.append("body type")
        if "rented for" in df.columns:
            categorical_features.append("rented for")
        if "category" in df.columns:
            categorical_features.append("category")

        for feature in categorical_features:
            le = LabelEncoder()
            df[f"{feature}_encoded"] = le.fit_transform(df[feature].astype(str))

        return df

    def generate_text_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        if "full_review" in df.columns:
            print("Generating text embeddings...")
            embeddings = self.embedding_model.encode(df["full_review"].fillna("").values, show_progress_bar=True)
            emb_dim = embeddings.shape[1]
            for i in range(emb_dim):
                df[f"text_emb_{i}"] = embeddings[:, i]
            print(f"Text embeddings added as columns: text_emb_0 to text_emb_{emb_dim-1}")
        return df

    def scale_features(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        if len(cols) > 0:
            scaler = MinMaxScaler()
            df[cols] = scaler.fit_transform(df[cols])
        return df

    def save_preprocessed_data(self, df: pd.DataFrame):
        df.to_pickle(self.cleaned_data_file)

    def save_user_interactions(self, df: pd.DataFrame):
        df.to_pickle(self.user_interactions_file)

