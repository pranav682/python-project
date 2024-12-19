import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .config import Config

class Recommender:
    def __init__(self, user_interactions_file: str = Config.USER_INTERACTIONS_FILE, recommendation_file: str = Config.RECOMMENDATION_FILE):
        self.user_interactions_file = user_interactions_file
        self.recommendation_file = recommendation_file
        self.interaction_df = None

    def load_interactions(self):
        self.interaction_df = pd.read_pickle(self.user_interactions_file)

    def recommend_items(self, user_id: str, occasion: str, category: str, top_n: int = Config.NUM_RECOMMENDATIONS) -> pd.DataFrame:
        if self.interaction_df is None:
            raise ValueError("Interactions not loaded. Call load_interactions() first.")

        filtered_items = self.interaction_df[
            (self.interaction_df["rented for"] == occasion) & 
            (self.interaction_df["category"] == category)
        ]

        if filtered_items.empty:
            return pd.DataFrame()

        potential_features = [
            "fit_encoded","bust_cup_encoded","bust_band","BMI","size_encoded","days_since_review",
            "review_length","positive_word_count","negative_word_count"
        ]

        for col in ["body type_encoded", "rented for_encoded", "category_encoded"]:
            if col in self.interaction_df.columns:
                potential_features.append(col)

        embedding_features = [col for col in self.interaction_df.columns if col.startswith("text_emb_")]
        potential_features.extend(embedding_features)

        similarity_features = [f for f in potential_features if f in self.interaction_df.columns]

        
        #print("Features used for similarity:")
        #print(similarity_features)
        #print("Columns with NaN in selected features:")
        #print(filtered_items[similarity_features].isna().sum())


        user_vector = self.interaction_df[self.interaction_df["user_id"] == user_id][similarity_features].mean().values.reshape(1, -1)
        if user_vector.size == 0:
            return pd.DataFrame()
 
        print("User Vector NaN Check:")
        print(user_vector)
        print(pd.DataFrame(user_vector).isna().sum())


        item_matrix = filtered_items[similarity_features].values

        print("Item Matrix NaN Check:")
        print(pd.DataFrame(item_matrix).isna().sum())


        similarity_scores = cosine_similarity(user_vector, item_matrix).flatten()
        filtered_items = filtered_items.copy()
        filtered_items["similarity_score"] = similarity_scores

        recommendations = (
            filtered_items.groupby("item_id")
            .agg(
                average_rating=("rating", "mean"),
                review_count=("user_id", "count"),
                similarity_score=("similarity_score", "max"),
                category=("category", "first"),
                rented_for=("rented for", "first")
            )
            .reset_index()
            .sort_values(by="similarity_score", ascending=False)
            .head(top_n)
        )
        return recommendations

    def save_recommendations(self, df: pd.DataFrame):
        df.to_pickle(self.recommendation_file)

