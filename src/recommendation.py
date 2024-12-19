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
        """Load interaction data from a pickle file."""
        self.interaction_df = pd.read_pickle(self.user_interactions_file)

    def recommend_items(self, user_id: str, occasion: str, category: str, top_n: int = Config.NUM_RECOMMENDATIONS) -> pd.DataFrame:
        """Generate item recommendations for a specific user, occasion, and category."""
        if self.interaction_df is None:
            raise ValueError("Interactions not loaded. Call load_interactions() first.")

        # Filter items based on occasion and category
        filtered_items = self.interaction_df[
            (self.interaction_df["rented for"] == occasion) & 
            (self.interaction_df["category"] == category)
        ]

        if filtered_items.empty:
            return pd.DataFrame()

        # Define features for similarity calculation
        potential_features = [
            "fit_encoded", "bust_cup_encoded", "bust_band", "BMI", "size_encoded", "days_since_review",
            "review_length", "positive_word_count", "negative_word_count"
        ]

        for col in ["body type_encoded", "rented for_encoded", "category_encoded"]:
            if col in self.interaction_df.columns:
                potential_features.append(col)

        embedding_features = [col for col in self.interaction_df.columns if col.startswith("text_emb_")]
        potential_features.extend(embedding_features)

        similarity_features = [f for f in potential_features if f in self.interaction_df.columns]

        # Debugging: Check NaN in similarity features
        print("Debugging Similarity Features:")
        print("Selected Features:", similarity_features)
        print("Columns with NaN in Interaction Data (Filtered Items):")
        print(filtered_items[similarity_features].isna().sum())

        # Generate user vector
        user_data = self.interaction_df[self.interaction_df["user_id"] == float(user_id)][similarity_features]
        print("User Data Sample:")
        print(user_data.head())

        if user_data.empty:
            print(f"User ID {user_id} has no relevant data.")
            return pd.DataFrame()

        user_vector = user_data.mean().values.reshape(1, -1)

        # Debugging: Check NaN in user vector
        print("User Vector NaN Check:")
        print(pd.DataFrame(user_vector).isna().sum())

        # Handle cases where the user vector contains NaN values
        if np.isnan(user_vector).any():
            print("User vector contains NaN values. Filling NaN with 0.")
            user_vector = np.nan_to_num(user_vector)

        # Prepare item matrix
        item_matrix = filtered_items[similarity_features].values

        # Debugging: Check NaN in item matrix
        print("Item Matrix NaN Check:")
        print(pd.DataFrame(item_matrix).isna().sum())

        # Handle cases where the item matrix contains NaN values
        if np.isnan(item_matrix).any():
            print("Item matrix contains NaN values. Filling NaN with 0.")
            item_matrix = np.nan_to_num(item_matrix)

        # Calculate similarity scores
        similarity_scores = cosine_similarity(user_vector, item_matrix).flatten()

        # Attach similarity scores to filtered items
        filtered_items = filtered_items.copy()
        filtered_items["similarity_score"] = similarity_scores

        # Aggregate and sort recommendations
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
        recommendations['item_id'] = recommendations['item_id'].astype('int')
        return recommendations

    def save_recommendations(self, df: pd.DataFrame):
        """Save recommendations to a pickle file."""
        df.to_pickle(self.recommendation_file)

