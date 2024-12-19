from src.config import Config
from src.utils import LoggerFactory
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.recommendation import Recommender
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    logger = LoggerFactory.create_logger()

    # Step 1: Data Ingestion
    logger.info("Loading raw data...")
    ingestion = DataIngestion()
    df = ingestion.load_raw_data()

    # Step 2: Data Processing
    processor = DataProcessor()
    logger.info("Cleaning and preprocessing data...")
    df = processor.clean_data(df)
    df = processor.feature_engineering(df)

    logger.info("Saving cleaned data...")
    processor.save_preprocessed_data(df)

    # Scale numeric features
    numeric_cols_to_scale = [col for col in df.columns if col in [
        "BMI", "review_length", "days_since_review", "positive_word_count", "negative_word_count"
    ]]
    df = processor.scale_features(df, numeric_cols_to_scale)

    # Generate text embeddings for full_review if available
    if "full_review" in df.columns:
        logger.info("Generating text embeddings using SentenceTransformer...")
        model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        embeddings = model.encode(df["full_review"].values, show_progress_bar=True)
        emb_dim = embeddings.shape[1]
        for i in range(emb_dim):
            df[f"text_emb_{i}"] = embeddings[:, i]
        logger.info(f"Text embeddings added: text_emb_0 to text_emb_{emb_dim-1}")

    # df now represents user-item interactions after full processing
    processor.save_user_interactions(df)

    # Step 3: Generate recommendations for all user-category-occasion combos
    recommender = Recommender()
    logger.info("Loading user interactions for recommendations...")
    recommender.load_interactions()

    interaction_df = recommender.interaction_df
    if interaction_df is None or interaction_df.empty:
        logger.error("No interaction data available for recommendation.")
        return

    users = interaction_df["user_id"].unique()
    categories = interaction_df["category"].unique()
    occasions = interaction_df["rented for"].unique()

    logger.info("Generating top 3 recommendations for each user-category-occasion combination...")

    results = []
    for user_id in ["9","25"]:
        for category in categories:
            for occasion in occasions:
                recommendations = recommender.recommend_items(user_id, occasion, category, top_n=3)
                if recommendations.empty:
                    continue
                else:
                    for _, row in recommendations.iterrows():
                        results.append({
                            "user_id": user_id,
                            "category": category,
                            "occasion": occasion,
                            "item_id": row["item_id"],
                            "average_rating": row["average_rating"],
                            "review_count": row["review_count"]
                        })

    if not results:
        logger.warning("No recommendations found for any combination.")
        return

    results_df = pd.DataFrame(results)
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    output_file = os.path.join(Config.PROCESSED_DATA_DIR, "user_recommendation.pkl")
    results_df.to_pickle(output_file)

    logger.info(f"Recommendations saved successfully to {output_file}!")

if __name__ == "__main__":
    main()

