# python-project
Recommendation system based on cosine similarity 
# Rent the Runway Recommendation System

## Overview
This setup provides a complete codebase for a Rent the Runway recommendation system, including:
- Data ingestion and preprocessing
- Feature engineering (fit encoding, BMI, bust size parsing, text cleaning)
- Label encoding categorical features
- Generating text embeddings with SentenceTransformer
- Producing top-3 recommendations for every user-category-occasion combo
- A Flask web UI to browse recommendations interactively

## Steps to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

Place the raw dataset in data/raw/renttherunway_final_data.json.gz.

Run the pipeline:
python main.py

This will:
Clean and preprocess data
Compute embeddings
Generate user_recommendation.csv in data/processed/

Run the Web UI:
python app.py

Visit http://localhost:5000 to select user_id, category, occasion, and view recommendations.


