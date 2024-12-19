import pandas as pd
from src.config import Config

# Load interaction data
interaction_df = pd.read_pickle(Config.USER_INTERACTIONS_FILE)

# Check for NaN values
print(interaction_df.iloc[:,:-383].info())
print(interaction_df.iloc[:,:-383].head(10))
nan_summary = interaction_df.isna().sum()
#print columns
print(interaction_df.columns[:-383])
# Print summary of NaN values
print("Summary of NaN values in each column:")
print(nan_summary)

# Optional: Identify columns with any NaN values
nan_columns = nan_summary[nan_summary > 0].index.tolist()
if nan_columns:
    print("\nColumns with NaN values:")
    print(nan_columns)
else:
    print("\nNo NaN values detected.")

