import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import pickle

# --- Data Preparation ---
def load_and_prepare_data(file_path="C:/Users/RMK KOUSHIK/Downloads/intern/Video_Games.jsonl", sample_size=None):
    df = pd.read_json(file_path, lines=True)
    
    # Optional: Sample for faster processing (e.g., 10,000 rows)
    if sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # Feature engineering
    df['review_length'] = df['text'].apply(lambda x: len(str(x).split()))
    df['title_length'] = df['title'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce') / 1000
    user_counts = df.groupby('user_id').size().reset_index(name='user_review_count')
    df = df.merge(user_counts, on='user_id')
    df = df.sort_values(['user_id', 'timestamp'])
    df['time_delta'] = df.groupby('user_id')['timestamp'].diff().fillna(3600)
    item_avg_rating = df.groupby('asin')['rating'].mean().reset_index(name='item_avg_rating')
    df = df.merge(item_avg_rating, on='asin')
    df['rating_deviation'] = df['rating'] - df['item_avg_rating']
    user_days = df.groupby('user_id')['timestamp'].agg(['min', 'max']).reset_index()
    user_days['active_days'] = (user_days['max'] - user_days['min']) / 86400 + 1
    user_days = user_days.merge(user_counts, on='user_id')
    user_days['reviews_per_day'] = user_days['user_review_count'] / user_days['active_days']
    df = df.merge(user_days[['user_id', 'reviews_per_day']], on='user_id')
    df['image_count'] = df['images'].apply(len)
    df['verified_purchase'] = df['verified_purchase'].astype(int)

    feature_cols = ['user_review_count', 'time_delta', 'rating_deviation', 'reviews_per_day', 
                    'review_length', 'title_length', 'image_count', 'helpful_vote', 'verified_purchase']
    X_metadata = df[feature_cols].values
    X = np.hstack([embeddings, X_metadata])
    return df, X

# --- Train and Save Model ---
def train_and_save_model(file_path="C:/Users/RMK KOUSHIK/Downloads/intern/Video_Games.jsonl", sample_size=10000):
    df, X = load_and_prepare_data(file_path, sample_size)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iso_forest.fit(X_scaled)

    # Save model and scaler
    with open("isolation_forest_model.pkl", "wb") as f:
        pickle.dump(iso_forest, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("dataset_sample.pkl", "wb") as f:
        pickle.dump(df, f)  # Save df for metadata lookups
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_and_save_model(file_path="C:/Users/RMK KOUSHIK/Downloads/intern/Video_Games.jsonl", sample_size=10000)  # Adjust sample_size or remove