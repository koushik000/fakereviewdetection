import streamlit as st
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# Load pre-trained model and scaler
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Predict anomaly score
def predict_review(model, scaler, embedder, review_data):
    # Extract review data
    rating = review_data.get('rating', 3.0)
    title = review_data.get('title', '')
    text = review_data.get('text', '')
    
    # Text embedding - this should be 384 dimensions for 'paraphrase-MiniLM-L6-v2'
    embedding = embedder.encode([text], show_progress_bar=False)[0]
    
    # Metadata features - make sure we have all 9 features as in training
    review_length = len(text.split())
    title_length = len(title.split()) if title else 0
    user_review_count = review_data.get('user_review_count', 1)
    time_delta = review_data.get('time_delta', 3600)
    rating_deviation = review_data.get('rating_deviation', rating - 3.0)
    reviews_per_day = review_data.get('reviews_per_day', 1.0)
    image_count = review_data.get('image_count', 0)
    helpful_vote = review_data.get('helpful_vote', 0)
    verified_purchase = review_data.get('verified_purchase', 1)
    
    # Combine all features - make sure this matches training dimensions
    metadata = [
        user_review_count, 
        time_delta, 
        rating_deviation, 
        reviews_per_day, 
        review_length, 
        title_length, 
        image_count, 
        helpful_vote, 
        verified_purchase
    ]
    
    features = np.hstack([embedding, metadata])
    features_scaled = scaler.transform([features])
    score = -model.decision_function(features_scaled)[0]
    return score

# Streamlit app
def main():
    st.title("Fake Review Detector")
    st.markdown("""
        Enter the review details below to check if it's likely fake or real.
    """)

    # Load model and scaler
    model, scaler = load_model_and_scaler("C:/Users/RMK KOUSHIK/Downloads/Intern/isolation_forest_model.pkl", "C:/Users/RMK KOUSHIK/Downloads/Intern/scaler.pkl")
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Input fields
    rating = st.slider("Rating", 1.0, 5.0, 3.0, step=0.5)
    title = st.text_input("Review Title", "Great product")
    text = st.text_area("Review Text", "This product is amazing!")

    # Predict button
    if st.button("Check Review"):
        if not text.strip():
            st.error("Please enter review text!")
        else:
            review_data = {
            'rating': rating,
            'title': title,
            'text': text,
            'user_review_count': 1,  # Default for new users
            'time_delta': 3600,  # Default time delta
            'rating_deviation': rating - 3.0,  # Default average rating
            'reviews_per_day': 1.0,  # Default reviews per day
            'image_count': 0,  # Default image count
            'helpful_vote': 0,  # Default helpful votes
            'verified_purchase': 1  # Default verified purchase
        }
            score = predict_review(model, scaler, embedder, review_data)
            
            st.subheader("Result")
            st.write(f"Anomaly Score: **{score:.3f}**")
            if score > 0.6:
                st.warning("High chance this is a fake review!")
            elif score > 0.4:
                st.info("Possibly suspicious—review with caution.")
            else:
                st.success("Looks like a genuine review.")

if __name__ == "__main__":
    main()