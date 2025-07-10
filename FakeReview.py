import streamlit as st
import numpy as np
import joblib
import requests
from bs4 import BeautifulSoup
import re
from sentence_transformers import SentenceTransformer

# Load pre-trained model and scaler
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to scrape Amazon reviews
def scrape_amazon_review(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None, f"Failed to fetch URL. Status code: {response.status_code}"
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to extract review details
        review_data = {}
        
        # Extract title
        title_elem = soup.select_one('a.review-title') or soup.select_one('.review-title') or soup.select_one('[data-hook="review-title"]')
        if title_elem:
            review_data['title'] = title_elem.get_text().strip()
        else:
            review_data['title'] = ""
        
        # Extract text
        text_elem = soup.select_one('.review-text') or soup.select_one('[data-hook="review-body"]')
        if text_elem:
            review_data['text'] = text_elem.get_text().strip()
        else:
            return None, "Could not find review text on the page"
        
        # Extract rating
        rating_elem = soup.select_one('i.review-rating') or soup.select_one('[data-hook="review-star-rating"]')
        if rating_elem:
            rating_text = rating_elem.get_text().strip()
            rating_match = re.search(r'(\d+\.?\d*)\s*out\s*of\s*\d+\s*stars', rating_text)
            if rating_match:
                review_data['rating'] = float(rating_match.group(1))
            else:
                rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                if rating_match:
                    review_data['rating'] = float(rating_match.group(1))
                else:
                    review_data['rating'] = 3.0  # Default value
        else:
            review_data['rating'] = 3.0  # Default value
        
        # Try to find verified purchase
        verified_elem = soup.find(string=re.compile("Verified Purchase", re.IGNORECASE))
        review_data['verified_purchase'] = 1 if verified_elem else 0
        
        # Try to find helpful votes
        helpful_elem = soup.select_one('[data-hook="helpful-vote-statement"]')
        if helpful_elem:
            helpful_text = helpful_elem.get_text().strip()
            helpful_match = re.search(r'(\d+)', helpful_text)
            if helpful_match:
                review_data['helpful_vote'] = int(helpful_match.group(1))
            else:
                review_data['helpful_vote'] = 0
        else:
            review_data['helpful_vote'] = 0
        
        # Default values for other fields
        review_data['user_review_count'] = 1
        review_data['time_delta'] = 3600
        review_data['reviews_per_day'] = 1.0
        review_data['image_count'] = 0
        
        return review_data, None
        
    except Exception as e:
        return None, f"Error scraping review: {str(e)}"

# Predict anomaly score
def predict_review(model, scaler, embedder, review_data):
    # Extract review data
    rating = review_data.get('rating', 3.0)
    title = review_data.get('title', '')
    text = review_data.get('text', '')
    
    # Text embedding
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
    
    # Combine all features
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
        ## Check if a product review is potentially fake
        This app uses machine learning to analyze review text and metadata to detect potentially 
        fake or suspicious product reviews.
    """)

    # Sidebar for model paths
    with st.sidebar:
        st.header("Model Configuration")
        model_path = st.text_input(
            "Model Path", 
            "C:/Users/RMK KOUSHIK/Downloads/Intern/isolation_forest_model.pkl"
        )
        scaler_path = st.text_input(
            "Scaler Path", 
            "C:/Users/RMK KOUSHIK/Downloads/Intern/scaler.pkl"
        )
        st.markdown("---")
        st.markdown("### How it works")
        st.markdown("""
        This tool analyzes:
        - Review text semantics
        - User behavior patterns
        - Rating distributions
        - Text characteristics
        
        Higher anomaly scores indicate potentially fake reviews.
        """)

    try:
        # Load model, scaler and embedder
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        embedder = load_embedding_model()
        
        # Create tabs for URL input vs Manual input
        tab1, tab2 = st.tabs(["Scrape from URL", "Manual Input"])
        
        # Tab 1: URL Input
        with tab1:
            st.subheader("Scrape Review from URL")
            st.markdown("Enter an Amazon review URL to analyze:")
            url = st.text_input("Review URL", "https://www.amazon.com/product-reviews/...")
            
            if st.button("Scrape and Analyze", type="primary", key="scrape_btn"):
                if not url or not url.startswith("http"):
                    st.error("Please enter a valid URL")
                else:
                    with st.spinner("Scraping review from URL..."):
                        review_data, error = scrape_amazon_review(url)
                    
                    if error:
                        st.error(f"Error: {error}")
                    elif review_data:
                        st.success("Successfully scraped review!")
                        
                        # Display scraped data
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Scraped Review")
                            st.markdown(f"**Title:** {review_data['title']}")
                            st.markdown(f"**Rating:** {review_data['rating']} stars")
                            st.markdown(f"**Verified Purchase:** {'Yes' if review_data['verified_purchase'] == 1 else 'No'}")
                            st.markdown(f"**Helpful Votes:** {review_data['helpful_vote']}")
                            st.markdown("**Review Text:**")
                            st.markdown(f"_{review_data['text']}_")
                        
                        # Analyze the review
                        with st.spinner("Analyzing review..."):
                            score = predict_review(model, scaler, embedder, review_data)
                        
                        with col2:
                            st.subheader("Analysis Result")
                            st.write(f"Anomaly Score: **{score:.3f}**")
                            
                            # Progress bar for visualization
                            st.progress(min(score, 1.0))
                            
                            if score > 0.5:
                                st.error("⚠️ High probability of being a fake review!")
                            elif score > 0.3:
                                st.warning("⚠️ Suspicious review - interpret with caution")
                            elif score > 0.2:
                                st.info("ℹ️ Some unusual patterns detected")
                            else:
                                st.success("✅ Likely a genuine review")
        
        # Tab 2: Manual Input
        with tab2:
            st.subheader("Enter Review Details Manually")
            
            # Create two columns
            col1, col2 = st.columns([2, 1])
            
            # Input fields - left column
            with col1:
                rating = st.slider("Rating", 1.0, 5.0, 3.0, step=0.5)
                title = st.text_input("Review Title", "Great product")
                text = st.text_area("Review Text", "This product is amazing!", height=150)
                
                # Advanced options collapsible
                with st.expander("Advanced Options"):
                    user_review_count = st.number_input("User's Total Review Count", min_value=1, value=1)
                    time_delta = st.number_input("Time Since Last Review (seconds)", min_value=0, value=3600)
                    reviews_per_day = st.number_input("User's Reviews Per Day", min_value=0.01, value=1.0, step=0.1)
                    image_count = st.number_input("Number of Images", min_value=0, value=0)
                    helpful_vote = st.number_input("Helpful Votes", min_value=0, value=0)
                    verified_purchase = st.checkbox("Verified Purchase", value=True)
            
            # Right column
            with col2:
                st.subheader("Review Metrics")
                st.metric("Word Count", len(text.split()))
                st.metric("Title Word Count", len(title.split()) if title else 0)
                
                # Placeholder for result
                result_placeholder = st.empty()
                
            # Predict button
            if st.button("Check Review", type="primary", key="manual_btn"):
                if not text.strip():
                    st.error("Please enter review text!")
                else:
                    review_data = {
                        'rating': rating,
                        'title': title,
                        'text': text,
                        'user_review_count': user_review_count,
                        'time_delta': time_delta,
                        'rating_deviation': rating - 3.0,  # Using default average rating
                        'reviews_per_day': reviews_per_day,
                        'image_count': image_count,
                        'helpful_vote': helpful_vote,
                        'verified_purchase': 1 if verified_purchase else 0
                    }
                    
                    with st.spinner("Analyzing review..."):
                        score = predict_review(model, scaler, embedder, review_data)
                    
                    # Display result in the right column
                    with result_placeholder:
                        st.subheader("Analysis Result")
                        st.write(f"Anomaly Score: **{score:.3f}**")
                        
                        # Progress bar for visualization
                        st.progress(min(score, 1.0))
                        
                        if score > 0.5:
                            st.error("⚠️ High probability of being a fake review!")
                        elif score > 0.3:
                            st.warning("⚠️ Suspicious review - interpret with caution")
                        elif score > 0.2:
                            st.info("ℹ️ Some unusual patterns detected")
                        else:
                            st.success("✅ Likely a genuine review")
    
    except Exception as e:
        st.error(f"Error loading model or making prediction: {str(e)}")
        st.markdown("""
        **Troubleshooting:**
        - Check if the model and scaler file paths are correct
        - Ensure you have all required libraries installed
        - Make sure the model was trained with the same feature set being used for prediction
        """)

if __name__ == "__main__":
    main()