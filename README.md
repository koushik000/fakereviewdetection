# Fake Review Detector

This repository contains two Streamlit web applications:

- **Fake Review Detector**: Detects potentially fake product reviews using machine learning and text embeddings.


## Features

### Fake Review Detector
- Analyze Amazon reviews by scraping from a URL or manual input.
- Uses a pre-trained Isolation Forest model and sentence embeddings.
- Visualizes anomaly scores and provides interpretability.
- Customizable model and scaler paths.



## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Model Files**:
    - Ensure `isolation_forest_model.pkl` and `scaler.pkl` are present in the project directory.
    - To retrain the model, run:
      ```sh
      python train_isolation_model.py
      ```

4. **Run the Apps**:
    - **Fake Review Detector**:
      ```sh
      streamlit run FakeReview.py
      ```
   

## File Structure

- `FakeReview.py` - Main app for fake review detection (with scraping/manual input).
- `fake_review_detector.py` - Minimal version for manual review input.
- `main.py` - ATS Resume Scoring app.
- `train_isolation_model.py` - Script to train and save the Isolation Forest model.
- `requirements.txt` - Python dependencies.
- `isolation_forest_model.pkl`, `scaler.pkl` - Model and scaler files.
- `dataset_sample.pkl` - Sample dataset for reference.

## Notes

- The Fake Review Detector uses the [Sentence Transformers](https://www.sbert.net/) library for text embeddings.
- For Amazon review scraping, only public reviews are supported and scraping may break if Amazon changes its HTML structure.


## License

This project is for educational
