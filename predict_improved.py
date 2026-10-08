"""Load the separate improved artifacts while keeping the original prediction contract."""
from pathlib import Path
import html
import re
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
ENSEMBLE_DIR = PROJECT_ROOT / 'models_ensemble'
MODEL_DIR = ENSEMBLE_DIR
sentiment_pipeline = joblib.load(MODEL_DIR / 'sentiment_pipeline.joblib')
rating_pipeline = joblib.load(MODEL_DIR / 'rating_pipeline.joblib')
label_encoder = joblib.load(MODEL_DIR / 'sentiment_label_encoder.joblib')


def clean_review(text):
    text = html.unescape(str(text)).replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'^"|"$', '', text.strip())
    return re.sub(r'\s+', ' ', text).strip()


def predict_patient_experience(drug: str, condition: str, review: str):
    """Drug/condition are returned as context; text alone determines predictions."""
    if not isinstance(review, str) or not clean_review(review):
        raise ValueError('Please supply a non-empty review string.')
    text = [clean_review(review)]
    encoded = sentiment_pipeline.predict(text)[0]
    probabilities = sentiment_pipeline.predict_proba(text)[0]
    return {
        'drug': drug,
        'condition': condition,
        'predicted_experience_class': str(label_encoder.inverse_transform([encoded])[0]),
        'confidence': round(float(probabilities.max()), 3),
        'predicted_rating': int(rating_pipeline.predict(text)[0]),
    }
