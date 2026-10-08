"""Serve the trained review pipelines; training never runs in an HTTP request."""
from contextlib import asynccontextmanager
from pathlib import Path
import html
import json
import os
import re

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def clean_review(text):
    text = html.unescape(text).replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'^"|"$', '', text.strip())
    return re.sub(r'\s+', ' ', text).strip()


def model_directory():
    configured = os.getenv('DRUG_MODEL_DIR')
    if configured:
        return Path(configured).expanduser().resolve()
    current = PROJECT_ROOT / 'models_ensemble'
    return current


@asynccontextmanager
async def lifespan(app):
    directory = model_directory()
    # Load trusted local artifacts once. Missing artifacts fail startup explicitly.
    app.state.sentiment = joblib.load(directory / 'sentiment_pipeline.joblib')
    app.state.rating = joblib.load(directory / 'rating_pipeline.joblib')
    app.state.encoder = joblib.load(directory / 'sentiment_label_encoder.joblib')
    app.state.metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    app.state.model_directory = directory.name
    yield
    app.state.sentiment = None
    app.state.rating = None


app = FastAPI(title='Drug Review Analysis', version='1.0.0', lifespan=lifespan,
              description='Patient-reported experience from review text. Not medical advice.')
app.add_middleware(CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    drug: str = Field(default='', max_length=200)
    condition: str = Field(default='', max_length=200)
    review: str = Field(min_length=1, max_length=20000)

    @field_validator('review')
    @classmethod
    def review_must_have_text(cls, value):
        if not clean_review(value):
            raise ValueError('Enter a non-empty review.')
        return value


class ReviewPrediction(BaseModel):
    drug: str
    condition: str
    predicted_experience_class: str
    confidence: float = Field(ge=0, le=1)
    predicted_rating: int = Field(ge=1, le=10)


@app.get('/api/health')
def health(request: Request):
    return {'status': 'ready', 'model_directory': request.app.state.model_directory}


@app.post('/api/predict', response_model=ReviewPrediction)
def predict(payload: ReviewInput, request: Request):
    texts = [clean_review(payload.review)]
    sentiment = request.app.state.sentiment
    encoded = sentiment.predict(texts)[0]
    probabilities = sentiment.predict_proba(texts)[0]
    return {'drug': payload.drug, 'condition': payload.condition,
            'predicted_experience_class': str(request.app.state.encoder.inverse_transform([encoded])[0]),
            'confidence': round(float(probabilities.max()), 3),
            'predicted_rating': int(request.app.state.rating.predict(texts)[0])}


def table_records(path):
    if not path.exists():
        return []
    frame = pd.read_csv(path).astype(object)
    return frame.where(pd.notna(frame), None).to_dict('records')


@app.get('/api/results')
def results(request: Request):
    experiments = PROJECT_ROOT / 'experiments'
    extension = experiments / 'ensemble'
    has_extension = (extension / 'summary.json').exists()
    metadata = request.app.state.metadata
    sentiment_path = extension / 'model_comparison.csv' if has_extension else experiments / 'model_comparison.csv'
    rating_path = extension / 'rating_comparison.csv' if has_extension else experiments / 'rating_comparison.csv'
    tests = metadata.get('repeat_test_results') or metadata.get('test_results', {})
    return {'sentiment_selection': metadata.get('sentiment_selection'),
            'rating_selection': metadata.get('rating_selection'),
            'sentiment_models': table_records(sentiment_path),
            'rating_models': table_records(rating_path), 'test_metrics': tests,
            'test_note': metadata.get('test_status') or metadata.get('test_note'),
            'confidence_note': metadata.get('confidence_note'),
            'extension_complete': has_extension}


@app.get('/api/examples')
def examples():
    path = PROJECT_ROOT / 'webapp' / 'backend' / 'examples.json'
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding='utf-8'))


@app.get('/api/notebook')
def training_notebook():
    path = PROJECT_ROOT / 'Improved_ML_Training.ipynb'
    if not path.exists():
        raise HTTPException(status_code=404, detail='Training notebook is not available.')
    return FileResponse(path, filename=path.name, media_type='application/x-ipynb+json')


# npm run build creates this folder. API-only development works without it.
distribution = PROJECT_ROOT / 'webapp' / 'frontend' / 'dist'
if distribution.exists():
    app.mount('/', StaticFiles(directory=distribution, html=True), name='frontend')
