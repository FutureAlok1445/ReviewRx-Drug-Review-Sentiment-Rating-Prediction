"""Exercise real saved artifacts and boundary validation, not mock predictions."""
from fastapi.testclient import TestClient
from webapp.backend.main import app, clean_review


def test_real_model_contract_and_validation():
    with TestClient(app) as client:
        assert client.get('/api/health').json()['status'] == 'ready'
        examples = client.get('/api/examples').json()
        # Genuine examples are generated locally from a separately obtained dataset.
        # A fresh clone can still verify the model contract without patient-review data.
        assert isinstance(examples, list)
        if examples:
            assert len(examples) >= 3
        example = examples[0] if examples else {
            'drug': '', 'condition': '',
            'review': 'This is a synthetic API smoke test, not a dataset example.'}
        payload = {key: example[key] for key in ['drug', 'condition', 'review']}
        response = client.post('/api/predict', json=payload)
        assert response.status_code == 200
        prediction = response.json()
        assert set(prediction) == {'drug', 'condition', 'predicted_experience_class', 'confidence', 'predicted_rating'}
        assert prediction['drug'] == payload['drug']
        assert prediction['condition'] == payload['condition']
        assert 1 <= prediction['predicted_rating'] <= 10
        assert 0 <= prediction['confidence'] <= 1
        texts = [clean_review(payload['review'])]
        expected = app.state.encoder.inverse_transform(app.state.sentiment.predict(texts))[0]
        assert prediction['predicted_experience_class'] == expected
        assert prediction['predicted_rating'] == int(app.state.rating.predict(texts)[0])
        for invalid in ['', '  ', None, 12, 'x' * 20001]:
            assert client.post('/api/predict', json={**payload, 'review': invalid}).status_code == 422
        assert client.post('/api/predict', json={**payload, 'unknown': 'field'}).status_code == 422
        results = client.get('/api/results')
        assert results.status_code == 200
        assert len(results.json()['sentiment_models']) >= 6
        assert results.json()['sentiment_selection']
        notebook = client.get('/api/notebook')
        assert notebook.status_code == 200
        assert 'attachment' in notebook.headers['content-disposition']
        assert notebook.json()['nbformat'] == 4
        frontend = client.get('/')
        assert frontend.status_code == 200
        assert '<div id="root">' in frontend.text
