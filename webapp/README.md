# Drug Review Analysis web application

React frontend + FastAPI backend. The app uses the real trained review pipelines,
preserving `drug`, `condition`, `review` inputs and the five existing prediction fields.
Training stays in `Improved_ML_Training.ipynb`. Drug and condition are display context;
only review text is used for prediction. No review submission is written to disk.

## Quick start: one server

From the project root, install backend dependencies:

```powershell
python -m pip install -r webapp/backend/requirements.txt
```

Build the frontend:

```powershell
cd webapp/frontend
npm ci
npm run build
```

Return to the project root and start FastAPI:

```powershell
python -m uvicorn webapp.backend.main:app --host 127.0.0.1 --port 8000
```

Open [the application](http://127.0.0.1:8000).
[Interactive API documentation](http://127.0.0.1:8000/docs) is also available.
Python 3.14.2 was used for the recorded local checks.

## Development: two terminals

Start FastAPI from the project root with the command above. In a second terminal:

```powershell
cd webapp/frontend
npm ci
npm run dev
```

Open [the React development app](http://127.0.0.1:5173). Vite forwards `/api`
requests to FastAPI. Restart FastAPI after a frontend build if it was started before
`dist` existed. The app uses in-page navigation, so no server route rewrite is needed.

## Models and data

The backend loads the current `models_ensemble` artifacts. Set `DRUG_MODEL_DIR` to a trusted artifact directory to override.
Restart the backend after new training. Models load once during startup; missing files
fail startup explicitly. The trained scikit-learn version must match the requirements.
Metadata, comparison CSVs and genuine validation examples drive the results UI.
The raw training CSV is not needed for HTTP prediction.

- `POST /api/predict`: accepts `{ "drug": "...", "condition": "...", "review": "..." }`.
- `GET /api/health`: readiness and model-directory name.
- `GET /api/results`: recorded sentiment/rating comparisons and test-status notes.
- `GET /api/examples`: locally generated genuine review examples (empty until prepared).
- `GET /api/notebook`: download the complete project notebook; also linked in the app.

Empty, non-string or longer-than-20,000-character reviews return HTTP 422.
The confidence value is the maximum model probability, not a calibrated correctness
guarantee. The sentiment bins and rating model are separate, so their outputs may differ.
The test dataset was already viewed before ensemble extension; repeated metrics are
labeled accordingly. This student research app does not recommend medication.

## Checks

From the project root:

```powershell
python -m pip install -r webapp/backend/requirements-dev.txt
python -m pytest webapp/backend/test_api.py -q
```

From `webapp/frontend`, run `npm run build`. Backend tests use real saved artifacts,
genuine dataset inputs and invalid-input boundaries. No deployment or Git operations
are performed. The local application has no accounts, database or persisted history;
these were not requested for the initial review-analysis workflow.

Framework references: [FastAPI lifespan model loading](https://fastapi.tiangolo.com/advanced/events/)
and [Vite development proxy](https://vite.dev/config/server-options).

## Optional genuine examples

Obtain the dataset directly from the source linked in the root README, then run `python prepare_demo_examples.py` from the project root. It generates eight local examples; the patient-review file is ignored by Git. The application can predict manually entered text without the dataset.
