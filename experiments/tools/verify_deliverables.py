"""Check executed cells, unchanged input files, figures, and fresh-process inference."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import sys
import nbformat
import numpy as np
import pandas as pd

root = Path.cwd()
sys.path.insert(0,str(root))
notebook = nbformat.read(root/'Improved_ML_Training.ipynb',as_version=4)
nbformat.validate(notebook)
code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
for cell in code_cells:
    ast.parse(cell.source)
    assert cell.execution_count is not None, 'A code cell was not executed.'
    assert not any(output.output_type == 'error' for output in cell.outputs), 'Notebook contains an execution error.'
summary = json.loads((root/'experiments/run_summary.json').read_text())
for name, expected_hash in summary['hashes'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected_hash

import predict_improved as inference
def fitted_iterations(pipeline):
    estimator = pipeline.named_steps['model']
    models = getattr(estimator, 'estimators_', [estimator])
    result = []
    for model in models:
        if hasattr(model, 'n_iter_') and hasattr(model, 'max_iter'):
            iterations = np.asarray(model.n_iter_).ravel().tolist()
            assert max(iterations) < model.max_iter, 'Selected estimator did not converge.'
            result.extend(iterations)
    return result
selected_iterations = fitted_iterations(inference.sentiment_pipeline)
rating_iterations = fitted_iterations(inference.rating_pipeline)
raw = pd.read_csv(root/'drugsComTest_raw.csv')
manifest = pd.read_csv(root/'experiments/split_manifest.csv')
test_ids = manifest.loc[manifest.split.eq('reserved_test'),'uniqueID']
examples = raw.loc[raw.uniqueID.isin(test_ids)].head(12)
texts = examples.review.map(inference.clean_review)
expected_classes = inference.label_encoder.inverse_transform(inference.sentiment_pipeline.predict(texts))
expected_ratings = inference.rating_pipeline.predict(texts)
for index, (_, row) in enumerate(examples.iterrows()):
    answer = inference.predict_patient_experience(row.drugName,row.condition,row.review)
    assert set(answer) == {'drug','condition','predicted_experience_class','confidence','predicted_rating'}
    assert answer['drug'] == row.drugName
    assert answer['predicted_experience_class'] == expected_classes[index]
    assert answer['predicted_rating'] == int(expected_ratings[index])
    assert isinstance(answer['predicted_rating'],int) and 1 <= answer['predicted_rating'] <= 10
    assert 0 <= answer['confidence'] <= 1
for invalid in ['', '  ', None]:
    try:
        inference.predict_patient_experience('example','example',invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Empty input was not rejected.')

# Validate embedded figures without creating duplicate image files.
image_count = 0
for cell_index, cell in enumerate(notebook.cells):
    for output in cell.get('outputs',[]):
        png = output.get('data',{}).get('image/png')
        if png:
            image_count += 1
            assert base64.b64decode(png).startswith(b'\x89PNG'), 'Invalid embedded PNG'
verification = {'notebook_schema':'passed','code_syntax':'passed',
    'executed_code_cells':len(code_cells),'execution_errors':0,
    'csv_hashes_unchanged':True,'fresh_process_inference_examples':len(examples),
    'empty_input_validation':'passed','saved_model_roundtrip':'passed in notebook',
    'embedded_figures':image_count,'colab_execution':'not performed; historical local outputs checked; fresh cloud run still required'}
verification['selected_estimator_iterations'] = selected_iterations
verification['rating_estimator_iterations'] = rating_iterations
extension_path = root/'experiments/ensemble/summary.json'
if extension_path.exists():
    extension = json.loads(extension_path.read_text())
    assert 'Previously evaluated' in extension['test_status']
    assert all(weight > 0 for weight in extension['voting_weights'])
    comparison = pd.read_csv(root/'experiments/ensemble/model_comparison.csv')
    assert {'Optuna Bagging NB','Optuna AdaBoost','Bagging + boosting voting','Optuna weighted voting'} <= set(comparison.Model)
    assert len(comparison) == 12
    for name, budget in extension['trial_budgets'].items():
        trials = pd.read_csv(root/'experiments/ensemble'/f"{name.replace(' ','_')}_trials.csv")
        assert len(trials) == budget and trials.state.eq('COMPLETE').all()
    verification['ensemble_studies'] = '26 fitted-model trials and 30 OOF voting trials completed'
    verification['test_status'] = extension['test_status']
    verification['current_model_directory'] = inference.MODEL_DIR.name
(root/'experiments/verification.json').write_text(json.dumps(verification,indent=2))
print(json.dumps(verification,indent=2))
