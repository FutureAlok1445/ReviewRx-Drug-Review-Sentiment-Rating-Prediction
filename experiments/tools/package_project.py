"""Package the current notebook and runnable app without obsolete experiments."""
from pathlib import Path
import zipfile
root = Path.cwd()
paths = [root/name for name in [
    'README.md','Improved_ML_Training.ipynb','MODEL_ANALYSIS.md','MODEL_COMPARISON.md',
    'requirements-training.txt','predict_improved.py','prepare_demo_examples.py','DATASET.md','.gitignore']]
for folder in ['webapp','models_ensemble','docs']:
    for path in (root/folder).rglob('*'):
        if path.is_file() and path.name not in {'examples.json','app-preview.jpg'} and not {'node_modules','__pycache__','.pytest_cache'}.intersection(path.parts):
            paths.append(path)
for name in ['run_summary.json','model_comparison.csv','rating_comparison.csv',
             'original_reproduction.json','data_audit.csv','leakage_audit.json','verification.json',
             'final_test_results.json','sentiment_search.csv','rating_search.csv']:
    paths.append(root/'experiments'/name)
paths += [path for path in (root/'experiments/ensemble').iterdir()
          if path.is_file() and path.suffix in ['.csv','.json']]
paths += [root/'experiments/tools'/name for name in ['package_project.py','verify_deliverables.py']]
archive = root/'Drug_Review_Complete_Project.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as output:
    for path in sorted(set(paths)):
        assert path.is_file(), str(path)
        output.write(path,arcname=str(Path('Drug Review Analysis')/path.relative_to(root)))
with zipfile.ZipFile(archive) as output:
    assert output.testzip() is None
    names=output.namelist()
    assert sum(name.endswith('.ipynb') for name in names)==1
    assert not any('node_modules/' in name or '/models_improved/' in name or '/original/' in name for name in names)
print('Verified current-project archive:',archive.name,'| files:',len(set(paths)),
      '| MB:',round(archive.stat().st_size/1e6,1))
