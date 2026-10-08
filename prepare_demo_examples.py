"""Prepare eight genuine local examples without redistributing patient reviews."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from predict_improved import clean_review, predict_patient_experience

root = Path(__file__).resolve().parent
dataset_path = root / 'drugsComTrain_raw.csv'
if not dataset_path.exists():
    raise SystemExit('Download the dataset yourself and place drugsComTrain_raw.csv in the project root. See README.md.')

reviews = pd.read_csv(dataset_path)
if reviews.columns[0].lower().startswith('unnamed'):
    reviews.rename(columns={reviews.columns[0]: 'uniqueID'}, inplace=True)
reviews['review_clean'] = reviews.review.map(clean_review)
reviews['text_group'] = reviews.review_clean.str.lower()
reviews = reviews.drop_duplicates(['text_group', 'rating']).reset_index(drop=True)
reviews['experience'] = np.where(reviews.rating <= 3, 'Negative',
                                 np.where(reviews.rating <= 6, 'Neutral', 'Positive'))
splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
_, validation_rows = next(splitter.split(reviews, groups=reviews.text_group))
validation = reviews.iloc[validation_rows]

# Select by ID within each true class, never by whether the model is correct.
examples = []
for experience, count in [('Negative', 3), ('Neutral', 3), ('Positive', 2)]:
    selected = validation.loc[validation.experience.eq(experience)].sort_values('uniqueID').head(count)
    for _, row in selected.iterrows():
        drug = '' if pd.isna(row.drugName) else str(row.drugName)
        condition = '' if pd.isna(row.condition) else str(row.condition)
        prediction = predict_patient_experience(drug, condition, row.review_clean)
        example = {'uniqueID': int(row.uniqueID), 'label': experience, 'drug': drug,
                   'condition': condition, 'review': row.review_clean,
                   'true_rating': int(row.rating)}
        examples.append(example)
        print(f"Demo {len(examples)}: predicted {prediction['predicted_experience_class']}, "
              f"rating {prediction['predicted_rating']}/10, score {prediction['confidence']:.3f}")

output_path = root / 'webapp' / 'backend' / 'examples.json'
output_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Prepared {len(examples)} genuine examples locally. This file is ignored by Git.')
print('Reload the application page to see the example buttons.')
