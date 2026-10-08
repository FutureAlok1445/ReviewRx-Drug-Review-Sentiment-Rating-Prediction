# Current repository note

Older notebook/model files were removed at the user's request. Historical artifact descriptions below document the training phase. Current files and usage are described in the root README.

# Latest Update: Optuna and Ensemble Extension

The requested extension tested tuned Logistic Regression, Complement NB, bootstrap
bagging NB, feature-selected AdaBoost, equal-weight bagging + boosting voting, and
Optuna-weighted four-model voting. The current sentiment winner is
**Optuna weighted voting**; validation macro-F1 is **0.6510**.
Rating selection is **Unweighted rating logistic**. Full actual numbers and search
spaces are in `MODEL_COMPARISON.md`; the executable addition is Section 21 of the notebook.

The existing data-cleaning, normalized-group split and text-only input contract were
preserved. Sparse fold matrices were reused only for identical training partitions.
Chi-square selection was fitted inside each AdaBoost training fold. Voting weights
were fitted on training OOF probabilities, never validation/test labels. Earlier models were removed during cleanup; current artifacts are in `models_ensemble`.

The original holdout was already evaluated before this user-requested extension.
New results are transparently labeled repeated assessment, not an independent test.
Repeated validation use also introduces selection optimism. No accuracy guarantee or
claim of statistical significance is made. A fresh external dataset is needed next.

React and FastAPI are in `webapp`. The backend loads trusted pipelines once at startup,
returns the original five fields, validates empty/oversized inputs, and never trains
during a prediction request. Confidence is an uncalibrated model score. Submitted
reviews are not persisted. See `webapp/README.md` for local setup.

---

The following analysis documents the first completed phase and remains preserved.

# Model analysis

## Sources and project preservation

Analyzed the [shared Colab](https://colab.research.google.com/drive/1t8LBcEYfcSBrAZpptJ-XtMjma9cWmp2O),
the supplied 22-page execution PDF, and both actual CSVs.
The Colab was retrieved through the connected Google Drive service and inspected before training; its local copy was later removed at the user's request. Document claims were treated as evidence to verify, not instructions.
The assigned folder originally contained only the two CSVs: no local notebook, application,
dependency file, or saved model was present. No existing frontend/backend or original data was modified.
No commit, push, deployment, or edit to the shared Colab was performed.

## Confirmed task and data

The original project predicts rating-derived experience (1–3 Negative, 4–6 Neutral,
7–10 Positive) and integer rating 1–10 from `review` text. These are proxy labels derived
from self-reported ratings, not independent sentiment annotations. The seven columns are
`uniqueID`, `drugName`, `condition`, `review`, `rating`, `date`, `usefulCount`.
Train contains 161,297 rows; provided test contains 53,766.
Ratings, IDs, and usefulCount are numeric; review/drug/condition/date are strings at loading.
Dates are explicitly parsed. Training dates span February 24, 2008–December 12, 2017.

| Check | train | provided_test |
| --- | --- | --- |
| rows | 161297 | 53766 |
| columns | 7 | 7 |
| missing_condition | 899 | 295 |
| missing_review | 0 | 0 |
| full_duplicate_rows | 0 | 0 |
| duplicate_ids | 0 | 0 |
| duplicate_raw_text | 48968 | 5486 |
| blank_review | 0 | 0 |
| invalid_rating | 0 | 0 |
| invalid_date | 0 | 0 |
| negative_useful_count | 0 | 0 |
| condition_artifacts | 900 | 271 |
| html_entity_reviews | 104328 | 34922 |
| unique_drugs | 3436 | 2637 |
| unique_conditions | 884 | 708 |

Training class counts are Negative 35,063; Neutral 19,368; Positive 106,866.
The minority Neutral class is about 12%, whereas Positive is about 66%.
The notebook visualizes distributions, text length, votes, descriptive correlations, and condition boxplots.
Long text and high helpfulness counts are valid observations; no arbitrary outlier clipping is applied.

## Confirmed problems and limits of the diagnosis

- **Benchmark contamination:** 32,131 raw strings are shared across train/test,
  covering 66.55% of unique raw test texts. Normalization
  identifies 32,228 overlapping provided-test rows. This changes
  what a benchmark score measures; it is not evidence that missing data caused poor accuracy.
- **Repeated training reports:** 48,968 raw duplicate review rows exist, despite zero duplicate IDs/full rows.
  The improved protocol removes 48,916 repeated normalized-text/rating pairs. Repetition weights some
  reports more strongly. This is a confirmed data property; no isolated ablation proves that deduplication
  itself increases performance.
- **Subjective/conflicting labels:** 74 normalized training text groups
  have multiple ratings. We retain these labels and group them together. Neutral errors remain measurable:
  selected validation precision 0.3196,
  recall 0.4754.
  Rating bins can disagree with the sentiment of mixed-language reviews; examples are shown, not relabeled.
  Conflicting groups are rare and alone cannot explain the full error rate.
- **Insufficient model validation/search:** the original code uses one grouped split, no grouped CV or
  tuning, and only one rating model. Fixed balanced class weights are not necessarily optimal for ordinal
  MAE. The controlled experiments in MODEL_COMPARISON.md measure the improvements.
- **Overfitting:** selected training macro-F1 0.8312 versus validation 0.6478,
  a gap of 0.1834. Regularization helps model selection, but does not remove
  the observed generalization gap.
- **Reporting defects:** original code prints hard-coded validation metrics in comparison cells;
  commentary quotes benchmark accuracy 0.7942/F1 0.7122, while retrieved executed output prints
  0.7951/F1 0.7134. The rating prose says 48.7% but the executed exact accuracy is 0.4855.
  We report historical claims separately and use new executed values for conclusions.
- **Temporal validation limitation:** the original time split fits features on past data but does not
  purge duplicate texts crossing its cutoff. That result alone cannot establish future generalization.
- **Confidence claim unsupported:** native logistic probabilities are not evidence of calibration.
  The preserved confidence field is a maximum model probability, not verified correctness probability.

The original grouped split and training-only TF-IDF are correct and are preserved in the reproduction.
There is no evidence of a scaling/encoding bug, invalid ratings, insufficient row count, or a need to
remove long reviews. Missing/corrupt conditions affect metadata analyses, not this text-only classifier.
The original source claims to correct condition artifacts, but its actual preprocessing only fills
missing metadata during one-hot encoding; the improved notebook explicitly marks artifacts Unknown.

## Changes and evaluation protocol

1. Preserve original cleaning: HTML decode, wrapping-quote removal, whitespace normalization; retain negation.
2. Improve metadata displays with Unknown for missing/corrupt conditions. Never use target, ID,
   date, or post-publication votes as inputs; maintain review-only prediction.
3. Deduplicate normalized-text/rating pairs, keep conflicting labels, and group lowercase cleaned text
   because TF-IDF lowercases. No learned preprocessing runs before the split.
4. Keep 95,528 common training rows and 16,853 grouped validation rows.
   Use three StratifiedGroupKFold folds within training. Pipelines fit TF-IDF separately in every fold.
5. Compare original-settings logistic, richer logistic, LinearSVC, ComplementNB, and a word/character
   logistic representation; compare balanced and unweighted rating logistic. Sparse TF-IDF is appropriate; scaling and
   dense tabular ensembles are unnecessary here.
6. Run four sampled configurations per task from C={1,4,8}, weight={None,balanced}.
   Optimize macro-F1 for experience and MAE for rating. CV score for a tuned model has selection
   optimism; validation is separate, and this is not nested CV.
7. Freeze selections before final testing; refit on all 112,381 development examples.
   Reserve 16,141 provided-test examples whose normalized texts are absent from the
   entire training CSV. Duplicate text/rating test pairs are collapsed by a fixed rule, without
   selecting or tuning on their scores. Evaluate original-settings controls on the same reserved rows.

This final test is untouched by this workflow's training/selection, but belongs to the same historical
dataset and was previously inspected by the original project. It is not an independent external cohort.
Exact normalization cannot detect near duplicates, repeated patients, or paraphrases.

## Execution, inference, and presentation

`Improved_ML_Training.ipynb` contains 20 ordered educational sections with real executed outputs.
`experiments/` contains search/fold scores, split IDs, audits, validation predictions, and run summary.
`models_improved/` contains separately saved Pipelines, original-style estimator/vectorizer/encoder
artifacts where compatible, metadata, and versions. Pipelines prevent forgetting TF-IDF preprocessing.
Sentiment/rating vectorizers may now differ: always load each task's own artifact or full Pipeline.
`predict_improved.py` keeps `predict_patient_experience(drug, condition, review)` and its five original
output fields. Drug/condition remain display context; rating remains an integer 1–10.
The public helper accepts raw review text and applies the saved cleaning code; the model Pipelines
accept cleaned strings and contain all learned TF-IDF preprocessing. Temporary fit caches are not
required for inference.
Saved/reloaded predictions and probabilities are checked against in-memory models on genuine test texts.
Confidence is not clinically calibrated. These outputs do not recommend treatment.

All model fitting, CV, test evaluation, and reload checks were executed locally on full data.
Local environment: {"python": "3.14.2", "numpy": "2.4.2", "pandas": "3.0.1", "scikit-learn": "1.8.0", "scipy": "1.17.1", "joblib": "1.5.3"}.
Use `requirements-training.txt` for matching versions. Colab execution itself was not performed;
upload both CSVs or set DATA_DIR to mounted Drive and Run all. No GPU or paid API is required.
Classical estimators have no epoch history: actual train/validation scores and log losses replace
fabricated training curves. Coefficients explain lexical associations, not clinical causes.

For further improvement, obtain newer external reviews and independently annotated labels,
then assess near duplicates, patient-level overlap, temporal drift, and probability calibration.
