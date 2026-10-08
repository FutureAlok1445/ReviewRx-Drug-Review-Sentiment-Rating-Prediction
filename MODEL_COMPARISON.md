# Current repository note

Earlier notebook/model copies were removed during cleanup. The historical results below remain as experiment evidence; current artifacts are in `models_ensemble/`. Patient-review data is not redistributed.

# Latest Model Comparison: Optuna and Ensembles

The extension requested after the first evaluation is complete. The selected sentiment
model is **Optuna weighted voting**, selected by common-validation macro-F1;
the selected rating model is **Unweighted rating logistic**, selected by MAE.
The previous completed notebook is preserved in
`experiments/Improved_ML_Training_Pre_Ensemble.ipynb`. Current artifacts are in
`models_ensemble`; `models_improved` remains unchanged.

## Common validation comparison

All candidates share 95,528 training records and 16,853 validation records. Three
StratifiedGroupKFold partitions use lowercased cleaned-text groups, with fold-local
TF-IDF. New candidates use the 80,000-term/min_df=2 word/bigram representation.
AdaBoost additionally fits SelectKBest(chi2, k=8000) inside each fold. No dense text
conversion, random train/test reshuffling, or test-label tuning was used.
For the current Colab budget, each boosting-tree split considers a random 10% of
these selected terms. The earlier full-feature tree search is retained separately
in `slow_AdaBoost_trials.csv` and `slow_adaboost_validation.csv`; its validation
macro-F1 was 0.4025. It was replaced for computational efficiency using training/CV
evidence, not because of test results.

Precision, recall and F1 are macro averages. CV Score is training-only macro-F1.

| Model | CV Score | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Optuna weighted voting | 0.6453 | 0.7656 | 0.6393 | 0.6687 | 0.6510 | 0.8775 |
| Tuned logistic | 0.6399 | 0.7532 | 0.6369 | 0.6718 | 0.6478 | 0.8710 |
| Optuna Logistic Regression | 0.6399 | 0.7468 | 0.6353 | 0.6727 | 0.6456 | 0.8713 |
| Word + character logistic | 0.6352 | 0.7525 | 0.6343 | 0.6674 | 0.6448 | 0.8705 |
| Original settings (fair split) | 0.6372 | 0.7437 | 0.6298 | 0.6655 | 0.6396 | 0.8697 |
| Richer logistic | 0.6336 | 0.7593 | 0.6296 | 0.6527 | 0.6387 | 0.8647 |
| Linear SVM | 0.6194 | 0.7745 | 0.6281 | 0.6273 | 0.6272 | 0.8512 |
| Optuna Complement NB | 0.5779 | 0.7560 | 0.5990 | 0.5996 | 0.5784 | 0.8530 |
| Complement NB | 0.5724 | 0.7565 | 0.5997 | 0.5964 | 0.5739 | 0.8534 |
| Optuna Bagging NB | 0.5706 | 0.7579 | 0.5939 | 0.5899 | 0.5696 | 0.8493 |
| Bagging + boosting voting | 0.5666 | 0.7628 | 0.6010 | 0.5852 | 0.5628 | 0.8512 |
| Optuna AdaBoost | 0.4094 | 0.6947 | 0.5459 | 0.4104 | 0.3966 | 0.7017 |

The new selected validation macro-F1 is 0.6510, compared with the prior
winner's 0.6478: **+0.32 percentage points**.
Accuracy is 0.7656. Highest accuracy alone is not the selection
criterion. Training macro-F1 is 0.7976; its validation gap is
0.1466. Ensemble inclusion does not itself prove
better generalization.

## Actual Optuna studies

Optuna 5.0.0 TPE used seed 42, n_startup_trials=2 for individual models; the voting
study used five startup trials. All fitted-model trials completed three grouped folds.
This bounded search is not exhaustive, not nested CV, and does not prove global
optimality. Voting weights use OOF predictions from tuned models on the same folds;
these CV scores therefore include parameter and weight-selection optimism.

- Logistic Regression: 6 trials; C between 0.15 and 4 (log scale); class weights None/balanced.
- Complement Naive Bayes: 6 trials; alpha between 0.01 and 5 (log scale).
- Bagging NB: 4 trials; 3/5 bootstrap estimators, max_samples 0.5/0.8, alpha 0.05–2.
- AdaBoost: 4 successful trials; tree depth 1/2, 20/40 rounds, learning rate 0.3–1.5.
  The feature-sampling fraction is fixed at 0.1. An additional four completed,
  expensive all-feature AdaBoost trials remain archived, outside the current search.
- Rating Complement NB: 6 trials minimizing grouped CV MAE, alpha 0.01–5.
- Weighted voting: 30 trials; each nonzero weight from 0.01, 0.1, 0.5, 1, 2.

Best individual parameters:

```json
{
  "Logistic Regression": {
    "C": 0.5809584204731072,
    "class_weight": "balanced"
  },
  "Complement NB": {
    "alpha": 0.10253509690168493
  },
  "Bagging NB": {
    "alpha": 0.17505961902039982,
    "n_estimators": 3,
    "max_samples": 0.5
  },
  "AdaBoost": {
    "max_depth": 2,
    "n_estimators": 40,
    "learning_rate": 1.3955612437469505
  }
}
```

Voting order: Logistic Regression, Complement NB, Bagging NB, AdaBoost.
Selected unnormalized voting weights: [1.0, 0.5, 0.01, 0.5].
The separate bagging + boosting ensemble uses equal weights for those two models.
An initial AdaBoost trial with balanced tree class weights failed its weak-learner
threshold. It received no metric and did not enter selection. AdaBoost's own sample
reweighting was retained while tree class weights were removed using training-fold
evidence; the failed run is retained in `failed_balanced_adaboost.log`.

## Rating comparison

| Model | CV MAE | MAE | RMSE | Exact accuracy | Within-1 accuracy |
| --- | --- | --- | --- | --- | --- |
| Unweighted rating logistic | 1.5561 | 1.5269 | 2.6272 | 0.4198 | 0.6795 |
| Tuned rating logistic | 1.5561 | 1.5269 | 2.6272 | 0.4198 | 0.6795 |
| Original rating settings (fair split) | 1.6315 | 1.6151 | 2.5995 | 0.3643 | 0.6379 |
| Optuna rating Complement NB | 1.7832 | 1.8273 | 3.0899 | 0.4181 | 0.6391 |

The previous tuned and unweighted rating Logistic Regression rows describe the same
configuration. The extension does not claim new rating improvement unless the new
candidate actually wins common validation.

## Repeated assessment of the previously evaluated test

**This is not a newly untouched test.** Its normalized texts remain absent from the
development CSV, but its earlier results were already viewed before this extension
was requested. We froze CV/validation selection before this repeat assessment and
did not change it afterward. Independent confirmation needs a new external cohort.

```json
{
  "experience": {
    "Accuracy": 0.7669908927575739,
    "Precision": 0.6355514723956858,
    "Recall": 0.6655778981509829,
    "F1": 0.6477480327621948,
    "ROC-AUC": 0.8767344023741428
  },
  "rating": {
    "MAE": 1.48274580261446,
    "RMSE": 2.575097222383476,
    "Exact accuracy": 0.4314478656836627,
    "Within-1 accuracy": 0.6874419180967722
  }
}
```

Trial CSVs, comparisons, parameter JSON and the extension summary are under
`experiments/ensemble`. The notebook contains real outputs, comparison/tuning charts,
overfitting checks, repeated-test confusion matrix, and saved-artifact reload checks.
Frontend/backend consume these recorded values; they do not display fabricated scores.

---

The following records document the earlier completed phase. References to its final
test describe the status at that time; that dataset is now previously evaluated.

# Model comparison

All new values below come from executed notebook predictions and grouped CV, stored in
`experiments/run_summary.json`. Precision, recall, F1, and ROC-AUC are macro averaged.
ROC-AUC is one-vs-rest; SVM uses decision margins. CV SD is fold variability, not a confidence interval.

## Historical results (not new experiments)

The supplied PDF quotes experience validation accuracy 0.7434/macro-F1 0.6341 in prose;
its printed classification report rounds these to 0.74/0.63. Rating printed output:
MAE 1.6040, RMSE 2.5791, exact accuracy 0.3648, within-1 accuracy 0.6434.
Benchmark commentary and printed outputs differ. The retrieved notebook prints experience benchmark
accuracy 0.7951/macro-F1 0.7134, whereas prose quotes 0.7942/0.7122. Do not use these inconsistent
historical benchmark claims to calculate an improvement. Full provided-test results are contaminated
by overlapping reviews and are not the final-test protocol below.

## Newly reproduced original settings and original split

Original cleaned-text GroupShuffleSplit: 137,147 training rows,
24,150 validation rows, seed 42. Original vocabulary 40,000,
word 1–2 grams, min_df=3, sublinear TF; logistic C=1, balanced weights, max_iter=1000.
Both tasks use the original settings under local library versions rather than historical sklearn 1.6.1.

| Task / split | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE | RMSE | Exact accuracy | Within-1 accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experience_train | 0.8723 | 0.7997 | 0.8837 | 0.8292 | 0.9718 | — | — | — | — |
| experience_validation | 0.7431 | 0.6239 | 0.6569 | 0.6338 | 0.8627 | — | — | — | — |
| rating_train | — | — | — | — | — | 0.8513 | 1.9802 | 0.6958 | 0.8167 |
| rating_validation | — | — | — | — | — | 1.6040 | 2.5791 | 0.3648 | 0.6434 |

The reproduced baseline has a confusion matrix and class report in section 09.
CV of the same original algorithm/settings on the improved common protocol appears below;
the historical original notebook did not run CV. There is no valid epoch loss history.

## Fair experience comparison: common grouped validation

Train 95,528; validation 16,853; normalized-text groups are disjoint.
CV is three training-only grouped folds, with TF-IDF fitted inside each fold.
The tuned model's CV scores participated in hyperparameter selection; validation did not.

| Model | CV Score | CV SD | Accuracy | Precision | Recall | F1 | ROC-AUC | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original settings (fair split) | 0.6372 | 0.0005 | 0.7437 | 0.6298 | 0.6655 | 0.6396 | 0.8697 | Same grouped folds; macro metrics |
| Richer logistic | 0.6336 | 0.0021 | 0.7593 | 0.6296 | 0.6527 | 0.6387 | 0.8647 | Same grouped folds; macro metrics |
| Linear SVM | 0.6194 | 0.0037 | 0.7745 | 0.6281 | 0.6273 | 0.6272 | 0.8512 | Same grouped folds; macro metrics |
| Complement NB | 0.5724 | 0.0009 | 0.7565 | 0.5997 | 0.5964 | 0.5739 | 0.8534 | Same grouped folds; macro metrics |
| Tuned logistic | 0.6399 | 0.0005 | 0.7532 | 0.6369 | 0.6718 | 0.6478 | 0.8710 | CV used for tuning; validation confirms |
| Word + character logistic | 0.6352 | 0.0008 | 0.7525 | 0.6343 | 0.6674 | 0.6448 | 0.8705 | Same grouped folds; macro metrics |

Selected by validation macro-F1: **Tuned logistic**.
Same-split baseline macro-F1 0.6396 → selected 0.6478,
**+0.82 percentage points**.
Training macro-F1 0.8312; validation 0.6478;
gap 0.1834. The model still overfits to some extent.
Chosen search settings: `{"model__class_weight": "balanced", "model__C": 1.0}`.
Selection uses validation F1; CV supports stability. It does not use training accuracy or test scores.

## Fair rating comparison: common grouped validation

MAE is the primary ordinal metric (lower is better). Exact accuracy and within-one accuracy
are reported separately. All outputs remain integer ratings. Both initial alternatives and
tuning results use the same rows and folds as the experience comparison.

| Model | MAE | RMSE | Exact accuracy | Within-1 accuracy | Train MAE | CV MAE | CV SD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original rating settings (fair split) | 1.6151 | 2.5995 | 0.3643 | 0.6379 | 0.9793 | 1.6315 | 0.0149 |
| Unweighted rating logistic | 1.5269 | 2.6272 | 0.4198 | 0.6795 | 0.4118 | 1.5561 | 0.0029 |
| Tuned rating logistic | 1.5269 | 2.6272 | 0.4198 | 0.6795 | 0.4118 | 1.5561 | 0.0029 |

Selected by validation MAE: **Unweighted rating logistic**.
Validation MAE improved while RMSE increased; the model makes smaller typical errors but still has large misses. Selection uses the predefined MAE objective, not a claim that every metric improved.
Chosen search settings: `{"model__class_weight": null, "model__C": 4.0}`.
Search tested four sampled combinations per task from C=[1,4,8], weights=[None,balanced],
using richer 80,000-term word/bigram TF-IDF with min_df=2. Exact search and fold values are
in `experiments/sentiment_search.csv`, `rating_search.csv`, and `run_summary.json`.
The selected rating search configuration matches the initial unweighted C=4 candidate;
this confirms that configuration rather than introducing a different algorithm. Its existing
fold scores are reused because the model settings, training rows, and folds are identical.

## Final reserved-test results (separate from validation)

After freezing choices, selected models and original-settings controls were refitted on
all 112,381 development examples. The final test contains
16,141 non-overlapping normalized provided-test text/rating examples.
No test vocabulary, label, prediction, or score determined hyperparameters or model choice.

| Model / task | Accuracy | Precision | Recall | F1 | ROC-AUC | MAE | RMSE | Exact accuracy | Within-1 accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected_experience | 0.7534 | 0.6339 | 0.6700 | 0.6451 | 0.8717 | — | — | — | — |
| original_settings_experience | 0.7490 | 0.6320 | 0.6706 | 0.6434 | 0.8692 | — | — | — | — |
| selected_rating | — | — | — | — | — | 1.4827 | 2.5751 | 0.4314 | 0.6874 |
| original_settings_rating | — | — | — | — | — | 1.6108 | 2.6095 | 0.3727 | 0.6391 |

Same-test macro-F1 gain: **+0.17 percentage points**.
Same-test rating MAE reduction: **+0.1281 rating points**.
These are the verified final improvements, rather than differences from a different historical split.
The final test is a same-source holdout; it does not prove performance on future patients or newer reviews.

## Explanation and practical limits

Measured terms associated with the selected validation model:
{
  "Negative": [
    "worse",
    "worst",
    "not recommend",
    "not worth",
    "horrible",
    "never",
    "not",
    "disappointed",
    "no relief",
    "terrible"
  ],
  "Neutral": [
    "but",
    "however",
    "hoping",
    "yet",
    "was great",
    "just started",
    "although",
    "works but",
    "helps with",
    "though"
  ],
  "Positive": [
    "love",
    "miracle",
    "amazing",
    "great",
    "works",
    "best",
    "love it",
    "saved",
    "years",
    "no side"
  ]
}

These are coefficient associations with rating-derived targets. They are not effects caused
by a medication. The notebook shows real validation errors, review-length-bin counts,
ROC/precision-recall curves, confusion matrices, and rating-error distributions.
Neutral validation precision is 0.3196;
recall is 0.4754.
Mixed text and noisy labels remain limitations. No isolated experiment attributes improvements
to metadata cleanup or deduplication alone. Training-only vocabulary, broader phrase coverage,
regularization/weight selection, and a cleaner evaluation protocol were justified systematically.
Complete Pipelines and original-compatible output dictionaries were saved and verified after reload.
