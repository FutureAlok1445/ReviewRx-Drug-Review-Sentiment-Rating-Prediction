# Dataset provenance and local access

The experiments used the two supplied CSVs. Their filenames, schema and row counts match the commonly distributed Drug Review Dataset (Drugs.com). The exact original download location of the supplied copies was not recorded.

## Source and credit

- [Kaggle mirror/source](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018).
- [Historical UCI entry](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com), unavailable during the 9 October 2026 publication check.
- Felix Gräßer, Surya Kallumadi, Hagen Malberg and Sebastian Zaunseder. 2018. *Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning*. Proceedings of the 2018 International Conference on Digital Health, pages 121–125.

The official Kaggle API reported `Other (specified in description)` as the license. Its acknowledgments restrict use to research, prohibit commercial use and prohibit redistribution. These terms were checked on 9 October 2026. Raw CSVs, copied review examples and individual-record output tables are therefore excluded from Git. Do not infer a permissive license from a public download page.

## Obtain and use locally

1. Obtain the dataset directly from its source and follow the terms.
2. Put `drugsComTrain_raw.csv` and `drugsComTest_raw.csv` in the project root.
3. Run the notebook for training or `python prepare_demo_examples.py` for eight local genuine examples.
4. Do not publicly commit CSVs, generated patient-review examples or notebook outputs containing review rows.

The saved-model app can predict manually entered text without these CSVs. No new scraping or live patient-data collection is used.

## Exact supplied files behind the recorded results

| File | Rows | SHA-256 |
|---|---:|---|
| `drugsComTrain_raw.csv` | 161297 | `bc0902b21af0db5b99568a7639ed5f7dad3494a8186a20d05c5b4016c6caedf4` |
| `drugsComTest_raw.csv` | 53766 | `dfdaa5245b3f24e2af2b26ff4b21b798159c2d54184da9846c9458b83d1897c3` |

Different mirrors/conversions may change byte hashes. Check the schema and splits before claiming exact reproduction.
