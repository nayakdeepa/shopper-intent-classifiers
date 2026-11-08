# Shopper Intent Predictor: Predicting Online Purchase Intent with Five Classifiers

## a. Problem Statement

Online shops lose most of their visitors without a sale: in this dataset only about 15.6% of browsing sessions end in a purchase. The goal of this project is to predict, from the behaviour recorded during a session (pages viewed, time spent, bounce and exit rates, page value, month, visitor type and so on), whether the session will end in a purchase (`Revenue = 1`).

This is a **binary classification** problem with imbalanced classes. Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data and compared with six metrics: Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC). The trained models are served through an interactive Streamlit app deployed on Streamlit Community Cloud.

## b. Dataset Description

| Item | Details |
|---|---|
| Source | UCI Machine Learning Repository, *Online Shoppers Purchasing Intention Dataset* (id 468): [https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| Citation and licence | Sakar, C. & Kastro, Y. (2018). DOI 10.24432/C5F88Q. Licence: CC BY 4.0 |
| Original size | 12,330 sessions, 17 features + 1 target |
| Instances used | 12,205 (125 exact duplicate sessions removed) |
| Features | 22 (17 original + 5 engineered): 15 numeric and 7 categorical, expanding to 81 columns after one-hot encoding |
| Target column | `Revenue`: 1 = session ended in a purchase, 0 = no purchase |
| Task type | Binary classification |
| Class balance | 0: 10,297 sessions (84.4%), 1: 1,908 sessions (15.6%) |
| Missing values | None |
| Train / test split | 80 / 20, stratified, `random_state = 7`: 9,764 training and 2,441 test sessions (2,059 non-purchase, 382 purchase) |

**Feature groups**

| Group | Features |
|---|---|
| Page visits and time spent | Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration |
| Google Analytics metrics | BounceRates, ExitRates, PageValues |
| Calendar | SpecialDay, Month, Weekend, Quarter (engineered) |
| Technical and visitor | OperatingSystems, Browser, Region, TrafficType, VisitorType |
| Engineered | TotalPages, TotalDuration, AvgTimePerPage, ProductPageShare, Quarter |

**Data preparation** (`prepare_dataset.py`, applied to the original UCI file):

- **Cleaning:** removed 125 exact duplicate sessions so identical rows cannot appear in both the training and the test set.
- **Type fixes:** OperatingSystems, Browser, Region and TrafficType are integer codes, not quantities, so they were relabelled as categories (for example `OS_2`, `Region_3`) and one-hot encoded. Weekend and Revenue were converted to 0/1.
- **Feature engineering:** TotalPages (sum of the three page counts), TotalDuration (sum of the three durations), AvgTimePerPage, ProductPageShare and Quarter (derived from Month).

**Preprocessing inside every saved model:** numeric columns use median imputation and standard scaling; categorical columns use most-frequent imputation and one-hot encoding (unknown categories ignored). The recipe is stored together with each classifier in one scikit-learn Pipeline, so the app can score a raw CSV without retraining anything.

## c. GitHub Repository Link

Repository: [https://github.com/nayakdeepa/shopper-intent-classifiers](https://github.com/nayakdeepa/shopper-intent-classifiers)

Live Streamlit app: [https://shopper-intent-classifiers-82bgaqubalx8dnoc87dkv6.streamlit.app/](https://shopper-intent-classifiers-82bgaqubalx8dnoc87dkv6.streamlit.app/)

```
shopper-intent-classifiers/
|-- app.py                          Streamlit app
|-- requirements.txt                pinned dependencies
|-- README.md
|-- test_data.csv                   held-out test set (2,441 rows) used for every metric below
|-- train_models.py                 trains and evaluates the 5 models, saves the artifacts
|-- prepare_dataset.py              builds the prepared dataset from the original UCI file
|-- online_shoppers_intention.csv   original UCI data
|-- online_shoppers_prepared.csv    cleaned data with engineered features
`-- model/                          5 x .joblib pipelines, meta.json, metrics_summary.csv
```

## d. Models Used

All five models use the same preprocessing and the same train/test split. Settings: Logistic Regression (`max_iter=2000`), Decision Tree (`max_depth=8`, `min_samples_leaf=5`), KNN (`n_neighbors=7`), Gaussian Naive Bayes (`var_smoothing=0.1`), SVM (RBF kernel, `C=1.0`, `probability=True`); `random_state=7` wherever applicable. `var_smoothing` was chosen by 5-fold cross-validation on the training split only, because the default (1e-9) collapsed to 0.27 accuracy on the one-hot features.

### Comparison table (held-out test set, 2,441 sessions)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8837 | 0.8982 | 0.7526 | 0.3822 | 0.5069 | 0.4821 |
| Decision Tree | **0.8943** | **0.8997** | 0.7296 | 0.5157 | **0.6043** | **0.5563** |
| KNN | 0.8759 | 0.8038 | 0.7047 | 0.3560 | 0.4730 | 0.4421 |
| Naive Bayes | 0.7997 | 0.8301 | 0.4119 | **0.6545** | 0.5056 | 0.4044 |
| Support Vector Machine | 0.8927 | 0.8909 | **0.7752** | 0.4424 | 0.5633 | 0.5333 |

Precision, Recall and F1 are for the positive class (purchase = 1). The best value in each column is in bold.

| ML Model Name | True negatives | False positives | False negatives | True positives |
|---|---|---|---|---|
| Logistic Regression | 2011 | 48 | 236 | 146 |
| Decision Tree | 1986 | 73 | 185 | 197 |
| KNN | 2002 | 57 | 246 | 136 |
| Naive Bayes | 1702 | 357 | 132 | 250 |
| Support Vector Machine | 2010 | 49 | 213 | 169 |

### Observations on model performance

Only 15.6% of sessions are purchases, so a model that always predicts "no purchase" would already score about 0.84 accuracy. Accuracy differences between the models are therefore small, and MCC, F1 and recall are more informative.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.884 and AUC 0.898, nearly as high as the best model, so it ranks purchase sessions well. At the default 0.5 threshold it is cautious: precision is good (0.75, 146 of the 194 sessions it flagged were real purchases) but recall is only 0.38, so it misses 236 of 382 buyers. Its largest coefficient belongs to PageValues. A linear boundary limits how much it can capture. |
| Decision Tree | Best on four of the six metrics: accuracy 0.894, AUC 0.900, F1 0.604 and MCC 0.556. It catches 197 of 382 buyers (recall 0.52) with 73 false alarms. The fitted tree relies heavily on PageValues (about 64% of its feature importance), which suggests that sessions that visit pages with a non-zero page value are the clearest purchase signal. The depth and leaf-size limits are there to curb overfitting, but a single tree can be less stable than an ensemble. |
| KNN | Lowest AUC (0.804), recall (0.356) and F1 (0.473) of the five models; it finds only 136 of 382 buyers. After one-hot encoding the data has 81 columns, which probably dilutes the distances KNN depends on (a likely reason, not tested here). k = 7 was not tuned, and the model has to store the whole training set. |
| Naive Bayes | Lowest accuracy (0.800), precision (0.412) and MCC (0.404), but the highest recall (0.655): it finds 250 of 382 buyers at the price of 357 false alarms, the most of any model. Its independence assumption fits poorly because many features are related (TotalPages is the sum of three page counts, for example) and Gaussian densities suit one-hot columns badly. It needed tuned smoothing to be usable at all. It is only attractive when missing a buyer is much costlier than a false alarm. |
| Support Vector Machine | Close second to the Decision Tree: MCC 0.533, F1 0.563, accuracy 0.893, AUC 0.891 and the highest precision (0.775; 169 of the 218 sessions it flagged were purchases). The RBF kernel can model non-linear interactions that logistic regression cannot. Recall is still modest (0.442), and `probability=True` (needed for AUC) adds extra internal cross-validation to training. |
| **Overall, Winner for your dataset?** | **Decision Tree.** It has the highest MCC (0.556), F1 (0.604), accuracy (0.894) and AUC (0.900). Because of the class imbalance the choice is based on MCC and F1 rather than accuracy. The SVM is a close second and is preferable if precision matters most, while Naive Bayes suits a recall-first goal. Caveat: results come from one stratified 80/20 split (seed 7); the gap between the Decision Tree and the SVM is small and could change with a different split or cross-validation. |

## How to Run Locally

```
pip install -r requirements.txt
python prepare_dataset.py --src online_shoppers_intention.csv --out online_shoppers_prepared.csv
python train_models.py --data online_shoppers_prepared.csv --target Revenue --seed 7
streamlit run app.py
```

Training also writes the saved pipelines to `model/` and the held-out `test_data.csv`.

## Streamlit App Features

- CSV upload of test data (it must contain all 22 feature columns and the `Revenue` column); a bundled-sample checkbox is also available.
- Model selection dropdown with all five classifiers.
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1 and MCC.
- Confusion matrix, classification report and ROC curve for the selected model.
- Model comparison tab (all five models on the uploaded data), data and predictions tab with CSV download, and an About tab.
- Saved models are loaded from disk; nothing is retrained on the uploaded data.

Uploading `test_data.csv` in the app reproduces the comparison table above.
