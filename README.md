# 📱 Predicting Smartphone Addiction

A machine learning project built for the **Kaggle Playground Series – Predicting Smartphone Addiction** competition.

The goal is to predict whether a user is likely to be addicted to smartphones based on demographic, academic, lifestyle, and behavioral features.

## 🏆 Kaggle Result

**Best ROC-AUC Score: `0.96054`**

This score was achieved using a **CatBoostClassifier** with preprocessing and feature engineering.

## 📊 Competition

**Competition:** Predicting Smartphone Addiction
**Platform:** Kaggle
**Competition:** Playground Series S6E8

The competition dataset contains user-level information related to smartphone usage and addiction.

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* Matplotlib
* Seaborn
* Jupyter / VS Code
* Git & GitHub
* Kaggle

## 📁 Project Structure

```text
smartphone-addiction-kaggle/
│
├── src/
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── requirements.txt
├── submission.csv
├── .gitignore
└── README.md
```

### Files

| File                         | Description                         |
| ---------------------------- | ----------------------------------- |
| `src/eda.py`                 | Exploratory Data Analysis           |
| `src/feature_engineering.py` | Feature creation and transformation |
| `src/preprocessing.py`       | Data preprocessing                  |
| `src/train.py`               | Model training                      |
| `src/predict.py`             | Generate test predictions           |
| `requirements.txt`           | Python dependencies                 |
| `submission.csv`             | Final Kaggle predictions            |

> The original datasets, virtual environment, and trained model files are excluded from GitHub using `.gitignore`.

## 🔍 Machine Learning Workflow

The project follows a typical machine learning pipeline:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train / Validation Split
   ↓
CatBoost Model
   ↓
Model Evaluation
   ↓
Prediction
   ↓
Kaggle Submission
```

## 📈 Model

The main model used in this project is:

### CatBoostClassifier

CatBoost was selected because it performs well with datasets containing categorical features and requires relatively little manual encoding.

Categorical features were handled directly by CatBoost after appropriate missing-value preprocessing.

## 🎯 Evaluation

The primary evaluation metric for the competition is **ROC-AUC**.

During experimentation, the model achieved approximately:

```text
Validation ROC-AUC ≈ 0.95
```

The final Kaggle submission achieved:

```text
Kaggle ROC-AUC = 0.96054
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/roshnichouhan/smartphone-addiction-kaggle.git
cd smartphone-addiction-kaggle
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Kaggle dataset

Place the competition files inside:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

The `data/` directory is intentionally excluded from GitHub.

### 5. Run the pipeline

Exploratory analysis:

```bash
python src/eda.py
```

Feature engineering:

```bash
python src/feature_engineering.py
```

Train the model:

```bash
python src/train.py
```

Generate predictions:

```bash
python src/predict.py
```

The resulting predictions can then be submitted to Kaggle.

## 💡 Key Learnings

Through this project, I practiced:

* Exploratory Data Analysis
* Handling categorical variables
* Missing-value treatment
* Feature engineering
* Train/validation splitting
* CatBoost classification
* ROC-AUC evaluation
* Kaggle submission workflow
* Building a reproducible ML project structure
* Using Git and GitHub for version control

## 📌 Future Improvements

Possible improvements include:

* Cross-validation
* Hyperparameter tuning
* Additional feature engineering
* Model ensembling
* Comparing CatBoost with XGBoost and LightGBM
* Experiment tracking
* Further Kaggle leaderboard optimization

## 👩‍💻 Author

**Roshni Chauhan**

GitHub: [roshnichouhan](https://github.com/roshnichouhan)

---

⭐ If you find this project useful, consider giving the repository a star!
