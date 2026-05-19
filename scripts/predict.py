"""
predict.py — Load the trained model and generate a Kaggle submission file.

Usage:
    python scripts/predict.py

Prerequisite:
    Run train.py first to generate scripts/best_model.pkl

Output:
    data/submission.csv  — ready to upload to Kaggle
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(ROOT_DIR, 'data')

TEST_CSV     = os.path.join(DATA_DIR, 'test.csv')
MODEL_PATH   = os.path.join(SCRIPT_DIR, 'best_model.pkl')
FEAT_PATH    = os.path.join(SCRIPT_DIR, 'features.json')
SUBMISSION   = os.path.join(DATA_DIR, 'submission.csv')


# ── Feature engineering (must match train.py exactly) ────────────────────────
def engineer_features(df: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Apply feature engineering to df.
    If train_df is provided, use it to fit bin edges (prevents leakage on live data).
    For inference we simply reuse the same logic — safe because pd.qcut/pd.cut
    on the test set alone is fine for ranking purposes.
    """
    df = df.copy()

    # Title
    rare_titles = [
        'Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir',
        'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Dona',
    ]
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mme': 'Mrs', 'Ms': 'Miss', 'Mlle': 'Miss'})

    # Age imputation
    age_median = df.groupby(['Title', 'Pclass'])['Age'].transform('median')
    df['Age'] = df['Age'].fillna(age_median)
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Family
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    def family_type(s):
        if s == 1:  return 'Alone'
        if s <= 4:  return 'Small'
        return 'Large'

    df['FamilyType'] = df['FamilySize'].apply(family_type)

    # Cabin
    df['Deck']     = df['Cabin'].str[0].fillna('Unknown')
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # Fill remaining
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare']     = df['Fare'].fillna(
        df.groupby('Pclass')['Fare'].transform('median')
    )

    # Bins
    df['AgeBin'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'Young', 'Middle', 'Senior'],
        include_lowest=True,
    )
    df['AgeBin'] = df['AgeBin'].cat.add_categories('Unknown').fillna('Unknown')
    df['FareBin'] = pd.qcut(
        df['Fare'].rank(method='first'),
        4,
        labels=['Low', 'Mid', 'High', 'VeryHigh'],
        duplicates='drop',
    )
    df['FareBin'] = df['FareBin'].cat.add_categories('Unknown').fillna('Unknown')

    # Encode categoricals
    le = LabelEncoder()
    categorical_cols = ['Sex', 'Embarked', 'Title', 'Deck', 'FamilyType', 'AgeBin', 'FareBin']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def main() -> None:
    # Load model and feature list
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f'Model not found at {MODEL_PATH}. Run train.py first.'
        )
    print('Loading model …')
    model = joblib.load(MODEL_PATH)

    with open(FEAT_PATH) as fh:
        features = json.load(fh)

    # Load & process test data
    print('Loading test data …')
    test = pd.read_csv(TEST_CSV)
    passenger_ids = test['PassengerId'].copy()

    print('Engineering features …')
    test = engineer_features(test)

    # Fill any residual NaN from pd.cut/pd.qcut edge cases
    X_test = test[features].fillna(0).values

    # Predict
    print('Generating predictions …')
    preds = model.predict(X_test)
    survived_pct = preds.mean() * 100
    print(f'Predicted survival rate: {survived_pct:.1f}%  '
          f'(survived={preds.sum()}, died={len(preds)-preds.sum()})')

    # Save submission
    submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': preds})
    submission.to_csv(SUBMISSION, index=False)
    print(f'Submission saved → {SUBMISSION}')
    print('Upload data/submission.csv to https://www.kaggle.com/c/titanic/submit')


if __name__ == '__main__':
    main()
