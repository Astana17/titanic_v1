"""
train.py — Train the best Titanic survival model and save it.

Usage:
    python scripts/train.py

Outputs:
    scripts/best_model.pkl   — trained VotingClassifier
    scripts/features.json    — list of feature names used
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')

TRAIN_CSV  = os.path.join(DATA_DIR, 'train.csv')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_model.pkl')
FEAT_PATH  = os.path.join(SCRIPT_DIR, 'features.json')


# ── Feature engineering (shared with predict.py) ─────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to a dataframe."""
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
        if s == 1:   return 'Alone'
        if s <= 4:   return 'Small'
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
        df['Fare'],
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


FEATURES = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked', 'Title', 'FamilySize', 'IsAlone', 'FamilyType',
    'Deck', 'AgeBin', 'FareBin', 'HasCabin',
]


def build_model() -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=6,
        min_samples_split=10, min_samples_leaf=2,
        max_features='sqrt', random_state=42,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.8,
        min_samples_split=10, random_state=42,
    )
    lr  = LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs', random_state=42)
    svc = SVC(probability=True, kernel='rbf', C=1, gamma='scale', random_state=42)

    return VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr), ('svc', svc)],
        voting='soft',
    )


def main() -> None:
    print('Loading training data …')
    train = pd.read_csv(TRAIN_CSV)

    print('Engineering features …')
    train = engineer_features(train)

    # Fill any residual NaN from pd.cut/pd.qcut edge cases
    X = train[FEATURES].fillna(0).values
    y = train['Survived'].astype(int).values

    # Cross-validation
    print('Running 5-fold cross-validation …')
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model  = build_model()
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f'CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}')

    if scores.mean() < 0.789:
        print('WARNING: CV accuracy is below the 78.9% minimum!', file=sys.stderr)

    # Train on full dataset
    print('Training on full dataset …')
    model.fit(X, y)
    train_acc = accuracy_score(y, model.predict(X))
    print(f'Train accuracy (optimistic): {train_acc:.4f}')

    # Save artefacts
    joblib.dump(model, MODEL_PATH)
    with open(FEAT_PATH, 'w') as fh:
        json.dump(FEATURES, fh)

    print(f'Model saved  → {MODEL_PATH}')
    print(f'Features saved → {FEAT_PATH}')


if __name__ == '__main__':
    main()
