"""
Logistic regression model for lead scoring.

Trains a logistic regression classifier on the UCI Bank Marketing dataset
to predict term deposit subscription. Outputs AUC, predicted probabilities,
and feature importances.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import (
    engineer_features,
    get_categorical_features,
    get_numerical_features
)


def build_pipeline():
    """Build the preprocessing and modeling pipeline."""
    categorical_features = get_categorical_features()
    numerical_features = get_numerical_features()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])

    return pipeline


def train_model(df, target_column='y', test_size=0.2, random_state=42):
    """
    Train logistic regression on the engineered dataset.

    Returns trained pipeline, test set, predictions, and metrics dict.
    """
    df = engineer_features(df)

    # Map target to binary
    target_map = {'yes': 1, 'no': 0, 'True': 1, 'False': 0, True: 1, False: 0, 1: 1, 0: 0}
    if df[target_column].dtype == 'object' or pd.api.types.is_string_dtype(df[target_column]):
        df[target_column] = df[target_column].map(target_map)
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce').fillna(0).astype(int)

    feature_cols = get_categorical_features() + get_numerical_features()
    X = df[feature_cols]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    cv_scores = cross_val_score(
        pipeline, X, y, cv=5, scoring='roc_auc', n_jobs=-1
    )

    metrics = {
        'test_auc': auc,
        'cv_mean_auc': cv_scores.mean(),
        'cv_std_auc': cv_scores.std(),
        'baseline_rate': y.mean(),
        'classification_report': classification_report(y_test, y_pred)
    }

    return pipeline, X_test, y_test, y_pred_proba, metrics


def get_feature_importance(pipeline, top_n=20):
    """Extract and rank feature importances from the trained model."""
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']

    num_features = get_numerical_features()
    cat_features = get_categorical_features()

    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(cat_features)

    all_features = list(num_features) + list(cat_feature_names)
    coefficients = classifier.coef_[0]

    importance_df = pd.DataFrame({
        'feature': all_features,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)

    return importance_df.head(top_n).reset_index(drop=True)
