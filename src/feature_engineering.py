"""
Feature engineering for lead scoring on UCI Bank Marketing dataset.

Mirrors the bucketing and feature creation approach used in real-world
lead scoring systems. Converts raw inputs into model-ready features
and human-readable buckets for downstream scoring logic.
"""

import pandas as pd
import numpy as np


def create_age_bucket(age):
    """Bucket age into business-meaningful ranges."""
    try:
        age = float(age)
        if age < 30:
            return 'Under 30'
        elif age < 40:
            return '30-39'
        elif age < 50:
            return '40-49'
        elif age < 60:
            return '50-59'
        else:
            return '60+'
    except (ValueError, TypeError):
        return 'Unknown'


def create_balance_bucket(balance):
    """Bucket account balance into deciles meaningful for outreach."""
    try:
        balance = float(balance)
        if balance < 0:
            return 'Negative'
        elif balance < 500:
            return 'Low (0-500)'
        elif balance < 2000:
            return 'Medium (500-2K)'
        elif balance < 5000:
            return 'High (2K-5K)'
        else:
            return 'Very High (5K+)'
    except (ValueError, TypeError):
        return 'Unknown'


def create_contact_intensity(row):
    """Combine campaign and previous contact features."""
    try:
        campaign = float(row.get('campaign', 0))
        previous = float(row.get('previous', 0))
        total = campaign + previous
        if total <= 1:
            return 'First Touch'
        elif total <= 3:
            return 'Light (2-3)'
        elif total <= 6:
            return 'Medium (4-6)'
        else:
            return 'Heavy (7+)'
    except (ValueError, TypeError):
        return 'Unknown'


def create_duration_bucket(duration):
    """Bucket call duration in seconds."""
    try:
        duration = float(duration)
        if duration < 60:
            return 'Very Short (<1m)'
        elif duration < 180:
            return 'Short (1-3m)'
        elif duration < 360:
            return 'Medium (3-6m)'
        elif duration < 600:
            return 'Long (6-10m)'
        else:
            return 'Very Long (10m+)'
    except (ValueError, TypeError):
        return 'Unknown'


def create_prior_outcome_feature(outcome):
    """Simplify prior campaign outcome."""
    if pd.isna(outcome):
        return 'No Prior Contact'
    outcome = str(outcome).strip().lower()
    if outcome == 'success':
        return 'Prior Success'
    elif outcome == 'failure':
        return 'Prior Failure'
    elif outcome == 'other':
        return 'Prior Other'
    else:
        return 'No Prior Contact'


def engineer_features(df):
    """
    Apply all feature engineering to the raw dataframe.

    Returns a new dataframe with engineered features added alongside
    the original columns.
    """
    df = df.copy()

    df['age_bucket'] = df['age'].apply(create_age_bucket)
    df['balance_bucket'] = df['balance'].apply(create_balance_bucket)
    df['contact_intensity'] = df.apply(create_contact_intensity, axis=1)
    df['duration_bucket'] = df['duration'].apply(create_duration_bucket)
    df['prior_outcome'] = df['poutcome'].apply(create_prior_outcome_feature)

    # Binary flag engineering
    df['has_housing_loan'] = (df['housing'].astype(str).str.lower() == 'yes').astype(int)
    df['has_personal_loan'] = (df['loan'].astype(str).str.lower() == 'yes').astype(int)
    df['has_default'] = (df['default'].astype(str).str.lower() == 'yes').astype(int)

    return df


def get_categorical_features():
    """Return list of categorical features used in modeling."""
    return [
        'job', 'marital', 'education', 'contact', 'month',
        'age_bucket', 'balance_bucket', 'contact_intensity',
        'duration_bucket', 'prior_outcome'
    ]


def get_numerical_features():
    """Return list of numerical features used in modeling."""
    return [
        'age', 'balance', 'duration', 'campaign', 'previous',
        'has_housing_loan', 'has_personal_loan', 'has_default'
    ]
