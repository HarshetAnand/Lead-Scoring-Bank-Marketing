"""
Scoring layer that converts model probabilities into a 0-100 lead score.

Provides bucketing logic, score-tier labels, and success rate analysis
by score range. This is the layer that makes the model output usable
by a sales or marketing team.
"""

import pandas as pd
import numpy as np


def probability_to_score(probabilities):
    """Convert model probabilities to a 0-100 scale."""
    return np.clip(np.round(probabilities * 100), 0, 100).astype(int)


def assign_tier(score):
    """Assign a tier label to a numeric score."""
    if score >= 80:
        return 'Hot (80-100)'
    elif score >= 60:
        return 'Warm (60-79)'
    elif score >= 40:
        return 'Lukewarm (40-59)'
    elif score >= 20:
        return 'Cold (20-39)'
    else:
        return 'Frozen (0-19)'


def score_leads(probabilities, actuals=None):
    """
    Build a scored dataframe from model probabilities.

    If actuals are provided, includes the true outcome for analysis.
    """
    scores = probability_to_score(probabilities)
    tiers = [assign_tier(s) for s in scores]

    result = pd.DataFrame({
        'probability': probabilities,
        'score': scores,
        'tier': tiers
    })

    if actuals is not None:
        result['actual_outcome'] = np.asarray(actuals)

    return result


def success_rate_by_tier(scored_df):
    """Calculate success rate within each score tier."""
    if 'actual_outcome' not in scored_df.columns:
        raise ValueError("scored_df must contain actual_outcome column")

    tier_order = [
        'Frozen (0-19)', 'Cold (20-39)', 'Lukewarm (40-59)',
        'Warm (60-79)', 'Hot (80-100)'
    ]

    summary = scored_df.groupby('tier').agg(
        leads=('score', 'count'),
        success_rate=('actual_outcome', 'mean'),
        avg_score=('score', 'mean')
    ).reset_index()

    summary['tier'] = pd.Categorical(summary['tier'], categories=tier_order, ordered=True)
    summary = summary.sort_values('tier').reset_index(drop=True)
    summary['success_rate'] = (summary['success_rate'] * 100).round(1)
    summary['avg_score'] = summary['avg_score'].round(1)

    return summary


def success_rate_by_decile(scored_df):
    """Calculate success rate by score decile for lift analysis."""
    if 'actual_outcome' not in scored_df.columns:
        raise ValueError("scored_df must contain actual_outcome column")

    df = scored_df.copy()
    df['decile'] = pd.qcut(df['score'], q=10, labels=False, duplicates='drop') + 1

    summary = df.groupby('decile').agg(
        leads=('score', 'count'),
        success_rate=('actual_outcome', 'mean'),
        avg_score=('score', 'mean'),
        min_score=('score', 'min'),
        max_score=('score', 'max')
    ).reset_index()

    summary['success_rate'] = (summary['success_rate'] * 100).round(1)
    summary['avg_score'] = summary['avg_score'].round(1)

    return summary
