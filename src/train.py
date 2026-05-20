"""
End-to-end training script for the lead scoring pipeline.

Usage:
    python src/train.py

Reads data/bank-full.csv, trains the model, generates outputs/ charts,
and prints summary metrics.
"""

import os
import sys
import pickle

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import train_model, get_feature_importance
from scoring import score_leads, success_rate_by_tier, success_rate_by_decile
from visualizations import generate_all_plots


DATA_PATH = 'data/bank-full.csv'
OUTPUT_DIR = 'outputs'
MODEL_PATH = 'outputs/model.pkl'


def main():
    print("=" * 70)
    print("LEAD SCORING MODEL - TRAINING PIPELINE")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: {DATA_PATH} not found.")
        print("Download the UCI Bank Marketing dataset and place bank-full.csv in data/")
        print("See data/README.md for instructions.")
        sys.exit(1)

    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep=';')
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")

    print("\nTraining logistic regression model...")
    pipeline, X_test, y_test, y_pred_proba, metrics = train_model(df)

    print(f"\nBaseline conversion rate: {metrics['baseline_rate']:.1%}")
    print(f"Test set AUC: {metrics['test_auc']:.3f}")
    print(f"5-fold CV AUC: {metrics['cv_mean_auc']:.3f} +/- {metrics['cv_std_auc']:.3f}")

    print("\nScoring test set...")
    scored = score_leads(y_pred_proba, y_test.values)

    tier_summary = success_rate_by_tier(scored)
    print("\nConversion rate by tier:")
    print(tier_summary.to_string(index=False))

    decile_summary = success_rate_by_decile(scored)
    print("\nConversion rate by decile:")
    print(decile_summary.to_string(index=False))

    importance_df = get_feature_importance(pipeline, top_n=15)
    print("\nTop 10 features by absolute coefficient:")
    print(importance_df.head(10).to_string(index=False))

    print(f"\nGenerating charts in {OUTPUT_DIR}/...")
    files = generate_all_plots(
        scored, y_test, y_pred_proba, tier_summary, importance_df, OUTPUT_DIR
    )
    for f in files:
        print(f"  - {f}")

    print(f"\nSaving trained model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    scored_csv = os.path.join(OUTPUT_DIR, 'scored_test_set.csv')
    scored.to_csv(scored_csv, index=False)
    tier_summary.to_csv(os.path.join(OUTPUT_DIR, 'tier_summary.csv'), index=False)
    decile_summary.to_csv(os.path.join(OUTPUT_DIR, 'decile_summary.csv'), index=False)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

    print(f"\nPipeline complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
