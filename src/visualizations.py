"""
Visualization utilities for the lead scoring project.

Generates the four core charts saved to outputs/:
1. Score distribution
2. ROC curve
3. Success rate by tier
4. Feature importance
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_score_distribution(scored_df, save_path):
    """Plot score distribution histogram, optionally split by outcome."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'actual_outcome' in scored_df.columns:
        failed = scored_df[scored_df['actual_outcome'] == 0]['score']
        success = scored_df[scored_df['actual_outcome'] == 1]['score']
        ax.hist([failed, success], bins=25, alpha=0.75,
                label=['Did Not Convert', 'Converted'],
                color=['#d4574e', '#4e9b6b'], edgecolor='white')
        ax.legend(loc='upper right', frameon=False)
    else:
        ax.hist(scored_df['score'], bins=25, alpha=0.8,
                color='#4a7ab8', edgecolor='white')

    ax.set_xlabel('Lead Score (0-100)', fontsize=11)
    ax.set_ylabel('Number of Leads', fontsize=11)
    ax.set_title('Lead Score Distribution', fontsize=13, pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path):
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='#4a7ab8', linewidth=2.5,
            label=f'Logistic Regression (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#999999', linestyle='--', linewidth=1.2,
            label='Random Baseline')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=13, pad=15)
    ax.legend(loc='lower right', frameon=False)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_success_by_tier(tier_summary, save_path):
    """Bar chart of success rate by tier."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#3b6ea5', '#5a8fc4', '#f0c674', '#e88a3c', '#c94f3a']
    bars = ax.bar(tier_summary['tier'].astype(str), tier_summary['success_rate'],
                  color=colors[:len(tier_summary)], edgecolor='white', linewidth=1.5)

    for bar, rate in zip(bars, tier_summary['success_rate']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{rate}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Lead Tier', fontsize=11)
    ax.set_ylabel('Conversion Rate (%)', fontsize=11)
    ax.set_title('Conversion Rate by Lead Tier', fontsize=13, pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance_df, save_path, top_n=15):
    """Horizontal bar chart of top feature coefficients."""
    df = importance_df.head(top_n).iloc[::-1]

    colors = ['#4e9b6b' if c > 0 else '#d4574e' for c in df['coefficient']]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df['feature'], df['coefficient'], color=colors, edgecolor='white', linewidth=1)

    ax.axvline(x=0, color='#333333', linewidth=0.8)
    ax.set_xlabel('Coefficient (positive = more likely to convert)', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Coefficients', fontsize=13, pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def generate_all_plots(scored_df, y_test, y_pred_proba, tier_summary, importance_df, output_dir):
    """Generate all four charts and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    plot_score_distribution(scored_df, os.path.join(output_dir, 'score_distribution.png'))
    plot_roc_curve(y_test, y_pred_proba, os.path.join(output_dir, 'roc_curve.png'))
    plot_success_by_tier(tier_summary, os.path.join(output_dir, 'success_by_tier.png'))
    plot_feature_importance(importance_df, os.path.join(output_dir, 'feature_importance.png'))

    return [
        'score_distribution.png',
        'roc_curve.png',
        'success_by_tier.png',
        'feature_importance.png'
    ]
