"""
Evaluation Metrics Module — Sentiment Classification
=====================================================
Comprehensive evaluation utilities for binary sentiment classifiers.

Beyond the baseline accuracy-only evaluation, this module provides:
    - Precision, Recall, F1, ROC-AUC, PR-AUC
    - Per-class breakdown via classification_report
    - Threshold optimisation (useful when class imbalance exists)
    - Model comparison scorecard across LSTM / Conv1D / BiLSTM
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve,
)

CLASS_NAMES = ['Negative', 'Positive']


def evaluate_predictions(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    model_name: str   = 'Model',
    threshold:  float = 0.50,
    verbose:    bool  = True,
) -> Dict:
    """
    Compute comprehensive binary classification metrics.

    Args:
        y_true:     True binary labels (0=Negative, 1=Positive)
        y_prob:     Predicted probabilities for class 1 (Positive)
        model_name: Label for reporting
        threshold:  Decision boundary (default 0.5)
        verbose:    Print results to stdout

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc, pr_auc,
              y_pred, report, confusion_matrix
    """
    y_pred = (y_prob >= threshold).astype(int)

    results = {
        'model_name':       model_name,
        'threshold':        threshold,
        'accuracy':         float(accuracy_score(y_true, y_pred)),
        'precision':        float(precision_score(y_true, y_pred,  zero_division=0)),
        'recall':           float(recall_score(y_true, y_pred,     zero_division=0)),
        'f1':               float(f1_score(y_true, y_pred,         zero_division=0)),
        'roc_auc':          float(roc_auc_score(y_true, y_prob)),
        'pr_auc':           float(average_precision_score(y_true, y_prob)),
        'y_pred':           y_pred,
        'y_prob':           y_prob,
        'report':           classification_report(
                                y_true, y_pred,
                                target_names=CLASS_NAMES,
                                zero_division=0,
                            ),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }

    if verbose:
        print(f'\n[{model_name}]')
        print(f'  Accuracy  : {results["accuracy"]:.4f}')
        print(f'  Precision : {results["precision"]:.4f}')
        print(f'  Recall    : {results["recall"]:.4f}')
        print(f'  F1 Score  : {results["f1"]:.4f}')
        print(f'  ROC-AUC   : {results["roc_auc"]:.4f}')
        print(f'  PR-AUC    : {results["pr_auc"]:.4f}')

    return results


def find_best_threshold(
    y_true:        np.ndarray,
    y_prob:        np.ndarray,
    optimise_for:  str   = 'f1',
    n_thresholds:  int   = 100,
) -> Tuple[float, Dict]:
    """
    Find the decision threshold that maximises a chosen metric.

    Useful when there is class imbalance (many more positive than
    negative reviews) and 0.5 is not the optimal threshold.

    Args:
        y_true:       True binary labels
        y_prob:       Predicted probabilities
        optimise_for: Metric to maximise — 'f1', 'accuracy', 'recall'
        n_thresholds: Number of threshold candidates to evaluate

    Returns:
        Tuple of (optimal_threshold, metrics_at_optimal_threshold)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_score = -1.0
    best_thresh = 0.5

    metric_fns = {
        'f1':       lambda yt, yp: f1_score(yt, yp, zero_division=0),
        'accuracy': lambda yt, yp: accuracy_score(yt, yp),
        'recall':   lambda yt, yp: recall_score(yt, yp, zero_division=0),
    }
    fn = metric_fns.get(optimise_for, metric_fns['f1'])

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score  = fn(y_true, y_pred)
        if score > best_score:
            best_score  = score
            best_thresh = t

    results = evaluate_predictions(
        y_true, y_prob,
        model_name = f'Model @ {best_thresh:.3f}',
        threshold  = best_thresh,
        verbose    = False,
    )
    print(f'Best threshold ({optimise_for}): {best_thresh:.3f} '
          f'→ {optimise_for}={best_score:.4f}')
    return best_thresh, results


def compare_models(results_list: List[Dict]) -> Tuple[pd.DataFrame, str]:
    """
    Build a side-by-side scorecard comparing all evaluated models.

    Args:
        results_list: List of dicts from evaluate_predictions()

    Returns:
        Tuple of (scorecard_DataFrame, winning_model_name)
    """
    metric_keys = [
        ('Accuracy',  'accuracy'),
        ('Precision', 'precision'),
        ('Recall',    'recall'),
        ('F1 Score',  'f1'),
        ('ROC-AUC',   'roc_auc'),
        ('PR-AUC',    'pr_auc'),
    ]

    rows = []
    for label, key in metric_keys:
        row = {'Metric': label}
        scores = {}
        for res in results_list:
            val           = round(float(res.get(key, 0)), 4)
            row[res['model_name']] = val
            scores[res['model_name']] = val
        row['Winner'] = max(scores, key=scores.get)
        rows.append(row)

    df = pd.DataFrame(rows)

    win_counts   = {res['model_name']: (df['Winner'] == res['model_name']).sum()
                    for res in results_list}
    best_model   = max(win_counts, key=win_counts.get)

    print('\n' + '=' * 62)
    print('  MODEL SCORECARD — Sentiment Classification')
    print('=' * 62)
    print(df.to_string(index=False))
    print()
    for name, wins in win_counts.items():
        print(f'  {name}: {wins}/{len(df)} metrics won')
    print(f'\n  Overall winner: {best_model}')
    print('=' * 62)

    return df, best_model
