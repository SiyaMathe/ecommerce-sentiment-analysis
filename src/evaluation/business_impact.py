"""
Business Impact Module — Sentiment Analysis on Amazon Appliance Reviews
========================================================================
Translates classifier performance into measurable business value for
an e-commerce or manufacturing context.

Business scenarios covered:
    1. E-commerce product reputation management
       Negative reviews that go undetected damage brand reputation and
       suppress conversion rates. Catching them early enables intervention.

    2. Manufacturing IoT feedback loop
       In a production context (e.g. Mercedes-Benz SA), customer reviews
       of parts and appliances feed back into quality improvement cycles.
       Missing negative signals delays corrective action.

    3. Customer support triage
       Automatically flagging negative reviews for escalation reduces
       mean time to resolution and improves customer satisfaction scores.

Module structure:
    ReviewImpactConfig     — dataclass of ZAR cost/revenue assumptions
    SentimentImpactResult  — dataclass returned by calculate_sentiment_impact()
    calculate_sentiment_impact — convert confusion matrix to ZAR values
    compare_to_baseline    — CNN vs rule-based / human baseline
    simulate_annual_volume — scale daily results to annual business impact
    plot_impact            — three-panel visualisation
    plot_threshold_value   — net value vs threshold curve
    generate_report        — formatted business case to stdout
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

FIG_DIR = Path('reports/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    'positive':     '#1D9E75',
    'negative':     '#E24B4A',
    'lstm':         '#378ADD',
    'conv1d':       '#9B59B6',
    'bilstm':       '#F39C12',
    'neutral':      '#7F8C8D',
}


# ── Configuration dataclass ───────────────────────────────────────────────

@dataclass
class ReviewImpactConfig:
    """
    Business cost and revenue assumptions for review sentiment analysis.

    All monetary values are in South African Rand (ZAR) and reflect
    a realistic e-commerce / manufacturing feedback scenario.

    Scenario:
        A quality manager monitors customer product reviews daily.
        The sentiment model classifies each review as Positive or Negative.
        Correctly identified negatives are escalated for action.
        Missed negatives (False Negatives) delay quality intervention.
        False positives waste analyst time on benign reviews.

    Attributes:
        value_true_positive:     Revenue from correctly catching a negative review
                                 (enables intervention — prevent churn, fix defect)
        value_true_negative:     Value from correctly clearing a positive review
                                 (confirms product quality, no action needed)
        cost_false_negative:     Cost of missing a negative review
                                 (delayed quality action, brand damage, churn)
        cost_false_positive:     Cost of incorrectly flagging a positive as negative
                                 (wasted analyst escalation time)
        churn_cost_per_customer: Average customer lifetime value lost per churn
        churn_probability:       Fraction of missed negatives that lead to churn
        analyst_hourly_rate_zar: Analyst cost per hour for false-alarm review
        minutes_per_escalation:  Time to process one escalated review
        daily_review_volume:     Daily review volume for annual scaling
        working_days_per_year:   Working days for annual projection
        baseline_accuracy:       Accuracy of baseline rule-based classifier
        baseline_recall:         Recall of baseline rule-based classifier
    """
    # Revenue (ZAR)
    value_true_positive:      float = 850.0     # caught negative — enables intervention
    value_true_negative:      float = 120.0     # cleared positive — confirms quality

    # Costs (ZAR)
    cost_false_negative:      float = 4_200.0   # missed negative — delayed action + churn risk
    cost_false_positive:      float = 380.0     # false alarm — wasted analyst time

    # Churn modelling
    churn_cost_per_customer:  float = 8_500.0   # customer LTV lost
    churn_probability:        float = 0.08      # 8% of missed negatives lead to churn

    # Operational costs
    analyst_hourly_rate_zar:  float = 450.0
    minutes_per_escalation:   float = 12.0

    # Volume for annual projection
    daily_review_volume:      int   = 2_000
    working_days_per_year:    int   = 250

    # Baseline comparison
    baseline_accuracy:        float = 0.72
    baseline_recall:          float = 0.65      # simple keyword rules


@dataclass
class SentimentImpactResult:
    """Container for sentiment business impact calculation results."""
    model_name:   str
    tn:           int
    fp:           int
    fn:           int
    tp:           int
    value_tp:     float   # Revenue from caught negatives
    value_tn:     float   # Revenue from cleared positives
    cost_fp:      float   # Cost of false alarms
    cost_fn:      float   # Cost of missed negatives
    churn_cost:   float   # Expected churn cost from missed negatives
    net_value:    float   # Total net value
    accuracy:     float
    recall:       float
    precision:    float

    @property
    def breakdown(self) -> Dict[str, float]:
        """Impact by confusion matrix cell."""
        return {
            f'Caught negative (TP={self.tp:,})':  self.value_tp,
            f'Cleared positive (TN={self.tn:,})': self.value_tn,
            f'False alarm (FP={self.fp:,})':       self.cost_fp,
            f'Missed negative (FN={self.fn:,})':   self.cost_fn,
            f'Churn risk (8% of FN)':              self.churn_cost,
        }


# ── Core calculation ───────────────────────────────────────────────────────

def calculate_sentiment_impact(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str                      = 'Model',
    config:     Optional[ReviewImpactConfig] = None,
) -> SentimentImpactResult:
    """
    Compute ZAR business impact from sentiment classifier predictions.

    Translates each confusion matrix cell into a monetary value:
        TP (correctly caught negative review)  → intervention value
        TN (correctly cleared positive review) → operational efficiency
        FP (positive flagged as negative)      → wasted analyst time
        FN (negative missed)                   → delayed action + churn

    Args:
        y_true:     True binary labels  (0=Positive, 1=Negative)
        y_pred:     Predicted labels
        model_name: Label for reporting
        config:     ReviewImpactConfig. Uses defaults if None.

    Returns:
        SentimentImpactResult dataclass
    """
    if config is None:
        config = ReviewImpactConfig()

    cm           = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    value_tp   = int(tp) * config.value_true_positive
    value_tn   = int(tn) * config.value_true_negative
    cost_fp    = int(fp) * -config.cost_false_positive
    cost_fn    = int(fn) * -config.cost_false_negative
    churn_cost = int(fn) * config.churn_probability * -config.churn_cost_per_customer

    net = value_tp + value_tn + cost_fp + cost_fn + churn_cost

    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn)

    return SentimentImpactResult(
        model_name = model_name,
        tn         = int(tn), fp = int(fp),
        fn         = int(fn), tp = int(tp),
        value_tp   = value_tp,
        value_tn   = value_tn,
        cost_fp    = cost_fp,
        cost_fn    = cost_fn,
        churn_cost = churn_cost,
        net_value  = net,
        accuracy   = float(accuracy),
        recall     = float(recall),
        precision  = float(precision),
    )


def compare_to_baseline(
    model_result: SentimentImpactResult,
    y_true:       np.ndarray,
    config:       Optional[ReviewImpactConfig] = None,
) -> Tuple[SentimentImpactResult, float]:
    """
    Compare the deep learning model to a rule-based keyword baseline.

    The baseline represents a simple keyword-matching approach
    (e.g. 'good' → positive, 'bad'/'broken'/'terrible' → negative)
    typical of production systems before ML adoption.

    Args:
        model_result: SentimentImpactResult from the DL model
        y_true:       True binary labels for the same test set
        config:       ReviewImpactConfig with baseline parameters

    Returns:
        Tuple of (baseline_SentimentImpactResult, model_advantage_in_ZAR)
    """
    if config is None:
        config = ReviewImpactConfig()

    n_negative = int(y_true.sum())
    n_positive = int((1 - y_true).sum())

    # Construct synthetic baseline confusion matrix
    b_tp = round(n_negative * config.baseline_recall)
    b_fn = n_negative - b_tp
    b_tn = round(n_positive * config.baseline_accuracy)
    b_fp = n_positive - b_tn

    y_true_syn = np.array([0] * n_positive + [1] * n_negative, dtype=int)
    y_pred_syn = np.array(
        [0] * b_tn + [1] * b_fp +    # positive group
        [1] * b_tp + [0] * b_fn,     # negative group
        dtype=int,
    )

    baseline_result = calculate_sentiment_impact(
        y_true_syn, y_pred_syn, 'Keyword Baseline', config
    )
    advantage = model_result.net_value - baseline_result.net_value
    return baseline_result, advantage


def simulate_annual_volume(
    model_result:    SentimentImpactResult,
    test_set_size:   int,
    config:          Optional[ReviewImpactConfig] = None,
) -> Dict[str, float]:
    """
    Scale test-set impact to annual production review volume.

    Args:
        model_result:  SentimentImpactResult from the classifier
        test_set_size: Number of reviews in the test set
        config:        ReviewImpactConfig with volume parameters

    Returns:
        Dict of annualised financial and operational metrics
    """
    if config is None:
        config = ReviewImpactConfig()

    annual_volume = config.daily_review_volume * config.working_days_per_year
    scale         = annual_volume / test_set_size

    return {
        'annual_reviews_processed':   annual_volume,
        'annual_negatives_caught':    round(model_result.tp  * scale),
        'annual_negatives_missed':    round(model_result.fn  * scale),
        'annual_false_alarms':        round(model_result.fp  * scale),
        'annual_value_caught_zar':    model_result.value_tp  * scale,
        'annual_churn_prevented_zar': abs(model_result.churn_cost) * scale,
        'annual_false_alarm_cost_zar':abs(model_result.cost_fp)    * scale,
        'annual_missed_cost_zar':     abs(model_result.cost_fn)    * scale,
        'annual_net_value_zar':       model_result.net_value       * scale,
        'recall':                     model_result.recall,
        'precision':                  model_result.precision,
    }


# ── Reporting ──────────────────────────────────────────────────────────────

def generate_report(
    model_result:    SentimentImpactResult,
    baseline_result: SentimentImpactResult,
    advantage:       float,
    annual_stats:    Dict[str, float],
    config:          Optional[ReviewImpactConfig] = None,
) -> None:
    """
    Print a formatted business case report to the console.

    Args:
        model_result:    SentimentImpactResult for the DL model
        baseline_result: SentimentImpactResult for the keyword baseline
        advantage:       DL model net value minus baseline net value (ZAR)
        annual_stats:    Dict from simulate_annual_volume()
        config:          ReviewImpactConfig used for the analysis
    """
    if config is None:
        config = ReviewImpactConfig()

    sep = '=' * 68

    print(sep)
    print('  BUSINESS IMPACT REPORT — SENTIMENT ANALYSIS')
    print('  Amazon Appliance Reviews | Deep Learning vs Keyword Baseline')
    print(sep)
    print()
    print('  COST ASSUMPTIONS (ZAR)')
    print(f'  {"Value — caught negative (TP)":<44}: R{config.value_true_positive:>8,.0f}')
    print(f'  {"Value — cleared positive (TN)":<44}: R{config.value_true_negative:>8,.0f}')
    print(f'  {"Cost  — false alarm (FP)":<44}: R{config.cost_false_positive:>8,.0f}')
    print(f'  {"Cost  — missed negative (FN)":<44}: R{config.cost_false_negative:>8,.0f}')
    print(f'  {"Churn cost per customer":<44}: R{config.churn_cost_per_customer:>8,.0f}')
    print(f'  {"Churn probability per missed negative":<44}: {config.churn_probability:.0%}')
    print()
    print('  TEST SET RESULTS')
    print(f'  {"Metric":<30} {"Deep Learning":>15} {"Keyword Baseline":>17}')
    print(f'  {"-"*62}')
    for label, m_val, b_val in [
        ('Accuracy',               model_result.accuracy,  baseline_result.accuracy),
        ('Recall (negatives caught)', model_result.recall, baseline_result.recall),
        ('Precision',               model_result.precision, baseline_result.precision),
        ('True Positives (caught)', model_result.tp,        baseline_result.tp),
        ('False Negatives (missed)', model_result.fn,       baseline_result.fn),
        ('False Positives (alarms)', model_result.fp,       baseline_result.fp),
    ]:
        if isinstance(m_val, float):
            print(f'  {label:<30} {m_val:>15.4f} {b_val:>17.4f}')
        else:
            print(f'  {label:<30} {m_val:>15,} {b_val:>17,}')
    print()
    print('  FINANCIAL IMPACT (TEST SET)')
    labels_vals = [
        ('Caught negatives (TP)', model_result.value_tp,   baseline_result.value_tp),
        ('Cleared positives (TN)', model_result.value_tn,  baseline_result.value_tn),
        ('False alarms (FP)',       model_result.cost_fp,   baseline_result.cost_fp),
        ('Missed negatives (FN)',   model_result.cost_fn,   baseline_result.cost_fn),
        ('Churn risk (8% FN)',      model_result.churn_cost, baseline_result.churn_cost),
    ]
    for label, m_val, b_val in labels_vals:
        print(f'  {label:<44} R{m_val:>10,.0f}   R{b_val:>10,.0f}')
    print(f'  {"─" * 60}')
    print(f'  {"Net value":<44} R{model_result.net_value:>10,.0f}   R{baseline_result.net_value:>10,.0f}')
    print(f'  {"DL model advantage":<44} R{advantage:>10,.0f}')
    print()
    print(f'  ANNUAL PROJECTION ({config.daily_review_volume:,} reviews/day × {config.working_days_per_year} days)')
    for label, val in [
        ('Reviews processed per year',    annual_stats['annual_reviews_processed']),
        ('Negative reviews caught',        annual_stats['annual_negatives_caught']),
        ('Negative reviews missed',        annual_stats['annual_negatives_missed']),
        ('False alarms per year',          annual_stats['annual_false_alarms']),
    ]:
        print(f'  {label:<44}: {val:>12,.0f}')
    for label, val in [
        ('Annual value — caught negatives',   annual_stats['annual_value_caught_zar']),
        ('Annual churn prevention value',      annual_stats['annual_churn_prevented_zar']),
        ('Annual false alarm cost',            annual_stats['annual_false_alarm_cost_zar']),
        ('Annual missed negative cost',        annual_stats['annual_missed_cost_zar']),
        ('Annual NET value (DL model)',        annual_stats['annual_net_value_zar']),
    ]:
        print(f'  {label:<44}: R{val:>10,.0f}')
    print()
    print('  KEY TAKEAWAYS')
    extra_caught = model_result.tp - baseline_result.tp
    print(f'  1. Deep learning catches {extra_caught:,} more negatives than keyword baseline')
    print(f'  2. Recall {model_result.recall:.1%} vs baseline {baseline_result.recall:.1%} '
          f'(+{(model_result.recall - baseline_result.recall)*100:.1f} pp improvement)')
    print(f'  3. Annual churn prevention: R{annual_stats["annual_churn_prevented_zar"]/1e3:.0f}K')
    print(f'  4. Annual net value uplift vs baseline: '
          f'R{advantage * annual_stats["annual_reviews_processed"] / len(range(model_result.tp + model_result.tn + model_result.fp + model_result.fn)) / 1e3:.0f}K')
    print(sep)


# ── Visualisation ──────────────────────────────────────────────────────────

def plot_impact(
    model_result:    SentimentImpactResult,
    baseline_result: SentimentImpactResult,
    advantage:       float,
    annual_stats:    Dict[str, float],
    save:            bool = True,
) -> None:
    """
    Three-panel business impact visualisation.

    Panel 1 — Waterfall: per-outcome ZAR breakdown for the DL model
    Panel 2 — DL vs Baseline: net value comparison
    Panel 3 — Annual projection: scaled financial metrics

    Args:
        model_result:    SentimentImpactResult for the DL model
        baseline_result: SentimentImpactResult for the baseline
        advantage:       DL net minus baseline net (ZAR)
        annual_stats:    Dict from simulate_annual_volume()
        save:            Save to reports/figures/
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # ── Panel 1: Waterfall breakdown ──────────────────────────────────
    breakdown = model_result.breakdown
    labels    = [k.split(' (')[0] for k in breakdown]
    values    = list(breakdown.values())
    colors    = [PALETTE['positive'] if v >= 0 else PALETTE['negative'] for v in values]

    bars = axes[0].bar(range(len(labels)), values, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, values):
        offset = max(abs(v) for v in values) * 0.04
        y_pos  = bar.get_height() + offset if val >= 0 else bar.get_height() - offset * 3
        label_str = f'R{val/1e3:.1f}K' if abs(val) < 1e6 else f'R{val/1e6:.2f}M'
        axes[0].text(bar.get_x() + bar.get_width()/2, y_pos,
                     label_str, ha='center', fontweight='bold', fontsize=8)
    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].axhline(model_result.net_value, color='navy', ls='--', lw=1.5,
                    label=f'Net: R{model_result.net_value/1e3:.1f}K')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    axes[0].set_ylabel('ZAR value')
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'R{x/1e3:.0f}K')
    )
    axes[0].set_title(f'Cost-Benefit Breakdown\n{model_result.model_name}',
                      fontweight='bold')
    axes[0].legend(fontsize=9)

    # ── Panel 2: DL vs Baseline ────────────────────────────────────────
    comp_vals   = [model_result.net_value, baseline_result.net_value]
    comp_labels = [
        f'Deep Learning\n({model_result.model_name})\nRecall={model_result.recall:.1%}',
        f'Keyword Baseline\nRecall={baseline_result.recall:.1%}',
    ]
    comp_colors = [PALETTE['lstm'], PALETTE['neutral']]
    bars2 = axes[1].bar(comp_labels, comp_vals, color=comp_colors,
                         alpha=0.85, edgecolor='white', width=0.5)
    for bar, val in zip(bars2, comp_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                     f'R{val/1e3:.1f}K', ha='center', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('Net ZAR value (test set)')
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'R{x/1e3:.0f}K')
    )
    axes[1].set_title(f'DL Model vs Keyword Baseline\nAdvantage: R{advantage/1e3:.1f}K',
                      fontweight='bold')

    # ── Panel 3: Annual projection ─────────────────────────────────────
    ann_labels = ['Annual\nnet value', 'Caught\nnegatives', 'Churn\nprevented', 'False alarm\ncost']
    ann_values = [
        annual_stats['annual_net_value_zar'],
        annual_stats['annual_value_caught_zar'],
        annual_stats['annual_churn_prevented_zar'],
        -annual_stats['annual_false_alarm_cost_zar'],
    ]
    ann_colors = [
        PALETTE['positive'] if v >= 0 else PALETTE['negative']
        for v in ann_values
    ]
    bars3 = axes[2].bar(ann_labels, ann_values, color=ann_colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars3, ann_values):
        label_str = f'R{val/1e6:.1f}M' if abs(val) >= 1e6 else f'R{val/1e3:.0f}K'
        y_pos = bar.get_height() * (1.04 if val >= 0 else 0.90)
        axes[2].text(bar.get_x() + bar.get_width()/2, y_pos,
                     label_str, ha='center', fontweight='bold', fontsize=10)
    axes[2].axhline(0, color='black', lw=0.8)
    axes[2].set_ylabel('ZAR (annual)')
    axes[2].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'R{x/1e6:.1f}M' if abs(x) >= 1e6 else f'R{x/1e3:.0f}K')
    )
    axes[2].set_title(
        f'Annual Projection\n({annual_stats["annual_reviews_processed"]:,.0f} reviews/year)',
        fontweight='bold'
    )

    plt.suptitle(
        'Business Impact — Sentiment Analysis on Amazon Appliance Reviews\n'
        'ZAR values based on e-commerce / manufacturing feedback context',
        fontweight='bold', fontsize=13,
    )
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / '05_business_impact.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_threshold_value(
    y_true:     np.ndarray,
    y_prob:     np.ndarray,
    model_name: str                      = 'Model',
    config:     Optional[ReviewImpactConfig] = None,
    save:       bool = True,
) -> None:
    """
    Plot how net ZAR value changes as the decision threshold varies.

    The optimal threshold for sentiment analysis often differs from 0.5
    when reviews are imbalanced (more positive than negative).

    Args:
        y_true:     True binary labels
        y_prob:     Predicted probabilities
        model_name: Label for the chart
        config:     ReviewImpactConfig
        save:       Save to reports/figures/
    """
    if config is None:
        config = ReviewImpactConfig()

    thresholds  = np.linspace(0.05, 0.95, 91)
    net_values  = []
    recalls     = []
    precisions  = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        net = (
            int(tp) * config.value_true_positive +
            int(tn) * config.value_true_negative +
            int(fp) * -config.cost_false_positive +
            int(fn) * -config.cost_false_negative +
            int(fn) * config.churn_probability * -config.churn_cost_per_customer
        )
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        net_values.append(net)
        recalls.append(rec)
        precisions.append(prec)

    valid_len    = len(net_values)
    thresh_valid = thresholds[:valid_len]
    opt_idx      = int(np.argmax(net_values))
    opt_thresh   = float(thresh_valid[opt_idx])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(thresh_valid, net_values, color=PALETTE['lstm'], lw=2.5)
    axes[0].axvline(0.5,       color='grey',             ls='--', lw=1.5, label='Default (0.5)')
    axes[0].axvline(opt_thresh, color=PALETTE['positive'], ls='--', lw=2,
                    label=f'Optimal ({opt_thresh:.2f})')
    axes[0].set_xlabel('Decision threshold', fontsize=12)
    axes[0].set_ylabel('Net ZAR value', fontsize=12)
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'R{x/1e3:.0f}K')
    )
    axes[0].set_title('Net ZAR Value vs Decision Threshold\n'
                       '(optimal threshold ≠ 0.5 under class imbalance)',
                       fontweight='bold')
    axes[0].legend()

    axes[1].plot(thresh_valid, recalls,    color=PALETTE['positive'], lw=2.5, label='Recall')
    axes[1].plot(thresh_valid, precisions, color=PALETTE['negative'],  lw=2.5, label='Precision')
    axes[1].axvline(0.5,        color='grey',             ls='--', lw=1.5, label='Default (0.5)')
    axes[1].axvline(opt_thresh, color=PALETTE['lstm'],    ls='--', lw=2,
                    label=f'Optimal ({opt_thresh:.2f})')
    axes[1].set_xlabel('Decision threshold', fontsize=12)
    axes[1].set_ylabel('Score')
    axes[1].set_ylim([0, 1.05])
    axes[1].set_title('Recall vs Precision vs Threshold', fontweight='bold')
    axes[1].legend(fontsize=9)

    plt.suptitle(
        f'Threshold Sensitivity Analysis — {model_name}\n'
        f'Optimal threshold: {opt_thresh:.2f} | '
        f'Net value: R{net_values[opt_idx]/1e3:.1f}K | '
        f'Recall: {recalls[opt_idx]:.1%}',
        fontweight='bold', fontsize=13,
    )
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / '06_threshold_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'Optimal threshold : {opt_thresh:.3f}')
    print(f'At optimal — Recall: {recalls[opt_idx]:.4f} | '
          f'Precision: {precisions[opt_idx]:.4f} | '
          f'Net value: R{net_values[opt_idx]:,.0f}')
