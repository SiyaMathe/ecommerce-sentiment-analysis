"""
Visualisation Module — Sentiment Analysis Plots
================================================
All plotting functions for the Amazon Appliance Review
sentiment classification project.

Functions:
    plot_class_distribution     — bar + pie chart of sentiment labels
    plot_review_length          — histogram of review word counts
    plot_augmentation_examples  — (not applicable for NLP, replaced by
                                   word cloud / top-token visualisation)
    plot_top_tokens             — most frequent words per sentiment class
    plot_training_history       — loss + accuracy curves for all models
    plot_confusion_matrices     — heatmaps for all three models
    plot_roc_curves             — ROC curves with AUC for all models
    plot_pr_curves              — Precision-Recall curves
    plot_embedding_coverage     — GloVe coverage bar chart
    plot_model_scorecard        — horizontal bar scorecard comparison
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

FIG_DIR = Path('reports/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    'positive':  '#1D9E75',
    'negative':  '#E24B4A',
    'lstm':      '#378ADD',
    'conv1d':    '#9B59B6',
    'bilstm':    '#F39C12',
    'neutral':   '#7F8C8D',
}
CLASS_NAMES = ['Negative', 'Positive']


# ── Dataset visualisations ────────────────────────────────────────────────

def plot_class_distribution(
    sentiment_counts: Dict[int, int],
    total:            int,
    save:             bool = True,
) -> None:
    """
    Plot the binary sentiment class distribution.

    Args:
        sentiment_counts: Dict {0: n_negative, 1: n_positive}
        total:            Total number of reviews
        save:             Save to reports/figures/
    """
    labels = ['Negative (≤ 3★)', 'Positive (> 3★)']
    values = [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)]
    colors = [PALETTE['negative'], PALETTE['positive']]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart
    bars = axes[0].bar(labels, values, color=colors, alpha=0.85,
                        edgecolor='white', width=0.5)
    for bar, val in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f'{val:,}\n({val / total * 100:.1f}%)',
            ha='center', fontweight='bold', fontsize=10,
        )
    axes[0].set_ylabel('Review count')
    axes[0].set_ylim(0, max(values) * 1.25)
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
    )
    axes[0].set_title('Binary Sentiment Distribution', fontweight='bold')

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        values, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    )
    for at in autotexts:
        at.set_fontweight('bold')
    axes[1].set_title('Proportion', fontweight='bold')

    plt.suptitle(
        f'Amazon Appliances Reviews — Sentiment Distribution (n={total:,})',
        fontweight='bold', fontsize=14,
    )
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / '00_class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_review_length(
    texts:        List[str],
    maxlen_line:  int  = 100,
    sample_label: str  = '(full dataset)',
    save:         bool = True,
) -> None:
    """
    Plot the distribution of review word counts after preprocessing.

    Args:
        texts:        List of cleaned review strings
        maxlen_line:  The MAXLEN cutoff used for padding/truncation
        sample_label: Label for the chart subtitle
        save:         Save to reports/figures/
    """
    word_counts = [len(t.split()) for t in texts if t]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(word_counts, bins=80, color=PALETTE['lstm'], alpha=0.75, edgecolor='white')
    ax.axvline(x=maxlen_line, color='red', ls='--', lw=2,
               label=f'MAXLEN = {maxlen_line} words')
    ax.axvline(x=np.median(word_counts), color='orange', ls='--', lw=2,
               label=f'Median = {np.median(word_counts):.0f} words')
    ax.set_xlabel('Word count (after preprocessing)', fontsize=12)
    ax.set_ylabel('Number of reviews', fontsize=12)
    ax.set_title(
        f'Review Length Distribution {sample_label}\n'
        f'Reviews beyond MAXLEN are truncated; shorter reviews are zero-padded',
        fontweight='bold',
    )
    ax.legend()
    pct_covered = sum(1 for w in word_counts if w <= maxlen_line) / len(word_counts) * 100
    ax.text(
        0.98, 0.95,
        f'{pct_covered:.1f}% of reviews\nfit within MAXLEN={maxlen_line}',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        fontsize=10,
    )
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / '01_review_length.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'Word count stats:')
    print(f'  Mean   : {np.mean(word_counts):.1f}')
    print(f'  Median : {np.median(word_counts):.1f}')
    print(f'  Max    : {max(word_counts)}')
    print(f'  % within MAXLEN={maxlen_line}: {pct_covered:.1f}%')


def plot_top_tokens(
    texts:           List[str],
    labels:          np.ndarray,
    top_n:           int  = 20,
    save:            bool = True,
) -> None:
    """
    Plot the most frequent tokens in positive vs negative reviews.

    Provides a quick sanity check that preprocessing removed stopwords
    and that the most predictive words differ between classes.

    Args:
        texts:   List of cleaned review strings
        labels:  Binary label array (0=Negative, 1=Positive)
        top_n:   Number of top tokens to display per class
        save:    Save to reports/figures/
    """
    pos_tokens = Counter()
    neg_tokens = Counter()

    for text, label in zip(texts, labels):
        tokens = text.split()
        if label == 1:
            pos_tokens.update(tokens)
        else:
            neg_tokens.update(tokens)

    pos_common = pos_tokens.most_common(top_n)
    neg_common = neg_tokens.most_common(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, common, title, color in [
        (axes[0], pos_common, 'Top Tokens — POSITIVE reviews', PALETTE['positive']),
        (axes[1], neg_common, 'Top Tokens — NEGATIVE reviews', PALETTE['negative']),
    ]:
        words, counts = zip(*common)
        y_pos = range(len(words))
        ax.barh(y_pos, counts, color=color, alpha=0.80, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')
        ax.set_title(title, fontweight='bold')
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
        )

    plt.suptitle(
        f'Top {top_n} Tokens per Sentiment Class (after stopword removal)',
        fontweight='bold', fontsize=14,
    )
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / '02_top_tokens.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_embedding_coverage(
    vocab_size:    int,
    words_found:   int,
    words_missing: int,
    save:          bool = True,
) -> None:
    """
    Plot GloVe embedding coverage of the review vocabulary.

    Args:
        vocab_size:    Total vocabulary size
        words_found:   Words with a GloVe vector
        words_missing: Words not in GloVe (zero-initialised)
        save:          Save to reports/figures/
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['Words with\nGloVe vector', 'Words without\nGloVe vector\n(zero-init)']
    values     = [words_found, words_missing]
    colors     = [PALETTE['positive'], PALETTE['neutral']]
    coverage   = words_found / vocab_size * 100

    bars = ax.bar(categories, values, color=colors, alpha=0.85, edgecolor='white', width=0.4)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f'{val:,}\n({val / vocab_size * 100:.1f}%)',
            ha='center', fontweight='bold', fontsize=11,
        )
    ax.set_ylabel('Word count')
    ax.set_title(
        f'GloVe Embedding Coverage\n'
        f'Vocabulary size: {vocab_size:,} | Coverage: {coverage:.1f}%',
        fontweight='bold',
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
    )
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / '03_embedding_coverage.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Training history ───────────────────────────────────────────────────────

def plot_training_history(
    histories:   Dict[str, object],
    save:        bool = True,
) -> None:
    """
    Plot training and validation loss + accuracy for all models.

    FIX: Original code accessed history['acc'] — deprecated in TF2.
    This function uses history['accuracy'] (correct TF2 key) with a
    safe fallback to 'acc' for older TF versions.

    Note on validation > training metrics:
        It is NORMAL for val_accuracy to exceed train_accuracy in
        Keras. Dropout and BatchNorm are active during training
        (adding noise / scaling) but DISABLED during validation.
        This gives the validation pass a slight accuracy advantage.
        It is NOT a sign of data leakage.

    Args:
        histories: Dict mapping model_name → Keras History object
        save:      Save to reports/figures/
    """
    model_colors = {
        'LSTM':   PALETTE['lstm'],
        'Conv1D': PALETTE['conv1d'],
        'BiLSTM': PALETTE['bilstm'],
    }
    n_models = len(histories)
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 5 * n_models))
    if n_models == 1:
        axes = [axes]

    for row, (model_name, history) in enumerate(histories.items()):
        color  = model_colors.get(model_name, PALETTE['neutral'])
        hist   = history.history
        epochs = range(1, len(hist['loss']) + 1)

        # FIX: Use 'accuracy' not 'acc' — safe fallback for older TF
        acc_key     = 'accuracy'     if 'accuracy'     in hist else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in hist else 'val_acc'

        # Loss
        axes[row][0].plot(epochs, hist['loss'],     color=color,   lw=2.5, label='Train')
        axes[row][0].plot(epochs, hist['val_loss'], color='black', lw=2.5,
                          ls='--', label='Validation')
        axes[row][0].set_title(f'{model_name} — Loss', fontweight='bold')
        axes[row][0].set_xlabel('Epoch')
        axes[row][0].set_ylabel('Binary Cross-Entropy')
        axes[row][0].legend()

        # Accuracy
        axes[row][1].plot(epochs, hist[acc_key],     color=color,   lw=2.5, label='Train')
        axes[row][1].plot(epochs, hist[val_acc_key], color='black', lw=2.5,
                          ls='--', label='Validation')
        axes[row][1].set_title(f'{model_name} — Accuracy', fontweight='bold')
        axes[row][1].set_xlabel('Epoch')
        axes[row][1].set_ylabel('Accuracy')
        axes[row][1].set_ylim([0, 1.05])
        axes[row][1].legend()

    plt.suptitle(
        'Training History — LSTM vs Conv1D vs Bidirectional LSTM\n'
        '(val metrics can exceed train due to Dropout/BN being disabled at eval — expected in TF2)',
        fontweight='bold', fontsize=13,
    )
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / '04_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Evaluation visualisations ─────────────────────────────────────────────

def plot_confusion_matrices(
    results_list: List[Dict],
    y_true:       np.ndarray,
    save:         bool = True,
) -> None:
    """
    Plot confusion matrices for one or more models side by side.

    Args:
        results_list: List of dicts from evaluate_predictions()
        y_true:       True binary labels
        save:         Save to reports/figures/
    """
    n   = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        cm  = res['confusion_matrix']
        pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        ann = np.array(
            [[f'{cm[i, j]:,}\n({pct[i, j]:.1f}%)' for j in range(2)] for i in range(2)]
        )
        sns.heatmap(
            pct, annot=ann, fmt='', cmap='Blues',
            xticklabels=['Pred: Negative', 'Pred: Positive'],
            yticklabels=['True: Negative', 'True: Positive'],
            ax=ax, cbar=True, linewidths=0.5,
        )
        ax.set_title(
            f'{res["model_name"]}\n'
            f'Acc={res["accuracy"]:.4f} | '
            f'F1={res["f1"]:.4f} | '
            f'AUC={res["roc_auc"]:.4f}',
            fontweight='bold',
        )

    plt.suptitle('Confusion Matrices — Test Set', fontweight='bold', fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / '05_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_and_pr_curves(
    results_list: List[Dict],
    y_true:       np.ndarray,
    save:         bool = True,
) -> None:
    """
    Plot ROC curves and Precision-Recall curves side by side.

    The PR curve is particularly informative when the dataset is
    imbalanced (more positive than negative reviews) because it
    focuses on the minority class performance.

    Args:
        results_list: List of dicts from evaluate_predictions()
        y_true:       True binary labels
        save:         Save to reports/figures/
    """
    model_colors = [PALETTE['lstm'], PALETTE['conv1d'], PALETTE['bilstm'],
                    PALETTE['neutral']]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for res, color in zip(results_list, model_colors):
        name   = res['model_name']
        y_prob = res['y_prob']

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score   = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, color=color, lw=2.5,
                     label=f'{name} (AUC={auc_score:.3f})')

        # PR curve
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc       = average_precision_score(y_true, y_prob)
        axes[1].plot(rec, prec, color=color, lw=2.5,
                     label=f'{name} (PR-AUC={pr_auc:.3f})')

    # ROC panel
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curves', fontweight='bold', fontsize=13)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.02])

    # PR panel
    baseline_pr = y_true.mean()
    axes[1].axhline(y=baseline_pr, color='k', ls='--', lw=1,
                    label=f'Random baseline ({baseline_pr:.2f})')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curves\n'
                       '(More informative than ROC when classes are imbalanced)',
                       fontweight='bold', fontsize=13)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])

    plt.suptitle(
        'Classification Curves — LSTM vs Conv1D vs BiLSTM',
        fontweight='bold', fontsize=14,
    )
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / '06_roc_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_scorecard(
    scorecard_df: pd.DataFrame,
    winner:       str,
    save:         bool = True,
) -> None:
    """
    Plot a horizontal bar chart comparing all models across all metrics.

    Args:
        scorecard_df: DataFrame from compare_models() with columns:
                      Metric, LSTM, Conv1D, BiLSTM, Winner
        winner:       Name of the overall winning model
        save:         Save to reports/figures/
    """
    model_cols = [c for c in scorecard_df.columns if c not in ['Metric', 'Winner']]
    n_metrics  = len(scorecard_df)
    n_models   = len(model_cols)
    bar_height = 0.25
    y_pos      = np.arange(n_metrics)

    model_colors = [PALETTE['lstm'], PALETTE['conv1d'], PALETTE['bilstm']]

    fig, ax = plt.subplots(figsize=(12, max(6, n_metrics * 0.9)))

    for i, (model, color) in enumerate(zip(model_cols, model_colors)):
        values = scorecard_df[model].values.astype(float)
        offset = (i - n_models / 2 + 0.5) * bar_height
        bars   = ax.barh(y_pos + offset, values, bar_height,
                          color=color, alpha=0.85, label=model, edgecolor='white')
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}',
                va='center', fontsize=8,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(scorecard_df['Metric'].values, fontsize=11)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_xlim([0, 1.12])
    ax.axvline(x=1.0, color='grey', ls='--', lw=1, alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title(
        f'Model Scorecard — All Metrics\nOverall winner: {winner}',
        fontweight='bold', fontsize=13,
    )
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / '07_model_scorecard.png', dpi=150, bbox_inches='tight')
    plt.show()
