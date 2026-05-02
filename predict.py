"""
Inference Script — Sentiment Prediction on New Reviews
=======================================================
Load a trained model and predict sentiment on new review text.

Usage:
    # Predict a single review
    python predict.py --text "This product is amazing, works perfectly!"

    # Predict from a file (one review per line)
    python predict.py --file reviews.txt

    # Specify model and threshold
    python predict.py --text "Broke after one week" --model bilstm --threshold 0.4
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np


# ── Preprocessing (mirrors notebook preprocessing) ────────────────────────

TAG_RE = re.compile(r'<[^>]+>')


def preprocess_text(sentence: str, stop_words: set) -> str:
    """Clean a review string for model input — mirrors notebook pipeline."""
    if not isinstance(sentence, str) or not sentence.strip():
        return ''
    sentence = sentence.lower()
    sentence = TAG_RE.sub('', sentence)
    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    tokens   = [t for t in sentence.split() if t not in stop_words and len(t) > 1]
    return ' '.join(tokens).strip()


# ── Model loading ─────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str = 'bilstm',
    models_dir: str = 'models',
):
    """
    Load a saved Keras model and its tokenizer.

    Expects:
        models/{model_name}_best.keras    — Keras SavedModel weights
        models/{model_name}_tokenizer.pkl — pickled Keras Tokenizer

    Args:
        model_name: One of 'lstm', 'conv1d', 'bilstm'
        models_dir: Directory containing saved model artefacts

    Returns:
        Tuple of (model, tokenizer)
    """
    import tensorflow as tf

    models_path = Path(models_dir)
    model_path  = models_path / f'{model_name}_best.keras'
    tok_path    = models_path / f'{model_name}_tokenizer.pkl'

    if not model_path.exists():
        raise FileNotFoundError(
            f'Model not found: {model_path}\n'
            f'Run the notebook first to train and save the model.'
        )
    if not tok_path.exists():
        raise FileNotFoundError(
            f'Tokenizer not found: {tok_path}\n'
            f'Run the notebook first to save the tokenizer.'
        )

    model     = tf.keras.models.load_model(str(model_path))
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)

    print(f'Loaded model     : {model_path}')
    print(f'Loaded tokenizer : {tok_path}')
    print(f'Model parameters : {model.count_params():,}')
    return model, tokenizer


# ── Prediction pipeline ───────────────────────────────────────────────────

def predict_sentiment(
    texts:      list[str],
    model,
    tokenizer,
    stop_words: set,
    maxlen:     int   = 100,
    threshold:  float = 0.5,
) -> list[dict]:
    """
    Predict sentiment for a list of review strings.

    Pipeline:
        raw text → preprocess → tokenise → pad → model.predict → threshold

    Args:
        texts:      List of raw review strings
        model:      Loaded Keras model (sigmoid output)
        tokenizer:  Fitted Keras Tokenizer
        stop_words: NLTK English stopword set
        maxlen:     Sequence length (must match training configuration)
        threshold:  Decision boundary (default 0.5;
                    lower = more sensitive to negative sentiment)

    Returns:
        List of dicts with keys:
            text, cleaned, sentiment, confidence, probability
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    results = []

    for text in texts:
        # Step 1: Clean text
        cleaned = preprocess_text(text, stop_words)
        if not cleaned:
            results.append({
                'text':       text,
                'cleaned':    '',
                'sentiment':  'UNKNOWN',
                'confidence': 0.0,
                'probability': 0.5,
                'error':      'Text became empty after preprocessing',
            })
            continue

        # Step 2: Tokenise and pad
        seq     = tokenizer.texts_to_sequences([cleaned])
        padded  = pad_sequences(seq, maxlen=maxlen, padding='post', truncating='post')

        # Step 3: Predict
        prob    = float(model.predict(padded, verbose=0).ravel()[0])
        is_pos  = prob >= threshold
        conf    = prob if is_pos else (1.0 - prob)

        results.append({
            'text':        text,
            'cleaned':     cleaned,
            'sentiment':   'POSITIVE ✅' if is_pos else 'NEGATIVE ❌',
            'confidence':  round(conf * 100, 1),
            'probability': round(prob, 4),
        })

    return results


def print_results(results: list[dict], threshold: float) -> None:
    """Print prediction results in a readable format."""
    sep = '─' * 70
    print(f'\n{"="*70}')
    print(f'  SENTIMENT PREDICTIONS  (threshold={threshold})')
    print(f'{"="*70}')

    for i, res in enumerate(results, start=1):
        print(f'\n[Review {i}]')
        print(f'  Input    : {res["text"][:100]}{"..." if len(res["text"]) > 100 else ""}')
        if res.get('error'):
            print(f'  Error    : {res["error"]}')
            continue
        print(f'  Cleaned  : {res["cleaned"][:80]}{"..." if len(res["cleaned"]) > 80 else ""}')
        print(f'  Sentiment: {res["sentiment"]}')
        print(f'  Confidence: {res["confidence"]}%')
        print(f'  P(Positive): {res["probability"]}')
        print(f'  {sep}')

    pos_count = sum(1 for r in results if 'POSITIVE' in r.get('sentiment', ''))
    neg_count = sum(1 for r in results if 'NEGATIVE' in r.get('sentiment', ''))
    print(f'\nSummary: {pos_count} Positive | {neg_count} Negative | {len(results)} total')
    print('=' * 70)


# ── CLI entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Predict sentiment on Amazon appliance reviews',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--text', '-t', type=str, default=None,
        help='Single review text to classify',
    )
    parser.add_argument(
        '--file', '-f', type=str, default=None,
        help='Path to text file with one review per line',
    )
    parser.add_argument(
        '--model', '-m', type=str, default='bilstm',
        choices=['lstm', 'conv1d', 'bilstm'],
        help='Which trained model to use (default: bilstm)',
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Decision threshold — lower = more sensitive to negatives (default: 0.5)',
    )
    parser.add_argument(
        '--maxlen', type=int, default=100,
        help='Sequence length used during training (default: 100)',
    )
    parser.add_argument(
        '--models-dir', type=str, default='models',
        help='Directory containing saved model artefacts (default: models/)',
    )

    args = parser.parse_args()

    # Validate input
    if args.text is None and args.file is None:
        parser.error('Provide either --text or --file')

    # Collect texts
    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f'Error: File not found: {args.file}', file=sys.stderr)
            sys.exit(1)
        with open(file_path, encoding='utf-8') as f:
            file_texts = [line.strip() for line in f if line.strip()]
        texts.extend(file_texts)
        print(f'Loaded {len(file_texts)} reviews from {args.file}')

    if not texts:
        print('Error: No reviews to classify', file=sys.stderr)
        sys.exit(1)

    # Load NLTK stopwords
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords as nltk_sw
    stop_words = set(nltk_sw.words('english'))

    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model, args.models_dir)
    except FileNotFoundError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    # Predict
    results = predict_sentiment(
        texts      = texts,
        model      = model,
        tokenizer  = tokenizer,
        stop_words = stop_words,
        maxlen     = args.maxlen,
        threshold  = args.threshold,
    )

    print_results(results, args.threshold)


if __name__ == '__main__':
    main()
