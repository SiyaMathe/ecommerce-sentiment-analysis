"""
Model Persistence Utilities — Save and Load Pipeline Artefacts
==============================================================
Handles saving and loading of trained models, tokenizers,
and preprocessing metadata so predict.py can run without
re-training.

Called from the notebook after training is complete:

    from src.utils.persistence import save_artefacts, load_artefacts
    save_artefacts(models, tokenizer, config)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def save_artefacts(
    models:       Dict,
    tokenizer,
    maxlen:       int,
    embed_dim:    int,
    vocab_size:   int,
    models_dir:   str = 'models',
) -> None:
    """
    Save trained models, tokenizer, and config to disk.

    Saves:
        models/{name}_best.keras      — Keras SavedModel (already saved by ModelCheckpoint)
        models/{name}_tokenizer.pkl   — pickled Keras Tokenizer
        models/pipeline_config.json   — preprocessing configuration

    Args:
        models:     Dict mapping model_name → trained Keras model
        tokenizer:  Fitted Keras Tokenizer (trained on training data only)
        maxlen:     Sequence padding length used during training
        embed_dim:  GloVe embedding dimensionality
        vocab_size: Tokenizer vocabulary size
        models_dir: Directory to save artefacts
    """
    save_path = Path(models_dir)
    save_path.mkdir(exist_ok=True)

    # Save tokenizer — same for all models (fitted on same training data)
    for name in models:
        tok_path = save_path / f'{name}_tokenizer.pkl'
        with open(tok_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f'Tokenizer saved : {tok_path}')

    # Save pipeline config so predict.py knows what params to use
    config = {
        'maxlen':     maxlen,
        'embed_dim':  embed_dim,
        'vocab_size': vocab_size,
        'models':     list(models.keys()),
    }
    config_path = save_path / 'pipeline_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Config saved    : {config_path}')
    print(f'\nAll artefacts saved to {models_dir}/')
    print(f'Run: python predict.py --text "Your review here" --model bilstm')
