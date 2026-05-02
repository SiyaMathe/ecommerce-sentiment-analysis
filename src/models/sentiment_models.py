import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from tensorflow.keras import layers

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────
MAXLEN      = 100 
EMBED_DIM   = 100 

def load_glove_embeddings(filepath: str, embedding_dim: int = 100) -> dict:
    """Load GloVe word vectors from a text file into a dictionary."""
    embeddings = {}
    glove_path = Path(filepath)

    if not glove_path.exists():
        return embeddings

    with open(glove_path, encoding='utf8') as f:
        for line in f:
            parts = line.split()
            word  = parts[0]
            try:
                vector = np.asarray(parts[1:], dtype='float32')
                if len(vector) == embedding_dim:
                    embeddings[word] = vector
            except ValueError:
                continue 
    return embeddings

def build_embedding_layer(vocab_size: int, embedding_matrix: np.ndarray = None, trainable: bool = False) -> layers.Embedding:
    """
    Build an embedding layer. If embedding_matrix is provided, it uses GloVe.
    Otherwise, it initializes with zeros (useful for CI/Testing).
    """
    if embedding_matrix is None:
        # Fallback for CI environments where the GloVe file isn't present
        weights = [np.zeros((vocab_size, EMBED_DIM))]
    else:
        weights = [embedding_matrix]

    return layers.Embedding(
        input_dim    = vocab_size,
        output_dim   = EMBED_DIM,
        weights      = weights,
        input_length = MAXLEN,
        trainable    = trainable,
        name         = 'glove_embedding',
    )

# ── Model Builders ────────────────────────────────────────────────────────

def build_lstm_model(vocab_size: int, embed_dim: int = 100, maxlen: int = 100) -> keras.Model:
    """LSTM for binary sentiment classification."""
    model = keras.Sequential([
        layers.Embedding(vocab_size, embed_dim, input_length=maxlen, name='embedding'),
        layers.LSTM(128, dropout=0.2, recurrent_dropout=0.1, name='lstm_1'),
        layers.Dropout(0.3, name='dropout_1'),
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.2, name='dropout_2'),
        layers.Dense(1, activation='sigmoid', name='output'),
    ], name='LSTM_Sentiment')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_conv1d_model(vocab_size: int, embed_dim: int = 100, maxlen: int = 100) -> keras.Model:
    """Conv1D for binary sentiment classification."""
    model = keras.Sequential([
        layers.Embedding(vocab_size, embed_dim, input_length=maxlen, name='embedding'),
        layers.Conv1D(filters=128, kernel_size=5, activation='relu', name='conv1d_1'),
        layers.GlobalMaxPooling1D(name='global_max_pool'),
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.3, name='dropout_1'),
        layers.Dense(1, activation='sigmoid', name='output'),
    ], name='Conv1D_Sentiment')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm_model(vocab_size: int, embed_dim: int = 100, maxlen: int = 100) -> keras.Model:
    """Bidirectional LSTM for binary sentiment classification."""
    model = keras.Sequential([
        layers.Embedding(vocab_size, embed_dim, input_length=maxlen, name='embedding'),
        layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.1), name='bilstm_1'),
        layers.Dropout(0.3, name='dropout_1'),
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.2, name='dropout_2'),
        layers.Dense(1, activation='sigmoid', name='output'),
    ], name='BiLSTM_Sentiment')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model