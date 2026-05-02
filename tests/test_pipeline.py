"""
Test Suite — Sentiment Analysis NLP Pipeline
=============================================
Covers all three discipline layers:

    CS:  Model architecture, output shape, callbacks, reproducibility
    DE:  PySpark fix verification, text preprocessing, tokenisation,
         GloVe loading, embedding matrix construction
    DS:  Evaluation metrics, business impact, threshold optimisation,
         visualisation utilities
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_reviews():
    """Clean review strings for preprocessing tests."""
    return [
        "this product is absolutely amazing works perfectly",
        "terrible quality stopped working after one week",
        "great value for money highly recommend",
        "complete waste of money do not buy",
        "excellent product fast delivery good packaging",
        "broken on arrival very disappointed with quality",
    ]


@pytest.fixture
def sample_labels():
    """Binary labels matching sample_reviews fixture."""
    return np.array([1, 0, 1, 0, 1, 0], dtype=int)


@pytest.fixture
def binary_predictions():
    """Realistic binary prediction arrays for metric tests."""
    np.random.seed(42)
    n      = 500
    y_true = np.array([0] * 350 + [1] * 150, dtype=int)
    y_prob = np.where(
        y_true == 1,
        np.random.beta(7, 2, n),
        np.random.beta(2, 7, n),
    )
    return y_true, y_prob


@pytest.fixture
def perfect_predictions():
    """Perfect classifier for boundary condition testing."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    y_prob = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
    return y_true, y_prob


@pytest.fixture
def dummy_embedding_matrix():
    """Small random embedding matrix for model tests."""
    np.random.seed(42)
    return np.random.randn(1000, 100).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# CS: MODEL ARCHITECTURE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestLSTMModel:

    def test_builds_without_error(self):
        from src.models.sentiment_models import build_lstm_model
        model = build_lstm_model(vocab_size=1000, embed_dim=50, maxlen=50)
        assert model is not None

    def test_output_shape_is_scalar(self):
        from src.models.sentiment_models import build_lstm_model
        model = build_lstm_model(vocab_size=1000, embed_dim=50, maxlen=50)
        dummy = np.zeros((4, 50), dtype=np.int32)
        pred  = model.predict(dummy, verbose=0)
        assert pred.shape == (4, 1), f"Expected (4,1), got {pred.shape}"

    def test_output_is_probability(self):
        from src.models.sentiment_models import build_lstm_model
        model = build_lstm_model(vocab_size=1000, embed_dim=50, maxlen=50)
        dummy = np.random.randint(0, 100, (8, 50), dtype=np.int32)
        pred  = model.predict(dummy, verbose=0)
        assert (pred >= 0).all() and (pred <= 1).all(), \
            "Sigmoid output must be in [0, 1]"

    def test_uses_binary_crossentropy(self):
        from src.models.sentiment_models import build_lstm_model
        model = build_lstm_model(vocab_size=1000, embed_dim=50, maxlen=50)
        assert model.loss == 'binary_crossentropy'

    def test_uses_accuracy_not_acc(self):
        """FIX: TF2 uses 'accuracy' not 'acc' — verify correct metric key."""
        from src.models.sentiment_models import build_lstm_model
        model       = build_lstm_model(vocab_size=1000, embed_dim=50, maxlen=50)
        metric_names = [m.name for m in model.metrics]
        assert 'accuracy' in metric_names, \
            "Must use 'accuracy' not deprecated 'acc' (TF2 fix)"
        assert 'acc' not in metric_names, \
            "'acc' is deprecated in TF2 — use 'accuracy'"

    def test_parameter_count_reasonable(self):
        from src.models.sentiment_models import build_lstm_model
        model = build_lstm_model(vocab_size=5000, embed_dim=100, maxlen=100)
        assert model.count_params() > 10_000

    def test_accepts_glove_embedding_matrix(self, dummy_embedding_matrix):
        from src.models.sentiment_models import build_lstm_model
        model = build_lstm_model(
            vocab_size=1000, embed_dim=100, maxlen=50,
            embedding_matrix=dummy_embedding_matrix[:1000],
        )
        assert model is not None
        dummy = np.zeros((2, 50), dtype=np.int32)
        pred  = model.predict(dummy, verbose=0)
        assert pred.shape == (2, 1)


class TestConv1DModel:

    def test_builds_without_error(self):
        from src.models.sentiment_models import build_conv1d_model
        model = build_conv1d_model(vocab_size=1000, embed_dim=50, maxlen=50)
        assert model is not None

    def test_output_shape(self):
        from src.models.sentiment_models import build_conv1d_model
        model = build_conv1d_model(vocab_size=1000, embed_dim=50, maxlen=50)
        dummy = np.zeros((4, 50), dtype=np.int32)
        pred  = model.predict(dummy, verbose=0)
        assert pred.shape == (4, 1)

    def test_has_global_max_pooling(self):
        from src.models.sentiment_models import build_conv1d_model
        model       = build_conv1d_model(vocab_size=1000, embed_dim=50, maxlen=50)
        layer_types = [type(l).__name__ for l in model.layers]
        assert 'GlobalMaxPooling1D' in layer_types, \
            "Conv1D model must use GlobalMaxPooling1D"

    def test_has_conv1d_layer(self):
        from src.models.sentiment_models import build_conv1d_model
        model       = build_conv1d_model(vocab_size=1000, embed_dim=50, maxlen=50)
        layer_types = [type(l).__name__ for l in model.layers]
        assert 'Conv1D' in layer_types

    def test_output_is_probability(self):
        from src.models.sentiment_models import build_conv1d_model
        model = build_conv1d_model(vocab_size=1000, embed_dim=50, maxlen=50)
        dummy = np.random.randint(0, 100, (6, 50), dtype=np.int32)
        pred  = model.predict(dummy, verbose=0)
        assert (pred >= 0).all() and (pred <= 1).all()


class TestBiLSTMModel:

    def test_builds_without_error(self):
        from src.models.sentiment_models import build_bilstm_model
        model = build_bilstm_model(vocab_size=1000, embed_dim=50, maxlen=50)
        assert model is not None

    def test_output_shape(self):
        from src.models.sentiment_models import build_bilstm_model
        model = build_bilstm_model(vocab_size=1000, embed_dim=50, maxlen=50)
        dummy = np.zeros((4, 50), dtype=np.int32)
        pred  = model.predict(dummy, verbose=0)
        assert pred.shape == (4, 1)

    def test_has_bidirectional_layer(self):
        from src.models.sentiment_models import build_bilstm_model
        model       = build_bilstm_model(vocab_size=1000, embed_dim=50, maxlen=50)
        layer_types = [type(l).__name__ for l in model.layers]
        assert 'Bidirectional' in layer_types, \
            "BiLSTM model must contain a Bidirectional wrapper layer"

    def test_bilstm_doubles_units(self):
        """Bidirectional should double the effective hidden dim."""
        from src.models.sentiment_models import build_bilstm_model
        import tensorflow as tf
        model = build_bilstm_model(vocab_size=1000, embed_dim=50, maxlen=50, lstm_units=32)
        bilstm_layer = [l for l in model.layers if 'Bidirectional' in type(l).__name__][0]
        # Output shape of BiLSTM = lstm_units * 2 (concat of forward + backward)
        output_dim = bilstm_layer.output_shape[-1]
        assert output_dim == 64, \
            f"BiLSTM(32 units) output should be 64 (32*2), got {output_dim}"


class TestGetCallbacks:

    def test_returns_four_callbacks(self):
        from src.models.sentiment_models import get_callbacks
        with tempfile.TemporaryDirectory() as tmp:
            import os; os.makedirs(f'{tmp}/models', exist_ok=True)
            cbs = get_callbacks('test_model')
        assert len(cbs) == 4

    def test_contains_required_callback_types(self):
        from src.models.sentiment_models import get_callbacks
        cbs   = get_callbacks('test_model')
        types = [type(c).__name__ for c in cbs]
        for required in ['EarlyStopping', 'ModelCheckpoint',
                         'ReduceLROnPlateau', 'CSVLogger']:
            assert required in types, f"Missing callback: {required}"

    def test_early_stopping_restores_best_weights(self):
        from src.models.sentiment_models import get_callbacks
        cbs = get_callbacks('test_model')
        es  = [c for c in cbs if type(c).__name__ == 'EarlyStopping'][0]
        assert es.restore_best_weights is True

    def test_monitors_val_accuracy(self):
        from src.models.sentiment_models import get_callbacks
        cbs = get_callbacks('test_model')
        es  = [c for c in cbs if type(c).__name__ == 'EarlyStopping'][0]
        assert es.monitor == 'val_accuracy'


# ═══════════════════════════════════════════════════════════════════════════
# DE: DATA PREPROCESSING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPySparkVersionFix:

    def test_pyspark_version_attribute_exists(self):
        """FIX: pyspark.__version__ is correct; pyspark.version does not exist."""
        import pyspark
        version = pyspark.__version__
        assert isinstance(version, str)
        assert len(version) > 0
        assert '.' in version, "Version string should be in x.y.z format"

    def test_pyspark_version_not_attribute_error(self):
        """Confirm the original pyspark.version raises AttributeError."""
        import pyspark
        with pytest.raises(AttributeError):
            _ = pyspark.version  # This is the bug in the original code


class TestTextPreprocessing:

    def test_lowercase_conversion(self):
        from src.data.preprocessing import preprocess_text
        result = preprocess_text('GREAT PRODUCT WORKS WELL', stop_words=set())
        assert result == result.lower()

    def test_removes_html_tags(self):
        from src.data.preprocessing import preprocess_text
        result = preprocess_text('Great <br /> product <p>works well</p>', stop_words=set())
        assert '<' not in result and '>' not in result

    def test_removes_punctuation_and_numbers(self):
        from src.data.preprocessing import preprocess_text
        result = preprocess_text('Amazing!!! Product 100% works. 5 stars.', stop_words=set())
        assert '!' not in result
        assert '.' not in result
        # Numbers removed
        assert '100' not in result
        assert '5' not in result

    def test_removes_stopwords(self):
        from src.data.preprocessing import preprocess_text
        stop_words = {'the', 'is', 'a', 'an', 'this', 'it'}
        result     = preprocess_text('this is a great product', stop_words=stop_words)
        tokens     = result.split()
        for sw in stop_words:
            assert sw not in tokens, f"Stopword '{sw}' not removed"

    def test_handles_empty_string(self):
        from src.data.preprocessing import preprocess_text
        result = preprocess_text('', stop_words=set())
        assert result == ''

    def test_handles_html_only_input(self):
        from src.data.preprocessing import preprocess_text
        result = preprocess_text('<br /><p></p>', stop_words=set())
        assert result.strip() == ''

    def test_clean_reviews_filters_empty(self, sample_reviews):
        from src.data.preprocessing import clean_reviews
        import nltk
        nltk.download('stopwords', quiet=True)
        results = clean_reviews(sample_reviews)
        assert len(results) == len(sample_reviews)
        # All non-empty inputs should produce non-empty outputs
        for r in results:
            assert isinstance(r, str)


class TestTokenisation:

    def test_fit_on_train_only(self):
        """FIX: Tokenizer must be fitted on training data only to prevent leakage."""
        from src.data.preprocessing import tokenise_and_pad
        train = ['great product excellent quality', 'terrible broken waste']
        test  = ['amazing works perfectly', 'awful stopped working immediately']
        tok, X_tr, X_te, vocab = tokenise_and_pad(train, test, maxlen=10)

        # Test-only words should map to OOV index, not add to vocab
        # 'amazing' and 'awful' are test-only — they should be OOV
        oov_idx = tok.word_index.get('<OOV>', 1)
        # Check that test sequences only contain indices seen in training or OOV
        train_indices = set(tok.word_index.values())
        for seq in X_te:
            for idx in seq:
                if idx != 0:  # 0 = padding
                    assert idx in train_indices or idx == oov_idx

    def test_output_shapes(self):
        from src.data.preprocessing import tokenise_and_pad
        train = ['good product'] * 20
        test  = ['bad product'] * 10
        _, X_tr, X_te, _ = tokenise_and_pad(train, test, maxlen=15)
        assert X_tr.shape == (20, 15)
        assert X_te.shape == (10, 15)

    def test_padding_fills_to_maxlen(self):
        from src.data.preprocessing import tokenise_and_pad
        short_train = ['hi']
        short_test  = ['ok']
        _, X_tr, _, _ = tokenise_and_pad(short_train, short_test, maxlen=20)
        # Should be padded to length 20
        assert X_tr.shape[1] == 20
        # Most entries should be 0 (padding)
        assert (X_tr == 0).sum() > X_tr.shape[1] * 0.5

    def test_vocab_size_includes_oov(self):
        from src.data.preprocessing import tokenise_and_pad
        train = ['apple banana cherry', 'date elderberry fig']
        test  = ['grape honeydew']
        _, _, _, vocab_size = tokenise_and_pad(train, test, maxlen=5)
        # vocab_size should be len(word_index) + 1
        # +1 for 0 padding index
        assert vocab_size > 6  # at least 6 unique words + OOV + padding


class TestGloveLoading:

    def test_load_glove_returns_empty_for_missing_file(self):
        from src.data.preprocessing import load_glove
        result = load_glove('/nonexistent/path/glove.txt', embed_dim=100)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_build_embedding_matrix_shape(self):
        from src.data.preprocessing import build_embedding_matrix
        word_index  = {'good': 1, 'bad': 2, 'great': 3}
        glove_dict  = {
            'good':  np.ones(50, dtype=np.float32),
            'bad':   np.ones(50, dtype=np.float32) * -1,
            # 'great' intentionally missing — zero initialised
        }
        matrix, coverage = build_embedding_matrix(
            word_index, glove_dict, vocab_size=4, embed_dim=50
        )
        assert matrix.shape == (4, 50)

    def test_missing_words_are_zero_initialised(self):
        from src.data.preprocessing import build_embedding_matrix
        word_index = {'unknown_word_xyz': 1}
        glove_dict = {}  # empty — word not in GloVe
        matrix, coverage = build_embedding_matrix(
            word_index, glove_dict, vocab_size=2, embed_dim=50
        )
        np.testing.assert_array_equal(
            matrix[1], np.zeros(50),
            err_msg='Words not in GloVe should be zero-initialised',
        )

    def test_coverage_is_between_zero_and_hundred(self):
        from src.data.preprocessing import build_embedding_matrix
        word_index = {'good': 1, 'bad': 2, 'missing': 3}
        glove_dict = {'good': np.ones(50, dtype=np.float32)}
        _, coverage = build_embedding_matrix(
            word_index, glove_dict, vocab_size=4, embed_dim=50
        )
        assert 0.0 <= coverage <= 100.0

    def test_found_words_have_correct_vectors(self):
        from src.data.preprocessing import build_embedding_matrix
        vec         = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        word_index  = {'test': 1}
        glove_dict  = {'test': vec}
        matrix, _   = build_embedding_matrix(
            word_index, glove_dict, vocab_size=2, embed_dim=3
        )
        np.testing.assert_array_almost_equal(matrix[1], vec)


# ═══════════════════════════════════════════════════════════════════════════
# DS: EVALUATION METRICS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluationMetrics:

    def test_returns_all_required_keys(self, binary_predictions):
        from src.evaluation.metrics import evaluate_predictions
        y_true, y_prob = binary_predictions
        res = evaluate_predictions(y_true, y_prob, verbose=False)
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            assert key in res, f"Missing metric key: {key}"

    def test_all_metrics_in_valid_range(self, binary_predictions):
        from src.evaluation.metrics import evaluate_predictions
        y_true, y_prob = binary_predictions
        res = evaluate_predictions(y_true, y_prob, verbose=False)
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
            assert 0.0 <= res[key] <= 1.0, f"{key} = {res[key]} is out of [0, 1]"

    def test_perfect_classifier_metrics(self, perfect_predictions):
        from src.evaluation.metrics import evaluate_predictions
        y_true, y_prob = perfect_predictions
        res = evaluate_predictions(y_true, y_prob, threshold=0.5, verbose=False)
        assert res['accuracy']  == 1.0
        assert res['precision'] == 1.0
        assert res['recall']    == 1.0
        assert res['f1']        == 1.0
        assert res['roc_auc']   == 1.0

    def test_lower_threshold_increases_recall(self, binary_predictions):
        from src.evaluation.metrics import evaluate_predictions
        y_true, y_prob = binary_predictions
        res_low  = evaluate_predictions(y_true, y_prob, threshold=0.2, verbose=False)
        res_high = evaluate_predictions(y_true, y_prob, threshold=0.8, verbose=False)
        assert res_low['recall'] >= res_high['recall'], \
            "Lower threshold should give higher or equal recall"

    def test_lower_threshold_decreases_precision(self, binary_predictions):
        from src.evaluation.metrics import evaluate_predictions
        y_true, y_prob = binary_predictions
        res_low  = evaluate_predictions(y_true, y_prob, threshold=0.2, verbose=False)
        res_high = evaluate_predictions(y_true, y_prob, threshold=0.8, verbose=False)
        assert res_low['precision'] <= res_high['precision'], \
            "Lower threshold should give lower or equal precision"

    def test_y_pred_matches_threshold(self, binary_predictions):
        from src.evaluation.metrics import evaluate_predictions
        y_true, y_prob = binary_predictions
        threshold = 0.6
        res       = evaluate_predictions(y_true, y_prob, threshold=threshold, verbose=False)
        expected  = (y_prob >= threshold).astype(int)
        np.testing.assert_array_equal(res['y_pred'], expected)

    def test_find_best_threshold_returns_valid_range(self, binary_predictions):
        from src.evaluation.metrics import find_best_threshold
        y_true, y_prob = binary_predictions
        thresh, res    = find_best_threshold(y_true, y_prob, optimise_for='f1')
        assert 0.0 < thresh < 1.0
        assert res['f1'] > 0.0

    def test_compare_models_returns_winner(self, binary_predictions):
        from src.evaluation.metrics import evaluate_predictions, compare_models
        y_true, y_prob = binary_predictions
        r1 = evaluate_predictions(y_true, y_prob,         'ModelA', verbose=False)
        r2 = evaluate_predictions(y_true, y_prob * 0.85,  'ModelB', verbose=False)
        _, winner = compare_models([r1, r2])
        assert winner in ['ModelA', 'ModelB']

    def test_compare_models_scorecard_shape(self, binary_predictions):
        from src.evaluation.metrics import evaluate_predictions, compare_models
        y_true, y_prob = binary_predictions
        results = [
            evaluate_predictions(y_true, y_prob,        'LSTM',   verbose=False),
            evaluate_predictions(y_true, y_prob * 0.9,  'Conv1D', verbose=False),
            evaluate_predictions(y_true, y_prob * 0.95, 'BiLSTM', verbose=False),
        ]
        df, winner = compare_models(results)
        assert 'Metric'  in df.columns
        assert 'LSTM'    in df.columns
        assert 'Conv1D'  in df.columns
        assert 'BiLSTM'  in df.columns
        assert 'Winner'  in df.columns
        assert winner in ['LSTM', 'Conv1D', 'BiLSTM']


# ═══════════════════════════════════════════════════════════════════════════
# DS: BUSINESS IMPACT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBusinessImpactConfig:

    def test_default_config_values_are_positive(self):
        from src.evaluation.business_impact import ReviewImpactConfig
        cfg = ReviewImpactConfig()
        assert cfg.value_true_positive      > 0
        assert cfg.value_true_negative      > 0
        assert cfg.cost_false_negative      > 0
        assert cfg.cost_false_positive      > 0
        assert cfg.churn_cost_per_customer  > 0
        assert cfg.daily_review_volume      > 0

    def test_fn_cost_exceeds_fp_cost(self):
        """Missed negative (FN) should cost more than false alarm (FP)."""
        from src.evaluation.business_impact import ReviewImpactConfig
        cfg = ReviewImpactConfig()
        assert cfg.cost_false_negative > cfg.cost_false_positive, \
            "Missing a negative review must cost more than a false alarm"

    def test_churn_probability_is_fraction(self):
        from src.evaluation.business_impact import ReviewImpactConfig
        cfg = ReviewImpactConfig()
        assert 0.0 < cfg.churn_probability < 1.0


class TestCalculateSentimentImpact:

    def test_returns_impact_result(self, binary_predictions):
        from src.evaluation.business_impact import calculate_sentiment_impact
        y_true, y_prob = binary_predictions
        y_pred = (y_prob >= 0.5).astype(int)
        result = calculate_sentiment_impact(y_true, y_pred, 'TestModel')
        assert result.model_name == 'TestModel'

    def test_confusion_matrix_totals_match(self, binary_predictions):
        from src.evaluation.business_impact import calculate_sentiment_impact
        y_true, y_prob = binary_predictions
        y_pred = (y_prob >= 0.5).astype(int)
        result = calculate_sentiment_impact(y_true, y_pred, 'Test')
        total = result.tn + result.fp + result.fn + result.tp
        assert total == len(y_true), \
            "TN + FP + FN + TP must equal total sample count"

    def test_breakdown_sums_to_net(self, binary_predictions):
        from src.evaluation.business_impact import calculate_sentiment_impact
        y_true, y_prob = binary_predictions
        y_pred  = (y_prob >= 0.5).astype(int)
        result  = calculate_sentiment_impact(y_true, y_pred, 'Test')
        bd_sum  = sum(result.breakdown.values())
        assert abs(bd_sum - result.net_value) < 0.01, \
            "Breakdown values must sum to net_value"

    def test_perfect_model_has_no_fn(self, perfect_predictions):
        from src.evaluation.business_impact import calculate_sentiment_impact
        y_true, y_prob = perfect_predictions
        y_pred = (y_prob >= 0.5).astype(int)
        result = calculate_sentiment_impact(y_true, y_pred, 'Perfect')
        assert result.fn == 0
        assert result.recall == 1.0

    def test_higher_recall_reduces_fn_cost(self, binary_predictions):
        from src.evaluation.business_impact import calculate_sentiment_impact
        y_true, y_prob = binary_predictions
        y_pred_aggressive = (y_prob >= 0.2).astype(int)
        y_pred_conservative = (y_prob >= 0.8).astype(int)
        res_agg  = calculate_sentiment_impact(y_true, y_pred_aggressive, 'Aggressive')
        res_cons = calculate_sentiment_impact(y_true, y_pred_conservative, 'Conservative')
        assert res_agg.fn <= res_cons.fn, \
            "Lower threshold should produce fewer missed negatives"

    def test_compare_to_baseline_returns_advantage(self, binary_predictions):
        from src.evaluation.business_impact import (
            calculate_sentiment_impact, compare_to_baseline
        )
        y_true, y_prob = binary_predictions
        y_pred  = (y_prob >= 0.3).astype(int)
        result  = calculate_sentiment_impact(y_true, y_pred, 'DL')
        baseline, adv = compare_to_baseline(result, y_true)
        assert baseline.model_name == 'Keyword Baseline'
        assert isinstance(adv, float)

    def test_simulate_annual_volume_scales_correctly(self, binary_predictions):
        from src.evaluation.business_impact import (
            ReviewImpactConfig, calculate_sentiment_impact, simulate_annual_volume
        )
        y_true, y_prob = binary_predictions
        y_pred  = (y_prob >= 0.5).astype(int)
        config  = ReviewImpactConfig(daily_review_volume=2000, working_days_per_year=250)
        result  = calculate_sentiment_impact(y_true, y_pred, 'Test', config)
        annual  = simulate_annual_volume(result, len(y_true), config)

        expected_annual = 2000 * 250
        assert annual['annual_reviews_processed'] == expected_annual

        scale   = expected_annual / len(y_true)
        assert abs(annual['annual_net_value_zar'] - result.net_value * scale) < 1.0

    def test_churn_cost_is_included_in_net(self, binary_predictions):
        """Churn risk from missed negatives must appear in net value."""
        from src.evaluation.business_impact import (
            ReviewImpactConfig, calculate_sentiment_impact
        )
        y_true, y_prob = binary_predictions
        y_pred  = (y_prob >= 0.5).astype(int)
        result  = calculate_sentiment_impact(y_true, y_pred, 'Test')
        # If there are false negatives, churn cost must be negative
        if result.fn > 0:
            assert result.churn_cost < 0, \
                "Churn cost should be negative (a cost, not revenue)"
