import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to system path to allow imports from src.data_loader
# Assuming tests/unit is at project_root/tests/unit
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
print("path:", str(Path(__file__).parent.parent.parent / 'src'))

from src.models.model_evaluator import evaluate_regression_model, evaluate_classification_model

def test_evaluate_regression_model_perfect_fit():
    """Test regression evaluation with perfect predictions."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    results = evaluate_regression_model(y_true, y_pred)
    assert np.isclose(results['RMSE'], 0.0)
    assert np.isclose(results['R-squared'], 1.0)

def test_evaluate_regression_model_some_error():
    """Test regression evaluation with some prediction error."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    results = evaluate_regression_model(y_true, y_pred)
    assert results['RMSE'] > 0
    assert results['R-squared'] < 1.0
    assert np.isclose(results['RMSE'], 0.1, atol=1e-6) # All errors are 0.1, so RMSE should be 0.1

def test_evaluate_regression_model_empty_arrays(capsys):
    """Test regression evaluation with empty input arrays."""
    y_true = np.array([])
    y_pred = np.array([])
    results = evaluate_regression_model(y_true, y_pred)
    assert np.isnan(results['RMSE'])
    assert np.isnan(results['R-squared'])
    captured = capsys.readouterr()
    assert "Warning: Empty true or predicted arrays for regression evaluation." in captured.out

def test_evaluate_classification_model_perfect_fit():
    """Test classification evaluation with perfect predictions."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7]) # Threshold 0.5 -> 0,1,0,1,1
    results = evaluate_classification_model(y_true, y_pred_proba, threshold=0.5)
    assert np.isclose(results['Accuracy'], 1.0)
    assert np.isclose(results['Precision'], 1.0)
    assert np.isclose(results['Recall'], 1.0)
    assert np.isclose(results['F1-score'], 1.0)

def test_evaluate_classification_model_mixed_results():
    """Test classification evaluation with mixed prediction results."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.6, 0.4, 0.1, 0.9, 0.7]) # Threshold 0.5 -> 1,0,0,1,1
    # y_pred_binary = np.array([1, 0, 0, 1, 1])
    # TP: 2 (1->1)
    # FP: 1 (0->1)
    # TN: 1 (0->0)
    # FN: 1 (1->0)

    # Accuracy: (2+1)/5 = 3/5 = 0.6
    # Precision: TP / (TP + FP) = 2 / (2 + 1) = 2/3 = 0.666...
    # Recall: TP / (TP + FN) = 2 / (2 + 1) = 2/3 = 0.666...
    # F1: 2 * (Prec*Rec) / (Prec+Rec) = 2 * (2/3*2/3) / (2/3+2/3) = 2 * (4/9) / (4/3) = 8/9 / 4/3 = 8/9 * 3/4 = 24/36 = 2/3 = 0.666...

    results = evaluate_classification_model(y_true, y_pred_proba, threshold=0.5)
    assert np.isclose(results['Accuracy'], 0.6)
    assert np.isclose(results['Precision'], 2/3)
    assert np.isclose(results['Recall'], 2/3)
    assert np.isclose(results['F1-score'], 2/3)

def test_evaluate_classification_model_custom_threshold():
    """Test classification evaluation with a custom threshold."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.6, 0.2, 0.8, 0.7]) # Threshold 0.7 -> 0,0,0,1,1
    # y_pred_binary = np.array([0, 0, 0, 1, 1])
    # TP: 2 (1->1)
    # FP: 0
    # TN: 2 (0->0)
    # FN: 1 (1->0)

    # Accuracy: (2+2)/5 = 4/5 = 0.8
    # Precision: 2 / (2+0) = 1.0
    # Recall: 2 / (2+1) = 2/3 = 0.666...
    # F1: 2 * (1.0 * 2/3) / (1.0 + 2/3) = 2 * (2/3) / (5/3) = 4/3 * 3/5 = 4/5 = 0.8

    results = evaluate_classification_model(y_true, y_pred_proba, threshold=0.7)
    assert np.isclose(results['Accuracy'], 0.8)
    assert np.isclose(results['Precision'], 1.0)
    assert np.isclose(results['Recall'], 2/3)
    assert np.isclose(results['F1-score'], 0.8)

def test_evaluate_classification_model_empty_arrays(capsys):
    """Test classification evaluation with empty input arrays."""
    y_true = np.array([])
    y_pred_proba = np.array([])
    results = evaluate_classification_model(y_true, y_pred_proba)
    assert np.isnan(results['Accuracy'])
    assert np.isnan(results['Precision'])
    captured = capsys.readouterr()
    assert "Warning: Empty true or predicted arrays for classification evaluation." in captured.out
