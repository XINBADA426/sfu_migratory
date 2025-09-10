#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: CB Ren
# @Created Date:   2025/9/10 9:34
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

import click
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)

from utils import get_features, auto_load_data

logger = logging.getLogger(__file__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Available metrics
AVAILABLE_METRICS = [
    'roc_auc',  # ROC-AUC
    'pr_auc',  # PR-AUC (Average Precision)
    'accuracy',  # Accuracy
    'sensitivity',  # Sensitivity (Recall/TPR)
    'specificity',  # Specificity (TNR)
    'precision',  # Precision (Macro Average by default)
    'precision_weighted',  # Precision (Weighted Average)
    'precision_pos',  # Precision (Positive Class Only)
    'precision_neg',  # Precision (Negative Class Only)
    'recall',  # Recall (Macro Average)
    'f1',  # F1 Score (Macro Average)
    'npv'  # Negative Predictive Value
]


def load_model(model_path: Path) -> Any:
    """Load model from pickle file with error handling."""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def get_model_probabilities(model, x_test: np.ndarray, model_name: str) -> np.ndarray:
    """Get probability predictions from model, handling different model types."""

    # Try predict_proba first (most common)
    if hasattr(model, 'predict_proba'):
        try:
            prob_predictions = model.predict_proba(x_test)

            # Handle different output shapes
            if prob_predictions.ndim == 2:
                if prob_predictions.shape[1] == 2:
                    # Binary classification with 2 columns [prob_class_0, prob_class_1]
                    return prob_predictions[:, 1]
                elif prob_predictions.shape[1] == 1:
                    # Binary classification with 1 column
                    return prob_predictions.ravel()
            else:
                # 1D array
                return prob_predictions.ravel()

        except Exception as e:
            logger.warning(f"predict_proba failed for {model_name}: {e}")

    # Try decision_function (for SVM, etc.)
    if hasattr(model, 'decision_function'):
        try:
            decision_scores = model.decision_function(x_test)
            # Convert decision scores to probabilities using sigmoid
            return 1 / (1 + np.exp(-decision_scores))
        except Exception as e:
            logger.warning(f"decision_function failed for {model_name}: {e}")

    # Try predict if nothing else works
    if hasattr(model, 'predict'):
        logger.warning(f"{model_name} does not support probability predictions. Using class predictions.")
        return model.predict(x_test).astype(float)

    raise ValueError(f"Model {model_name} does not have any prediction method")


def get_model_predictions(model, x_test: np.ndarray, model_name: str) -> np.ndarray:
    """Get class predictions from model."""
    if hasattr(model, 'predict'):
        try:
            predictions = model.predict(x_test)
            return predictions
        except Exception as e:
            logger.error(f"predict failed for {model_name}: {e}")
            raise
    else:
        raise ValueError(f"Model {model_name} does not have predict method")


def standardize_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Standardize labels to 0 and 1, returning the mapping."""
    y = np.asarray(y).ravel()
    unique_labels = np.unique(y)

    if len(unique_labels) > 2:
        raise ValueError(f"Multi-class classification detected ({len(unique_labels)} classes). "
                         "This tool only supports binary classification.")

    if len(unique_labels) == 2:
        # Sort to ensure consistency
        unique_labels = np.sort(unique_labels)
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        y_binary = np.array([label_map[label] for label in y])

    elif len(unique_labels) == 1:
        # logger.warning(f"Only one class found: {unique_labels[0]}")
        # Treat the single class as negative class
        label_map = {unique_labels[0]: 0}
        y_binary = np.zeros_like(y, dtype=int)
    else:
        raise ValueError("No labels found in the data")

    return y_binary, label_map


def calculate_metric_value(
        y_true: np.ndarray,
        y_prob: np.ndarray = None,
        y_pred: np.ndarray = None,
        metric: str = None
) -> float:
    """Calculate a single metric value.

    Args:
        y_true: True labels
        y_prob: Probability predictions (for ROC-AUC, PR-AUC)
        y_pred: Class predictions (for other metrics)
        metric: Metric name
    """
    y_true, _ = standardize_labels(y_true)

    # For probability-based metrics
    if metric in ['roc_auc', 'pr_auc']:
        if y_prob is None:
            raise ValueError(f"{metric} requires probability predictions")

        # Ensure probabilities are in valid range
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)

        if metric == 'roc_auc':
            if len(np.unique(y_true)) < 2:
                logger.warning("ROC-AUC requires both classes to be present")
                return np.nan
            return roc_auc_score(y_true, y_prob)

        elif metric == 'pr_auc':
            if len(np.unique(y_true)) < 2:
                logger.warning("PR-AUC requires both classes to be present")
                return np.nan
            return average_precision_score(y_true, y_prob)

    # For classification-based metrics
    if y_pred is None:
        raise ValueError(f"{metric} requires class predictions")

    # Standardize predictions
    y_pred, _ = standardize_labels(y_pred)

    if metric == 'accuracy':
        return accuracy_score(y_true, y_pred)

    elif metric == 'precision':
        return precision_score(y_true, y_pred, average='macro', zero_division=0)

    elif metric == 'precision_weighted':
        return precision_score(y_true, y_pred, average='weighted', zero_division=0)

    elif metric == 'precision_pos':
        return precision_score(y_true, y_pred, pos_label=1, zero_division=0)

    elif metric == 'precision_neg':
        return precision_score(y_true, y_pred, pos_label=0, zero_division=0)

    elif metric in ['recall', 'sensitivity']:
        return recall_score(y_true, y_pred, average='macro', zero_division=0)

    elif metric == 'specificity':
        if len(np.unique(y_true)) < 2:
            return np.nan
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp = cm[0, 0], cm[0, 1]
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    elif metric == 'npv':
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fn = cm[0, 0], cm[1, 0]
        return tn / (tn + fn) if (tn + fn) > 0 else 0

    elif metric == 'f1':
        return f1_score(y_true, y_pred, average='macro', zero_division=0)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def bootstrap_ci(
        y_true: np.ndarray,
        y_prob: np.ndarray = None,
        y_pred: np.ndarray = None,
        metric: str = None,
        n_bootstraps: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 42
) -> Tuple[float, float, float]:
    """Calculate metric with bootstrap confidence interval."""

    np.random.seed(random_state)
    n_samples = len(y_true)

    y_true, label_map = standardize_labels(y_true)

    # Calculate original value
    original_value = calculate_metric_value(y_true, y_prob, y_pred, metric)

    # If original value is NaN, return NaN for CI as well
    if np.isnan(original_value):
        return original_value, np.nan, np.nan

    bootstrap_values = []
    for i in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]

        # Bootstrap the appropriate predictions
        y_prob_boot = y_prob[indices] if y_prob is not None else None
        y_pred_boot = y_pred[indices] if y_pred is not None else None

        try:
            value = calculate_metric_value(y_true_boot, y_prob_boot, y_pred_boot, metric)
            if not np.isnan(value):
                bootstrap_values.append(value)
        except Exception as e:
            if i < 5:
                logger.debug(f"Bootstrap iteration {i} failed: {e}")
            continue

    if len(bootstrap_values) < 10:
        logger.warning(f"Too few successful bootstrap iterations ({len(bootstrap_values)})")
        return original_value, np.nan, np.nan

    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_values, lower_percentile)
    ci_upper = np.percentile(bootstrap_values, upper_percentile)

    return original_value, ci_lower, ci_upper


def diagnose_predictions(y_true: np.ndarray, y_prob: np.ndarray = None,
                         y_pred: np.ndarray = None) -> None:
    """Diagnose potential issues with predictions and show detailed metrics."""

    y_true_binary, _ = standardize_labels(y_true)

    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC INFORMATION")
    logger.info("=" * 60)

    if y_prob is not None:
        logger.info(f"Probability statistics:")
        logger.info(f"  • Min: {y_prob.min():.6f}")
        logger.info(f"  • Max: {y_prob.max():.6f}")
        logger.info(f"  • Mean: {y_prob.mean():.6f}")
        logger.info(f"  • Median: {np.median(y_prob):.6f}")
        logger.info(f"  • Std: {y_prob.std():.6f}")

    if y_pred is not None:
        y_pred_binary, _ = standardize_labels(y_pred)

        n_positive_pred = np.sum(y_pred_binary)
        n_negative_pred = len(y_pred_binary) - n_positive_pred
        logger.info(f"\nPredictions (from model.predict()):")
        logger.info(
            f"  • Predicted positive (1): {n_positive_pred} ({n_positive_pred / len(y_pred_binary) * 100:.1f}%)")
        logger.info(
            f"  • Predicted negative (0): {n_negative_pred} ({n_negative_pred / len(y_pred_binary) * 100:.1f}%)")

        n_positive_true = np.sum(y_true_binary)
        n_negative_true = len(y_true_binary) - n_positive_true
        logger.info(f"\nTrue label distribution:")
        logger.info(f"  • Actual positive (1): {n_positive_true} ({n_positive_true / len(y_true_binary) * 100:.1f}%)")
        logger.info(f"  • Actual negative (0): {n_negative_true} ({n_negative_true / len(y_true_binary) * 100:.1f}%)")

        if len(np.unique(y_true_binary)) >= 2:
            report = classification_report(y_true_binary, y_pred_binary, output_dict=True, zero_division=0)

            logger.info(f"\nPrecision Breakdown:")
            logger.info(f"  • Class 0 (Negative): {report['0']['precision']:.4f}")
            logger.info(f"  • Class 1 (Positive): {report['1']['precision']:.4f}")
            logger.info(f"  • Macro Average: {report['macro avg']['precision']:.4f} "
                        f"(simple average of both classes)")
            logger.info(f"  • Weighted Average: {report['weighted avg']['precision']:.4f} "
                        f"(weighted by support)")

            logger.info(f"\nRecall Breakdown:")
            logger.info(f"  • Class 0 (Negative): {report['0']['recall']:.4f}")
            logger.info(f"  • Class 1 (Positive): {report['1']['recall']:.4f}")
            logger.info(f"  • Macro Average: {report['macro avg']['recall']:.4f}")
            logger.info(f"  • Weighted Average: {report['weighted avg']['recall']:.4f}")

            logger.info(f"\nF1 Score Breakdown:")
            logger.info(f"  • Class 0 (Negative): {report['0']['f1-score']:.4f}")
            logger.info(f"  • Class 1 (Positive): {report['1']['f1-score']:.4f}")
            logger.info(f"  • Macro Average: {report['macro avg']['f1-score']:.4f}")
            logger.info(f"  • Weighted Average: {report['weighted avg']['f1-score']:.4f}")

            cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            logger.info(f"\nConfusion Matrix:")
            logger.info(f"  • True Negatives: {tn}")
            logger.info(f"  • False Positives: {fp}")
            logger.info(f"  • False Negatives: {fn}")
            logger.info(f"  • True Positives: {tp}")

    logger.info("=" * 60 + "\n")


def process_model(
        model_path: Path,
        model_name: str,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        metrics: List[str],
        n_bootstraps: int,
        random_state: int,
        diagnose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Process a single model and calculate metrics."""

    logger.info(f"Loading {model_name} model from {model_path}...")
    model = load_model(model_path)

    logger.info(f"Model type: {type(model).__name__}")
    if hasattr(model, 'classes_'):
        logger.info(f"Model classes: {model.classes_}")

    # Determine which predictions we need
    need_probabilities = any(m in ['roc_auc', 'pr_auc'] for m in metrics)
    need_predictions = any(m not in ['roc_auc', 'pr_auc'] for m in metrics)

    y_test_prob = None
    y_test_pred = None

    # Get probability predictions if needed
    if need_probabilities:
        try:
            y_test_prob = get_model_probabilities(model, x_test, model_name)
            logger.info(f"Successfully obtained probability predictions for {model_name}")
        except Exception as e:
            logger.error(f"Failed to get probability predictions from {model_name}: {e}")
            if not need_predictions:
                raise

    # Get class predictions if needed
    if need_predictions:
        try:
            y_test_pred = get_model_predictions(model, x_test, model_name)
            logger.info(f"Successfully obtained class predictions for {model_name}")
        except Exception as e:
            logger.error(f"Failed to get class predictions from {model_name}: {e}")
            raise

    # Ensure y_test is 1D array
    if isinstance(y_test, pd.DataFrame):
        y_test_values = y_test.values.ravel()
    else:
        y_test_values = np.asarray(y_test).ravel()

    if diagnose:
        diagnose_predictions(y_test_values, y_test_prob, y_test_pred)

    logger.info(f"Calculating metrics for {model_name}: {', '.join(metrics)}")
    results = {}

    for metric in metrics:
        try:
            # Determine which predictions to use for this metric
            if metric in ['roc_auc', 'pr_auc']:
                value, ci_lower, ci_upper = bootstrap_ci(
                    y_test_values,
                    y_prob=y_test_prob,
                    y_pred=None,
                    metric=metric,
                    n_bootstraps=n_bootstraps,
                    random_state=random_state
                )
            else:
                value, ci_lower, ci_upper = bootstrap_ci(
                    y_test_values,
                    y_prob=None,
                    y_pred=y_test_pred,
                    metric=metric,
                    n_bootstraps=n_bootstraps,
                    random_state=random_state
                )

            # Format display name
            if metric == 'precision':
                display_name = 'PRECISION(MACRO)'
            elif metric == 'precision_weighted':
                display_name = 'PRECISION(WEIGHTED)'
            elif metric == 'precision_pos':
                display_name = 'PRECISION(POS)'
            elif metric == 'precision_neg':
                display_name = 'PRECISION(NEG)'
            else:
                display_name = metric.upper().replace('_', '-')

            if not np.isnan(value):
                results[display_name] = {
                    'value': value,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_formatted': f"[{ci_lower:.4f}, {ci_upper:.4f}]" if not np.isnan(ci_lower) else "N/A"
                }
            else:
                results[display_name] = {
                    'value': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'ci_formatted': "N/A"
                }

        except Exception as e:
            logger.warning(f"Failed to calculate {metric}: {e}")
            display_name = metric.upper().replace('_', '-')
            results[display_name] = {
                'value': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'ci_formatted': "N/A"
            }

    return results


def format_output_table(results: Dict[str, Dict[str, Dict[str, Any]]], metrics: List[str]) -> pd.DataFrame:
    """Format results into a well-structured DataFrame."""
    data = []

    display_metrics = []
    for m in metrics:
        if m == 'precision':
            display_metrics.append('PRECISION(MACRO)')
        elif m == 'precision_weighted':
            display_metrics.append('PRECISION(WEIGHTED)')
        elif m == 'precision_pos':
            display_metrics.append('PRECISION(POS)')
        elif m == 'precision_neg':
            display_metrics.append('PRECISION(NEG)')
        else:
            display_metrics.append(m.upper().replace('_', '-'))

    for model_name, model_metrics in results.items():
        row = {'Model': model_name}
        for metric, display_metric in zip(metrics, display_metrics):
            if display_metric in model_metrics:
                metric_data = model_metrics[display_metric]
                if not np.isnan(metric_data['value']):
                    row[f'{display_metric}'] = f"{metric_data['value']:.4f}"
                    row[f'{display_metric}_95%CI'] = metric_data['ci_formatted']
                else:
                    row[f'{display_metric}'] = "N/A"
                    row[f'{display_metric}_95%CI'] = "N/A"
        data.append(row)

    df = pd.DataFrame(data)

    columns = ['Model']
    for display_metric in display_metrics:
        if f'{display_metric}' in df.columns:
            columns.append(f'{display_metric}')
            columns.append(f'{display_metric}_95%CI')

    df = df[columns]
    return df


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-vg", "--vgenotype",
              type=click.Path(exists=True),
              help="The validation genotype input file (must be provided with --vphenotype)")
@click.option("-vp", "--vphenotype",
              type=click.Path(exists=True),
              help="The validation phenotype input file (must be provided with --vgenotype)")
@click.option("--feature",
              type=click.Path(exists=True),
              help="A file containing features to be used")
@click.option("-dt", "--decisiontree",
              type=click.Path(exists=True),
              help="Decision Tree model")
@click.option("-rf", "--randomforest",
              type=click.Path(exists=True),
              help="Random Forest model")
@click.option("-nb", "--naivebayes",
              type=click.Path(exists=True),
              help="Naive Bayes model")
@click.option("-svm", "--svm",
              type=click.Path(exists=True),
              help="SVM model")
@click.option("-xgb", "--xgboost",
              type=click.Path(exists=True),
              help="XGBoost model")
@click.option("-mlp", "--mlpclassifier",
              type=click.Path(exists=True),
              help="MLPClassifier model")
@click.option("-lr", "--logisticregression",
              type=click.Path(exists=True),
              help="Logistic Regression model")
@click.option("-g", "--genotype",
              type=click.Path(exists=True),
              help="The genotype input file")
@click.option("-p", "--phenotype",
              type=click.Path(exists=True),
              help="The phenotype input file")
@click.option("--metrics",
              type=str,
              default="roc_auc",
              show_default=True,
              help=f"Comma-separated list of metrics. Available: {', '.join(AVAILABLE_METRICS)}")
@click.option("--precision-type",
              type=click.Choice(['macro', 'weighted', 'pos', 'neg', 'all']),
              default='macro',
              show_default=True,
              help="Type of precision to calculate: macro (simple average), weighted, pos (positive class only), neg (negative class only), or all")
@click.option("--diagnose/--no-diagnose",
              default=True,
              show_default=True,
              help="Show diagnostic information for predictions")
@click.option("--testsize",
              type=float,
              default=0.3,
              show_default=True,
              help="Test set proportion (when not using validation files)")
@click.option("--seed",
              type=int,
              default=42,
              show_default=True,
              help="Random seed for reproducibility")
@click.option("--bootstraps",
              default=1000,
              type=int,
              show_default=True,
              help="Number of bootstrap iterations for CI calculation")
@click.option("-o", "--out",
              type=click.Path(),
              required=True,
              help="Output file (.csv, .tsv, or .xlsx)")
def main(vgenotype, vphenotype, feature, decisiontree, randomforest, naivebayes, svm, xgboost,
         mlpclassifier, logisticregression, genotype, phenotype, metrics, precision_type,
         diagnose, testsize, seed, bootstraps, out):
    """
    Calculate metrics with 95% confidence intervals for classification models.

    ROC-AUC and PR-AUC are calculated using predict_proba (probability predictions).
    All other metrics are calculated using predict (class predictions).

    By default, precision, recall, and f1 use macro average (simple average of all classes).

    Examples:

        # Calculate macro average precision (default):
        python script.py -rf model.pkl -g genotype.tsv -p phenotype.tsv \
            --metrics "precision,recall,f1" -o results.csv

        # Calculate all types of precision:
        python script.py -rf model.pkl -g genotype.tsv -p phenotype.tsv \
            --metrics "precision" --precision-type all -o results.csv

        # Calculate weighted precision:
        python script.py -rf model.pkl -g genotype.tsv -p phenotype.tsv \
            --metrics "precision" --precision-type weighted -o results.csv

        # Multiple metrics with ROC and PR curves:
        python script.py -rf rf.pkl -xgb xgb.pkl -nb nb.pkl \
            -g genotype.tsv -p phenotype.tsv \
            --metrics "roc_auc,pr_auc,accuracy,precision,recall,f1" \
            -o results.xlsx
    """
    f_out = Path(out).absolute()
    d_out = f_out.parent
    d_out.mkdir(parents=True, exist_ok=True)

    selected_metrics = [m.strip().lower() for m in metrics.split(',')]

    if 'precision' in selected_metrics:
        selected_metrics.remove('precision')

        if precision_type == 'all':
            selected_metrics.extend(['precision', 'precision_weighted', 'precision_pos', 'precision_neg'])
            logger.info("Calculating all precision types: macro, weighted, positive class, negative class")
        elif precision_type == 'macro':
            selected_metrics.append('precision')  # Default is macro
            logger.info("Using macro average for precision (simple average of all classes)")
        elif precision_type == 'weighted':
            selected_metrics.append('precision_weighted')
            logger.info("Using weighted average for precision (weighted by class support)")
        elif precision_type == 'pos':
            selected_metrics.append('precision_pos')
            logger.info("Calculating precision for positive class only")
        elif precision_type == 'neg':
            selected_metrics.append('precision_neg')
            logger.info("Calculating precision for negative class only")

    invalid_metrics = [m for m in selected_metrics if m not in AVAILABLE_METRICS]
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {', '.join(invalid_metrics)}. "
                         f"Available metrics: {', '.join(AVAILABLE_METRICS)}")

    logger.info("=" * 60)
    logger.info("CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Output file: {f_out}")
    logger.info(f"Bootstrap iterations: {bootstraps}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Selected metrics: {', '.join(selected_metrics)}")
    logger.info(f"Diagnostic mode: {'ON' if diagnose else 'OFF'}")

    logger.info("\nPrediction Methods:")
    logger.info("  • ROC-AUC, PR-AUC: Using predict_proba (probabilities)")
    logger.info("  • Other metrics: Using predict (class predictions)")

    logger.info("\nMetric Descriptions:")
    metric_descriptions = {
        'roc_auc': 'ROC-AUC (Area Under ROC Curve) - uses probabilities',
        'pr_auc': 'PR-AUC (Precision-Recall AUC) - uses probabilities',
        'accuracy': 'Accuracy - uses class predictions',
        'precision': 'Precision (Macro Average) - uses class predictions',
        'precision_weighted': 'Precision (Weighted Average) - uses class predictions',
        'precision_pos': 'Precision (Positive Class Only) - uses class predictions',
        'precision_neg': 'Precision (Negative Class Only) - uses class predictions',
        'recall': 'Recall (Macro Average) - uses class predictions',
        'sensitivity': 'Sensitivity (Macro Average) - uses class predictions',
        'specificity': 'Specificity (True Negative Rate) - uses class predictions',
        'f1': 'F1 Score (Macro Average) - uses class predictions',
        'npv': 'Negative Predictive Value - uses class predictions'
    }
    for metric in selected_metrics:
        if metric in metric_descriptions:
            logger.info(f"  • {metric}: {metric_descriptions[metric]}")

    features = []
    if feature:
        f_feature = Path(feature).absolute()
        logger.info(f"\nLoading features from: {f_feature}")
        features = get_features(f_feature)
        logger.info(f"Number of features loaded: {len(features)}")

    logger.info("\n" + "=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)

    if vgenotype and vphenotype:
        f_vgenotype = Path(vgenotype).absolute()
        f_vphenotype = Path(vphenotype).absolute()
        logger.info(f"Loading validation data:")
        logger.info(f"  • Genotype: {f_vgenotype}")
        logger.info(f"  • Phenotype: {f_vphenotype}")

        x_test = pd.read_csv(f_vgenotype, sep="\t", index_col=0)
        y_test = pd.read_csv(f_vphenotype, sep="\t", index_col=0)

        if len(features) == 0:
            features = list(x_test.columns)
        x_test = x_test.loc[:, features]

    elif genotype and phenotype:
        f_genotype = Path(genotype).absolute()
        f_phenotype = Path(phenotype).absolute()
        logger.info(f"Loading and splitting data:")
        logger.info(f"  • Genotype: {f_genotype}")
        logger.info(f"  • Phenotype: {f_phenotype}")
        logger.info(f"  • Test size: {testsize:.1%}")

        x_train, x_test, y_train, y_test = auto_load_data(
            f_genotype,
            f_phenotype,
            test_size=testsize,
            random_seed=seed,
            features=features
        )
    else:
        raise ValueError(
            "Data input required: provide either (--vgenotype and --vphenotype) "
            "or (--genotype and --phenotype)"
        )

    logger.info(f"\nData summary:")
    logger.info(f"  • Test samples: {len(x_test)}")
    logger.info(f"  • Features: {len(features)}")

    if isinstance(y_test, pd.DataFrame):
        y_test_values = y_test.values.ravel()
    else:
        y_test_values = np.asarray(y_test).ravel()

    unique_labels = np.unique(y_test_values)
    logger.info(f"  • Unique labels: {unique_labels}")
    for label in unique_labels:
        count = np.sum(y_test_values == label)
        logger.info(f"    - Label {label}: {count} samples ({count / len(y_test_values) * 100:.1f}%)")

    models_config = [
        (decisiontree, "Decision Tree"),
        (randomforest, "Random Forest"),
        (naivebayes, "Naive Bayes"),
        (svm, "SVM"),
        (xgboost, "XGBoost"),
        (mlpclassifier, "MLP"),
        (logisticregression, "Logistic Regression")
    ]

    logger.info("\n" + "=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    results = {}
    for model_path, model_name in models_config:
        if model_path:
            logger.info(f"\n{'=' * 40}")
            logger.info(f"Processing {model_name}")
            logger.info(f"{'=' * 40}")
            try:
                model_results = process_model(
                    Path(model_path),
                    model_name,
                    x_test,
                    y_test,
                    selected_metrics,
                    bootstraps,
                    seed,
                    diagnose
                )
                results[model_name] = model_results
                logger.info(f"✓ {model_name} evaluation completed")
            except Exception as e:
                logger.error(f"✗ Failed to process {model_name}: {e}")
                continue

    if not results:
        raise ValueError("No model was successfully loaded and evaluated.")

    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)

    df_results = format_output_table(results, selected_metrics)

    file_extension = f_out.suffix.lower()

    if file_extension == '.csv':
        df_results.to_csv(f_out, index=False)
    elif file_extension == '.xlsx':
        df_results.to_excel(f_out, index=False, engine='openpyxl')
    else:
        df_results.to_csv(f_out, sep='\t', index=False)

    logger.info(f"Results saved to: {f_out}")

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Precision type: {precision_type}")
    print("Note: ROC-AUC and PR-AUC use predict_proba; other metrics use predict")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)

    if len(results) > 1:
        print("\nSUMMARY STATISTICS")
        print("-" * 80)

        for metric in selected_metrics:
            # Format display name
            if metric == 'precision':
                display_metric = 'PRECISION(MACRO)'
            elif metric == 'precision_weighted':
                display_metric = 'PRECISION(WEIGHTED)'
            elif metric == 'precision_pos':
                display_metric = 'PRECISION(POS)'
            elif metric == 'precision_neg':
                display_metric = 'PRECISION(NEG)'
            else:
                display_metric = metric.upper().replace('_', '-')

            values = []
            model_values = {}
            for model_name in results:
                if display_metric in results[model_name]:
                    val = results[model_name][display_metric]['value']
                    if not np.isnan(val):
                        values.append(val)
                        model_values[model_name] = val

            if values:
                best_model = max(model_values.keys(), key=lambda m: model_values[m])
                best_value = model_values[best_model]
                print(f"\n{display_metric}:")
                print(f"  • Best model: {best_model} ({best_value:.4f})")
                print(f"  • Mean across models: {np.mean(values):.4f}")
                print(f"  • Std across models: {np.std(values):.4f}")
                print(f"  • Min: {np.min(values):.4f}")
                print(f"  • Max: {np.max(values):.4f}")

    logger.info("\n✓ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
