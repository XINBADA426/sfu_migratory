#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: CB Ren
# @Created Date:   2025/6/5 9:34
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")

from pathlib import Path
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import (roc_curve, auc, confusion_matrix, classification_report,
                             precision_recall_curve, average_precision_score, f1_score,
                             accuracy_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, GridSearchCV, learning_curve, validation_curve)

#### Log Setting ####

logger = logging.getLogger(__file__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

#### Plot Setting ####
import matplotlib.font_manager as fm

font_path = "/data1/NFS/home/rcb/.fonts/arial.ttf"
fm.fontManager.addfont(font_path)
font_prop = fm.FontProperties(fname=font_path)
font_name = font_prop.get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['font.sans-serif'] = [font_name]
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

COLORS = {
    'primary': '#2E86AB',  # Deep blue
    'secondary': '#A23B72',  # Rose
    'tertiary': '#F18F01',  # Orange
    'quaternary': '#C73E1D',  # Red
    'success': '#2A9D8F',  # Teal
    'neutral': '#264653',  # Dark gray
    'light': '#E9C46A',  # Light yellow
    'background': '#F4F4F4'  # Light gray background
}


#### Functions ####
def plot_roc_pr_curves(model, X_test, y_test, f_out: Path, seed=42, n_bootstraps=1000):
    """ROC curve with CI and PR curve without CI"""
    try:
        d_out = f_out.parent
        d_out.mkdir(parents=True, exist_ok=True)

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate main curves
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

        # Bootstrap只用于ROC和AUC值的CI
        rng = np.random.RandomState(seed)
        tprs = []
        aucs = []
        pr_aucs = []

        base_fpr = np.linspace(0, 1, 101)

        for i in range(n_bootstraps):
            indices = rng.randint(0, len(y_test), len(y_test))
            if len(np.unique(y_test[indices])) < 2:
                continue

            # ROC的bootstrap
            fpr_boot, tpr_boot, _ = roc_curve(y_test[indices], y_pred_proba[indices])
            tpr_boot_interp = np.interp(base_fpr, fpr_boot, tpr_boot)
            tpr_boot_interp[0] = 0.0
            tprs.append(tpr_boot_interp)
            aucs.append(auc(fpr_boot, tpr_boot))

            # 只计算PR AUC的CI，不计算曲线CI
            pr_aucs.append(average_precision_score(y_test[indices], y_pred_proba[indices]))

        # ROC置信区间
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std_tprs = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + 1.96 * std_tprs, 1)
        tprs_lower = np.maximum(mean_tprs - 1.96 * std_tprs, 0)

        # AUC的95% CI
        aucs = np.array(aucs)
        auc_ci_lower = np.percentile(aucs, 2.5)
        auc_ci_upper = np.percentile(aucs, 97.5)

        # Panel A: ROC curve with CI
        ax1.plot(fpr, tpr, color=COLORS['primary'], linewidth=2,
                 label=f'AUC = {roc_auc:.3f} ({auc_ci_lower:.3f}-{auc_ci_upper:.3f})')
        ax1.fill_between(base_fpr, tprs_lower, tprs_upper,
                         color=COLORS['primary'], alpha=0.2)
        ax1.plot([0, 1], [0, 1], color=COLORS['neutral'],
                 linewidth=1, linestyle='--', alpha=0.5)

        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False positive rate', fontsize=9)
        ax1.set_ylabel('True positive rate', fontsize=9)
        ax1.set_title('A. ROC curve', fontsize=10, loc='left', pad=10)
        ax1.legend(loc="lower right", frameon=True, fancybox=False,
                   edgecolor='black', framealpha=0.9, fontsize=6)

        # PR AUC的95% CI
        pr_aucs = np.array(pr_aucs)
        pr_auc_ci_lower = np.percentile(pr_aucs, 2.5)
        pr_auc_ci_upper = np.percentile(pr_aucs, 97.5)

        # Panel B: PR curve (不显示曲线CI，只在标签中显示AUC的CI)
        ax2.plot(recall, precision, color=COLORS['secondary'], linewidth=2,
                 label=f'AP = {pr_auc:.3f} ({pr_auc_ci_lower:.3f}-{pr_auc_ci_upper:.3f})')

        # Add baseline
        baseline = len(y_test[y_test == 1]) / len(y_test)
        ax2.axhline(y=baseline, color=COLORS['neutral'],
                    linewidth=1, linestyle='--', alpha=0.5)

        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=9)
        ax2.set_ylabel('Precision', fontsize=9)
        ax2.set_title('B. Precision-recall curve', fontsize=10, loc='left', pad=10)
        ax2.legend(loc="lower left", frameon=True, fancybox=False,
                   edgecolor='black', framealpha=0.9, fontsize=6)

        # Customize both panels
        for ax in [ax1, ax2]:
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(f_out, bbox_inches='tight', dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error plotting ROC/PR curves: {e}")
        raise


def plot_confusion_matrix(model, X_test, y_test, f_out: Path, title="Confusion Matrix"):
    """confusion matrix"""
    try:
        d_out = f_out.parent
        d_out.mkdir(exist_ok=True, parents=True)

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Create figure
        fig, ax = plt.subplots(figsize=(3.5, 3))

        # Create custom colormap
        cmap = plt.cm.Blues

        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        # Add colorbar with proper size
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10, fontweight='bold')

        # Customize axes
        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(['Predicted\nNegative', 'Predicted\nPositive'], fontsize=8)
        ax.set_yticklabels(['True\nNegative', 'True\nPositive'], fontsize=8)

        # Add metrics as text box
        textstr = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-score: {f1:.3f}'
        props = dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='black', linewidth=0.5, alpha=0.9)
        ax.text(1.25, 0.5, textstr, transform=ax.transAxes, fontsize=7,
                verticalalignment='center', bbox=props)

        ax.set_title(title, fontsize=10, pad=10)

        plt.tight_layout()
        plt.savefig(f_out, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")


def get_ylabel_for_scoring(scoring):
    """根据评分标准返回合适的y轴标签"""

    ylabel_mapping = {
        # 分类指标
        'accuracy': 'Accuracy',
        'roc_auc': 'ROC-AUC',
        'roc_auc_ovr': 'ROC-AUC (OvR)',
        'roc_auc_ovo': 'ROC-AUC (OvO)',
        'f1': 'F1-Score',
        'f1_micro': 'F1-Score (Micro)',
        'f1_macro': 'F1-Score (Macro)',
        'f1_weighted': 'F1-Score (Weighted)',
        'precision': 'Precision',
        'precision_micro': 'Precision (Micro)',
        'precision_macro': 'Precision (Macro)',
        'precision_weighted': 'Precision (Weighted)',
        'recall': 'Recall',
        'recall_micro': 'Recall (Micro)',
        'recall_macro': 'Recall (Macro)',
        'recall_weighted': 'Recall (Weighted)',
        'balanced_accuracy': 'Balanced Accuracy',
        'average_precision': 'Average Precision',
        'jaccard': 'Jaccard Score',
        'matthews_corrcoef': 'MCC',

        # 回归指标
        'r2': 'R² Score',
        'neg_mean_squared_error': 'MSE',  # 注意是负值
        'neg_root_mean_squared_error': 'RMSE',  # 注意是负值
        'neg_mean_absolute_error': 'MAE',  # 注意是负值
        'neg_mean_absolute_percentage_error': 'MAPE',  # 注意是负值
        'neg_median_absolute_error': 'MedAE',  # 注意是负值
        'explained_variance': 'Explained Variance',
        'max_error': 'Max Error',
        'neg_mean_poisson_deviance': 'Mean Poisson Deviance',
        'neg_mean_gamma_deviance': 'Mean Gamma Deviance',

        # 聚类指标
        'adjusted_rand_score': 'Adjusted Rand Index',
        'adjusted_mutual_info_score': 'AMI',
        'homogeneity_score': 'Homogeneity',
        'completeness_score': 'Completeness',
        'v_measure_score': 'V-Measure',
        'fowlkes_mallows_score': 'Fowlkes-Mallows Score',
        'silhouette_score': 'Silhouette Score',
        'calinski_harabasz_score': 'Calinski-Harabasz Score',
        'davies_bouldin_score': 'Davies-Bouldin Score'
    }

    if scoring in ylabel_mapping:
        return ylabel_mapping[scoring]
    else:
        return scoring.replace('_', ' ').replace('-', ' ').title()


def plot_learning_curve(model, X_train, y_train, f_out: Path, scoring='accuracy', cv=5, n_jobs=-1, train_sizes=None,
                        random_state=None, ylim=None, title='Learning curves'):
    """learning curve"""
    try:
        d_out = f_out.parent
        d_out.mkdir(exist_ok=True, parents=True)

        # 设置默认的训练集大小
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        # 直接使用传入的model对象
        train_sizes_abs, train_scores, val_scores = learning_curve(model,
                                                                   X_train, y_train,
                                                                   train_sizes=train_sizes,
                                                                   cv=cv,
                                                                   scoring=scoring,
                                                                   n_jobs=n_jobs,
                                                                   random_state=random_state)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
        # Create figure
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        # Plot with shaded error regions
        ax.fill_between(train_sizes_abs,
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std,
                        alpha=0.2, color=COLORS['secondary'])
        ax.fill_between(train_sizes_abs,
                        val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std,
                        alpha=0.2, color=COLORS['primary'])

        ax.plot(train_sizes_abs, train_scores_mean, 'o-',
                color=COLORS['secondary'], label='Training',
                markersize=5, linewidth=2)
        ax.plot(train_sizes_abs, val_scores_mean, 's-',
                color=COLORS['primary'], label='Validation',
                markersize=5, linewidth=2)

        ax.set_xlabel('Training set size', fontsize=9)
        ax.set_ylabel(get_ylabel_for_scoring(scoring), fontsize=9)
        ax.set_title(title, fontsize=10, pad=10)
        ax.legend(frameon=True, fancybox=False, edgecolor='black',
                  framealpha=0.9, loc='lower right')

        # Customize appearance
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            if scoring in ['accuracy', 'roc_auc', 'f1', 'precision', 'recall',
                           'balanced_accuracy', 'average_precision', 'jaccard',
                           'roc_auc_ovr', 'roc_auc_ovo', 'f1_micro', 'f1_macro',
                           'f1_weighted', 'precision_micro', 'precision_macro',
                           'precision_weighted', 'recall_micro', 'recall_macro',
                           'recall_weighted']:
                ax.set_ylim(0, 1.05)
            elif scoring == 'matthews_corrcoef':
                ax.set_ylim(-1, 1)
            elif scoring == 'r2':
                ax.set_ylim(-0.5, 1.05)
            elif scoring in ['silhouette_score']:
                ax.set_ylim(-1, 1)

        plt.tight_layout()
        plt.savefig(f_out, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Error plotting learning curve: {e}")


def multi_roc_pr_plot(models_dict, X_test, y_test, d_out: Path,
                      seed=42, n_bootstraps=1000, show_ci=True):
    """
    Plot ROC and PR curves for multiple models

    :param models_dict: A dictionary of model names and their corresponding models
    :param X_test: Test features
    :param y_test: Test labels
    :param d_out: Output directory
    :param seed: Random seed
    :param n_bootstraps: Number of bootstraps
    :param show_ci: Whether to show confidence intervals
    """
    try:
        d_out.mkdir(parents=True, exist_ok=True)
        f_plot = d_out.joinpath("multi_roc_curve.pdf")
        f_stat = d_out.joinpath("multi_roc_curve_stat.txt")

        colors = list(COLORS.values())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

        results_summary = []

        for idx, (model_name, model) in enumerate(models_dict.items()):
            color = colors[idx % len(colors)]

            y_pred_proba = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)

            if n_bootstraps > 0:
                rng = np.random.RandomState(seed + idx)
                aucs = []
                pr_aucs = []
                tprs = []
                base_fpr = np.linspace(0, 1, 101)

                for i in range(n_bootstraps):
                    indices = rng.randint(0, len(y_test), len(y_test))
                    if len(np.unique(y_test[indices])) < 2:
                        continue

                    fpr_boot, tpr_boot, _ = roc_curve(y_test[indices], y_pred_proba[indices])
                    tpr_boot_interp = np.interp(base_fpr, fpr_boot, tpr_boot)
                    tpr_boot_interp[0] = 0.0
                    tprs.append(tpr_boot_interp)
                    aucs.append(auc(fpr_boot, tpr_boot))

                    pr_aucs.append(average_precision_score(y_test[indices], y_pred_proba[indices]))

                aucs = np.array(aucs)
                auc_ci_lower = np.percentile(aucs, 2.5)
                auc_ci_upper = np.percentile(aucs, 97.5)

                pr_aucs = np.array(pr_aucs)
                pr_auc_ci_lower = np.percentile(pr_aucs, 2.5)
                pr_auc_ci_upper = np.percentile(pr_aucs, 97.5)

                tprs = np.array(tprs)
                mean_tprs = tprs.mean(axis=0)
                std_tprs = tprs.std(axis=0)
                tprs_upper = np.minimum(mean_tprs + 1.96 * std_tprs, 1)
                tprs_lower = np.maximum(mean_tprs - 1.96 * std_tprs, 0)
            else:
                auc_ci_lower = auc_ci_upper = None
                pr_auc_ci_lower = pr_auc_ci_upper = None

            if show_ci and n_bootstraps > 0:
                label = f'{model_name} (AUC = {roc_auc:.3f}, 95% CI: {auc_ci_lower:.3f}-{auc_ci_upper:.3f})'
                ax1.fill_between(base_fpr, tprs_lower, tprs_upper,
                                 color=color, alpha=0.1)
            else:
                label = f'{model_name} (AUC = {roc_auc:.3f})'

            ax1.plot(fpr, tpr, color=color, linewidth=2, label=label)

            if show_ci and n_bootstraps > 0:
                pr_label = f'{model_name} (AP = {pr_auc:.3f}, 95% CI: {pr_auc_ci_lower:.3f}-{pr_auc_ci_upper:.3f})'
            else:
                pr_label = f'{model_name} (AP = {pr_auc:.3f})'

            ax2.plot(recall, precision, color=color, linewidth=2, label=pr_label)

            results_summary.append({
                'Model': model_name,
                'ROC-AUC': roc_auc,
                'ROC-AUC CI': f'[{auc_ci_lower:.3f}, {auc_ci_upper:.3f}]' if auc_ci_lower is not None else 'N/A',
                'PR-AUC': pr_auc,
                'PR-AUC CI': f'[{pr_auc_ci_lower:.3f}, {pr_auc_ci_upper:.3f}]' if pr_auc_ci_lower is not None else 'N/A'
            })

        ax1.plot([0, 1], [0, 1], color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('A. ROC Curves', fontsize=14, fontweight='bold', pad=15, loc='left')
        ax1.legend(loc="lower right", frameon=True, fancybox=False,
                   edgecolor='black', framealpha=0.9, fontsize=5)
        ax1.grid(True, alpha=0.3, linewidth=0.5)

        baseline = len(y_test[y_test == 1]) / len(y_test)
        ax2.axhline(y=baseline, color='gray', linewidth=1, linestyle='--',
                    alpha=0.5, label=f'Baseline (Positive rate: {baseline:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('B. Precision-Recall Curves', fontsize=14, fontweight='bold', pad=15, loc='left')
        ax2.legend(loc="lower left", frameon=True, fancybox=False,
                   edgecolor='black', framealpha=0.9,fontsize=5)
        ax2.grid(True, alpha=0.3, linewidth=0.5)

        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_axisbelow(True)
            ax.tick_params(axis='both', labelsize=10)

        plt.tight_layout(w_pad=0.5)

        plt.savefig(f_plot, bbox_inches='tight', dpi=300)
        plt.close()
        with open(f_stat, 'w') as OUT:
            print("\nModel Performance Summary:", file=OUT)
            print("-" * 80, file=OUT)
            print(f"{'Model':<20}\t{'ROC-AUC':<12}\t{'ROC-AUC 95% CI':<20}\t{'PR-AUC':<12}\t{'PR-AUC 95% CI':<20}",
                  file=OUT)
            print("-" * 80, file=OUT)
            for result in results_summary:
                print(f"{result['Model']:<20}\t{result['ROC-AUC']:<12.3f}\t{result['ROC-AUC CI']:<20}\t"
                      f"{result['PR-AUC']:<12.3f}\t{result['PR-AUC CI']:<20}", file=OUT)

        return results_summary

    except Exception as e:
        print(f"Error plotting multi-model ROC/PR curves: {e}")
        raise
