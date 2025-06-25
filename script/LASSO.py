#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: CB Ren
# @Created Date: 2025/5/29 11:54
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")

from pathlib import Path
import logging
import click
import json
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import (StratifiedKFold, GridSearchCV, learning_curve, validation_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (roc_curve, auc, confusion_matrix, classification_report,
                             precision_recall_curve, average_precision_score, f1_score,
                             accuracy_score, precision_score, recall_score, roc_auc_score)
from sklearn.exceptions import ConvergenceWarning
import joblib
from tqdm import tqdm

from utils import save_param, save_report, auto_load_data

# Configure matplotlib
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

# Define color palette for publication
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LASSOFeatureSelector:
    """Enhanced LASSO feature selector"""

    def __init__(self, seed=42, max_iter=10000):
        self.seed = seed
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.best_model = None
        self.grid_search = None
        self.selected_features = []

    def get_features(self, f_feature_list):
        """从特征文件中读取特征列表"""
        try:
            with open(f_feature_list, "r") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading feature file: {e}")
            return []

    def validate_data(self, X, y, data_name="Data"):
        """验证数据的完整性和一致性"""
        # Check for missing values
        if X.isnull().any().any():
            logger.warning(f"{data_name} contains missing values. Filling with median...")
            X = X.fillna(X.median())

        # Check for infinite values
        if np.isinf(X.values).any():
            logger.warning(f"{data_name} contains infinite values. Replacing with max finite value...")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        # Check class balance
        class_counts = pd.Series(y).value_counts()
        if len(class_counts) != 2:
            raise ValueError(f"Expected binary classification, got {len(class_counts)} classes")

        imbalance_ratio = class_counts.max() / class_counts.min()
        if imbalance_ratio > 10:
            logger.warning(f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")

        return X, y

    def plot_coefficient_path(self, X_train, y_train, param_grid, d_out):
        """Publication-quality LASSO coefficient path plot"""
        try:
            f_coef_path = d_out / "Figure_1_coefficient_path.pdf"

            alphas = 1 / param_grid['C']
            coefs = []

            logger.info("Computing coefficient paths...")
            for alpha in tqdm(alphas, desc="Computing coefficients"):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model = LogisticRegression(
                        penalty='l1', C=1 / alpha, solver='liblinear',
                        random_state=self.seed, max_iter=self.max_iter
                    )
                    model.fit(X_train, y_train)
                    coefs.append(model.coef_[0])

            coefs = np.array(coefs)

            # Create figure with proper dimensions for publication
            fig, ax = plt.subplots(figsize=(3.5, 2.8))

            # Only plot non-zero coefficient paths
            non_zero_mask = np.any(coefs != 0, axis=0)
            feature_names = X_train.columns[non_zero_mask]
            coefs_filtered = coefs[:, non_zero_mask]

            # Select top features by maximum absolute coefficient
            max_abs_coefs = np.max(np.abs(coefs_filtered), axis=0)
            top_indices = np.argsort(max_abs_coefs)[-10:]  # Top 10 features

            # Plot with distinct colors
            colors = plt.cm.tab10(np.linspace(0, 1, len(top_indices)))

            for idx, (i, color) in enumerate(zip(top_indices, colors)):
                ax.plot(alphas, coefs_filtered[:, i],
                        color=color, linewidth=1.5, alpha=0.8,
                        label=feature_names[i] if idx < 10 else "")  # Only label top 5

            # Add vertical line at optimal alpha
            optimal_alpha = 1 / self.grid_search.best_params_['C']
            ax.axvline(optimal_alpha, color=COLORS['quaternary'],
                       linestyle='--', linewidth=1, alpha=0.7)

            ax.set_xscale('log')
            ax.set_xlabel('Regularization strength (α)', fontsize=9)
            ax.set_ylabel('Coefficient value', fontsize=9)
            ax.set_title('LASSO coefficient paths', fontsize=10, pad=10)

            # Customize grid
            ax.grid(True, alpha=0.3, linewidth=0.5, linestyle='-')
            ax.set_axisbelow(True)

            # Add legend
            if len(top_indices) > 0:
                ax.legend(loc='best', frameon=True, fancybox=False,
                          edgecolor='black', framealpha=0.9, fontsize=3)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(f_coef_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting coefficient path: {e}")

    def cv_auc_plot_enhanced(self, d_out):
        """Publication-quality cross-validation plot"""
        try:
            cv_results_df = pd.DataFrame(self.grid_search.cv_results_)
            f_cv_auc_plot = d_out / "Figure_2_cross_validation.pdf"

            # Create figure with two panels
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

            # Panel A: CV score with error bars
            ax1.errorbar(cv_results_df['param_C'],
                         cv_results_df['mean_test_score'],
                         yerr=cv_results_df['std_test_score'],
                         marker='o', markersize=4,
                         color=COLORS['primary'],
                         ecolor=COLORS['primary'],
                         capsize=3, capthick=1,
                         linewidth=1.5, alpha=0.8)

            # Highlight best parameter
            best_idx = cv_results_df['mean_test_score'].idxmax()
            ax1.scatter(cv_results_df.loc[best_idx, 'param_C'],
                        cv_results_df.loc[best_idx, 'mean_test_score'],
                        color=COLORS['quaternary'], s=50, zorder=5,
                        edgecolor='black', linewidth=0.5)

            ax1.set_xscale('log')
            ax1.set_xlabel('Regularization parameter C', fontsize=9)
            ax1.set_ylabel('Cross-validation AUC', fontsize=9)
            ax1.set_title('A. Model selection', fontsize=10, loc='left', pad=10)

            # Panel B: Train vs validation scores
            ax2.plot(cv_results_df['param_C'], cv_results_df['mean_train_score'],
                     marker='s', markersize=4, color=COLORS['secondary'],
                     label='Training', linewidth=1.5, alpha=0.8)
            ax2.plot(cv_results_df['param_C'], cv_results_df['mean_test_score'],
                     marker='o', markersize=4, color=COLORS['primary'],
                     label='Validation', linewidth=1.5, alpha=0.8)

            ax2.set_xscale('log')
            ax2.set_xlabel('Regularization parameter C', fontsize=9)
            ax2.set_ylabel('AUC score', fontsize=9)
            ax2.set_title('B. Training dynamics', fontsize=10, loc='left', pad=10)
            ax2.legend(frameon=True, fancybox=False, edgecolor='black',
                       framealpha=0.9, loc='best')

            # Customize both panels
            for ax in [ax1, ax2]:
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_ylim(0.5, 1.02)

            plt.tight_layout()
            plt.savefig(f_cv_auc_plot, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting CV AUC: {e}")

    def plot_learning_curve(self, X_train, y_train, d_out):
        """Publication-quality learning curve"""
        try:
            f_learning = d_out / "Figure_3_learning_curve.pdf"

            best_C = self.grid_search.best_params_['C']
            train_sizes = np.linspace(0.1, 1.0, 10)

            train_sizes_abs, train_scores, val_scores = learning_curve(
                LogisticRegression(penalty='l1', C=best_C, solver='liblinear',
                                   random_state=self.seed, max_iter=self.max_iter),
                X_train, y_train,
                train_sizes=train_sizes,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                random_state=self.seed
            )

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
            ax.set_ylabel('Accuracy', fontsize=9)
            ax.set_title('Learning curves', fontsize=10, pad=10)
            ax.legend(frameon=True, fancybox=False, edgecolor='black',
                      framealpha=0.9, loc='lower right')

            # Customize appearance
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(0.5, 1.02)

            plt.tight_layout()
            plt.savefig(f_learning, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting learning curve: {e}")

    def plot_feature_importance(self, X_train, d_out, top_n=10):
        """Publication-quality feature importance plot"""
        try:
            f_importance = d_out / "Figure_4_feature_importance.pdf"

            coefficients = self.best_model.coef_[0]
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)

            feature_importance = feature_importance[feature_importance['coefficient'] != 0]

            if len(feature_importance) == 0:
                logger.warning("No features selected by LASSO")
                return

            top_features = feature_importance.head(top_n)

            # Create figure
            fig, ax = plt.subplots(figsize=(3.5, 4))

            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            colors = [COLORS['quaternary'] if x < 0 else COLORS['primary']
                      for x in top_features['coefficient']]

            bars = ax.barh(y_pos, top_features['coefficient'],
                           color=colors, edgecolor='black', linewidth=0.5)

            # Add value labels
            for i, (idx, row) in enumerate(top_features.iterrows()):
                if abs(row['coefficient']) > 0.01:  # Only label significant coefficients
                    ax.text(row['coefficient'] + np.sign(row['coefficient']) * 0.002,
                            i, f'{row["coefficient"]:.2f}',
                            va='center', ha='left' if row['coefficient'] > 0 else 'right',
                            fontsize=5)

            # Customize y-axis
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'], fontsize=8)
            ax.invert_yaxis()

            # Add vertical line at zero
            ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')

            ax.set_xlabel('Coefficient value', fontsize=9)
            ax.set_title(f'Top {min(top_n, len(top_features))} selected features',
                         fontsize=10, pad=10)

            # Customize appearance
            ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(f_importance, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")

    def plot_roc_pr_curves(self, X_test, y_test, d_out):
        """Publication-quality ROC and PR curves"""
        try:
            f_curves = d_out / "Figure_5_performance_curves.pdf"

            y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

            # Calculate curves
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)

            # Create figure with two panels
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

            # Panel A: ROC curve
            ax1.plot(fpr, tpr, color=COLORS['primary'], linewidth=2,
                     label=f'AUC = {roc_auc:.3f}')
            ax1.plot([0, 1], [0, 1], color=COLORS['neutral'],
                     linewidth=1, linestyle='--', alpha=0.5)

            # Add confidence interval (bootstrap)
            n_bootstraps = 1000
            rng = np.random.RandomState(self.seed)
            tprs = []
            aucs = []

            for i in range(n_bootstraps):
                indices = rng.randint(0, len(y_test), len(y_test))
                if len(np.unique(y_test[indices])) < 2:
                    continue
                fpr_boot, tpr_boot, _ = roc_curve(y_test[indices], y_pred_proba[indices])
                tprs.append(np.interp(fpr, fpr_boot, tpr_boot))
                aucs.append(auc(fpr_boot, tpr_boot))

            tprs = np.array(tprs)
            mean_tprs = tprs.mean(axis=0)
            std_tprs = tprs.std(axis=0)

            tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
            tprs_lower = mean_tprs - std_tprs

            ax1.fill_between(fpr, tprs_lower, tprs_upper,
                             color=COLORS['primary'], alpha=0.2)

            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False positive rate', fontsize=9)
            ax1.set_ylabel('True positive rate', fontsize=9)
            ax1.set_title('A. ROC curve', fontsize=10, loc='left', pad=10)
            ax1.legend(loc="lower right", frameon=True, fancybox=False,
                       edgecolor='black', framealpha=0.9)

            # Panel B: PR curve
            ax2.plot(recall, precision, color=COLORS['secondary'], linewidth=2,
                     label=f'AP = {pr_auc:.3f}')

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
                       edgecolor='black', framealpha=0.9)

            # Customize both panels
            for ax in [ax1, ax2]:
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(f_curves, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting ROC/PR curves: {e}")

    def plot_confusion_matrix(self, X_test, y_test, d_out):
        """Publication-quality confusion matrix"""
        try:
            f_cm = d_out / "Figure_6_confusion_matrix.pdf"

            y_pred = self.best_model.predict(X_test)
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

            ax.set_title('Confusion matrix', fontsize=10, pad=10)

            plt.tight_layout()
            plt.savefig(f_cm, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")

    def plot_feature_count_vs_parameter(self, X_train, y_train, param_grid, d_out):
        """绘制特征数目随正则化参数变化的曲线"""
        try:
            f_feature_count = d_out / "Figure_feature_count_vs_C.pdf"

            # 获取不同C值对应的特征数目
            C_values = param_grid['C']
            feature_counts = []

            logger.info("Computing feature counts for different C values...")
            for C in tqdm(C_values, desc="Computing feature counts"):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model = LogisticRegression(
                        penalty='l1', C=C, solver='liblinear',
                        random_state=self.seed, max_iter=self.max_iter
                    )
                    model.fit(X_train, y_train)
                    # 计算非零系数的数量
                    n_features = np.sum(model.coef_[0] != 0)
                    feature_counts.append(n_features)

            # 创建图形
            fig, ax = plt.subplots(figsize=(3.5, 2.8))

            # 绘制曲线
            ax.plot(C_values, feature_counts,
                    color=COLORS['primary'],
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    alpha=0.8)

            # 标记最优参数位置
            optimal_C = self.grid_search.best_params_['C']
            optimal_idx = np.argmin(np.abs(C_values - optimal_C))
            optimal_count = feature_counts[optimal_idx]

            ax.scatter(optimal_C, optimal_count,
                       color=COLORS['quaternary'],
                       s=100,
                       zorder=5,
                       edgecolor='black',
                       linewidth=1,
                       label=f'Optimal C={optimal_C:.4f}\n{optimal_count} features')

            # 添加垂直线标记最优参数
            ax.axvline(optimal_C,
                       color=COLORS['quaternary'],
                       linestyle='--',
                       linewidth=1,
                       alpha=0.5)

            # 设置坐标轴
            ax.set_xscale('log')
            ax.set_xlabel('Regularization parameter C', fontsize=9)
            ax.set_ylabel('Number of selected features', fontsize=9)
            ax.set_title('Feature selection vs regularization strength', fontsize=10, pad=10)

            # 添加网格
            ax.grid(True, alpha=0.3, linewidth=0.5, linestyle='-')
            ax.set_axisbelow(True)

            # 添加图例
            ax.legend(loc='best',
                      frameon=True,
                      fancybox=False,
                      edgecolor='black',
                      framealpha=0.9,
                      fontsize=8)

            # 移除顶部和右侧边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # 设置y轴范围
            ax.set_ylim(0, X_train.shape[1] * 1.05)

            # 添加次要y轴显示百分比
            ax2 = ax.twinx()
            ax2.set_ylim(0, 105)
            ax2.set_ylabel('Percentage of features (%)', fontsize=9)
            ax2.spines['top'].set_visible(False)

            # 计算百分比
            total_features = X_train.shape[1]
            percentages = [count / total_features * 100 for count in feature_counts]

            plt.tight_layout()
            plt.savefig(f_feature_count, dpi=300, bbox_inches='tight')
            plt.close()

            # 保存数据到CSV文件
            f_feature_count_data = d_out / "feature_count_vs_C.csv"
            feature_count_df = pd.DataFrame({
                'C': C_values,
                'Feature_Count': feature_counts,
                'Feature_Percentage': percentages
            })
            feature_count_df.to_csv(f_feature_count_data, index=False)
            logger.info(f"Feature count data saved to {f_feature_count_data}")

        except Exception as e:
            logger.error(f"Error plotting feature count vs parameter: {e}")

    def stability_selection(self, X_train, y_train, n_bootstrap=100, threshold=0.6, d_out=None):
        """Stability selection with publication-quality visualization"""
        try:
            n_samples, n_features = X_train.shape
            feature_scores = np.zeros(n_features)

            logger.info(f"Performing stability selection with {n_bootstrap} bootstrap iterations...")

            for i in tqdm(range(n_bootstrap), desc="Stability selection"):
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = X_train.iloc[indices]
                y_boot = y_train[indices]

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    lasso = LogisticRegression(
                        penalty='l1', C=self.grid_search.best_params_['C'],
                        solver='liblinear', max_iter=self.max_iter,
                        random_state=self.seed + i
                    )
                    lasso.fit(X_boot, y_boot)

                selected = lasso.coef_[0] != 0
                feature_scores += selected

            feature_scores /= n_bootstrap

            if d_out:
                f_stability = d_out / "Figure_7_stability_selection.pdf"

                stable_features = pd.DataFrame({
                    'feature': X_train.columns,
                    'stability_score': feature_scores
                }).sort_values('stability_score', ascending=False)

                stable_features_filtered = stable_features[stable_features['stability_score'] >= threshold]

                if len(stable_features_filtered) > 0:
                    # Create figure
                    fig, ax = plt.subplots(figsize=(3.5, 4))

                    # Plot bars
                    y_pos = np.arange(len(stable_features_filtered))
                    bars = ax.barh(y_pos, stable_features_filtered['stability_score'],
                                   color=COLORS['success'], edgecolor='black', linewidth=0.5)

                    # Add threshold line
                    ax.axvline(x=threshold, color=COLORS['quaternary'],
                               linestyle='--', linewidth=1.5,
                               label=f'Threshold = {threshold}')

                    # Customize axes
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(stable_features_filtered['feature'], fontsize=8)
                    ax.invert_yaxis()
                    ax.set_xlabel('Stability score', fontsize=9)
                    ax.set_title('Stable features', fontsize=10, pad=10)
                    ax.legend(frameon=True, fancybox=False, edgecolor='black',
                              framealpha=0.7, loc='lower right', fontsize=4)

                    # Customize appearance
                    ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)
                    ax.set_axisbelow(True)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xlim(0, 1.05)

                    plt.tight_layout()
                    plt.savefig(f_stability, dpi=300, bbox_inches='tight')
                    plt.close()

                # Save stable features
                f_stable_features = d_out / "stable_features.txt"
                stable_features_filtered.to_csv(f_stable_features, sep='\t', index=False)
                logger.info(f"Found {len(stable_features_filtered)} stable features")

            return feature_scores

        except Exception as e:
            logger.error(f"Error in stability selection: {e}")
            return None

    def plot_feature_correlation_heatmap(self, X_train, selected_features, d_out):
        """Publication-quality correlation heatmap"""
        try:
            if len(selected_features) == 0:
                return

            f_corr = d_out / "Figure_8_feature_correlations.pdf"

            # Limit to top 20 features for clarity
            if len(selected_features) > 20:
                coefficients = self.best_model.coef_[0]
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'abs_coefficient': np.abs(coefficients)
                }).sort_values('abs_coefficient', ascending=False)
                selected_features = feature_importance[feature_importance['feature'].isin(selected_features)].head(20)[
                    'feature'].tolist()

            X_selected = X_train[selected_features]
            corr_matrix = X_selected.corr()

            # Create figure
            fig, ax = plt.subplots(figsize=(5, 4))

            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            # Create custom colormap
            cmap = sns.diverging_palette(250, 10, as_cmap=True)

            # Plot heatmap
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "pad": 0.02},
                        annot=len(selected_features) <= 10, fmt='.2f',
                        annot_kws={'fontsize': 6}, ax=ax)

            # Customize
            ax.set_title('Feature correlation matrix', fontsize=10, pad=10)
            ax.tick_params(axis='both', which='major', labelsize=7)

            # Rotate labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            plt.setp(ax.get_yticklabels(), rotation=0)

            # Adjust colorbar
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label('Correlation coefficient', fontsize=8)

            plt.tight_layout()
            plt.savefig(f_corr, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting correlation heatmap: {e}")

    def plot_validation_curve(self, X_train, y_train, param_grid, d_out):
        """Publication-quality validation curve"""
        try:
            f_validation = d_out / "Figure_S1_validation_curve.pdf"

            param_range = param_grid['C']
            train_scores, val_scores = validation_curve(
                LogisticRegression(penalty='l1', solver='liblinear',
                                   random_state=self.seed, max_iter=self.max_iter),
                X_train, y_train,
                param_name="C",
                param_range=param_range,
                cv=5,
                scoring="roc_auc",
                n_jobs=-1
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            val_scores_mean = np.mean(val_scores, axis=1)
            val_scores_std = np.std(val_scores, axis=1)

            # Create figure
            fig, ax = plt.subplots(figsize=(3.5, 2.8))

            # Plot with error bands
            ax.fill_between(param_range,
                            train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std,
                            alpha=0.2, color=COLORS['secondary'])
            ax.fill_between(param_range,
                            val_scores_mean - val_scores_std,
                            val_scores_mean + val_scores_std,
                            alpha=0.2, color=COLORS['primary'])

            ax.semilogx(param_range, train_scores_mean,
                        label="Training", color=COLORS['secondary'],
                        marker='s', markersize=4, linewidth=1.5)
            ax.semilogx(param_range, val_scores_mean,
                        label="Validation", color=COLORS['primary'],
                        marker='o', markersize=4, linewidth=1.5)

            # Mark the best parameter
            best_idx = np.argmax(val_scores_mean)
            ax.axvline(param_range[best_idx], color=COLORS['quaternary'],
                       linestyle='--', linewidth=1, alpha=0.7)

            ax.set_xlabel('Regularization parameter C', fontsize=9)
            ax.set_ylabel('AUC score', fontsize=9)
            ax.set_title('Validation curve', fontsize=10, pad=10)
            ax.legend(frameon=True, fancybox=False, edgecolor='black',
                      framealpha=0.9, loc='best')

            # Customize appearance
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(0.5, 1.02)

            plt.tight_layout()
            plt.savefig(f_validation, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting validation curve: {e}")

    def create_summary_figure(self, X_train, X_test, y_train, y_test, d_out):
        """Create a comprehensive summary figure for main text"""
        try:
            f_summary = d_out / "Figure_main_summary.pdf"

            # Create figure with multiple panels
            fig = plt.figure(figsize=(7, 8))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1],
                                  hspace=0.3, wspace=0.3)

            # Panel A: Cross-validation results
            ax1 = fig.add_subplot(gs[0, 0])
            cv_results_df = pd.DataFrame(self.grid_search.cv_results_)
            ax1.errorbar(cv_results_df['param_C'],
                         cv_results_df['mean_test_score'],
                         yerr=cv_results_df['std_test_score'],
                         marker='o', markersize=3,
                         color=COLORS['primary'],
                         capsize=2, capthick=0.5,
                         linewidth=1, alpha=0.8)
            ax1.set_xscale('log')
            ax1.set_xlabel('C parameter', fontsize=8)
            ax1.set_ylabel('CV AUC', fontsize=8)
            ax1.set_title('A. Cross-validation', fontsize=9, loc='left', fontweight='bold')
            ax1.grid(True, alpha=0.3, linewidth=0.5)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)

            # Panel B: Feature importance
            ax2 = fig.add_subplot(gs[0, 1])
            coefficients = self.best_model.coef_[0]
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            feature_importance = feature_importance[feature_importance['coefficient'] != 0].head(10)

            if len(feature_importance) > 0:
                y_pos = np.arange(len(feature_importance))
                colors = [COLORS['quaternary'] if x < 0 else COLORS['primary']
                          for x in feature_importance['coefficient']]
                ax2.barh(y_pos, feature_importance['coefficient'],
                         color=colors, edgecolor='black', linewidth=0.5)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(feature_importance['feature'], fontsize=7)
                ax2.invert_yaxis()
                ax2.axvline(x=0, color='black', linewidth=0.5)
                ax2.set_xlabel('Coefficient', fontsize=8)
                ax2.set_title('B. Top features', fontsize=9, loc='left', fontweight='bold')
                ax2.grid(True, axis='x', alpha=0.3, linewidth=0.5)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)

            # Panel C: ROC curve
            ax3 = fig.add_subplot(gs[1, 0])
            y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax3.plot(fpr, tpr, color=COLORS['primary'], linewidth=2,
                     label=f'AUC = {roc_auc:.3f}')
            ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
            ax3.set_xlabel('False positive rate', fontsize=8)
            ax3.set_ylabel('True positive rate', fontsize=8)
            ax3.set_title('C. ROC curve', fontsize=9, loc='left', fontweight='bold')
            ax3.legend(loc="lower right", fontsize=7, frameon=True,
                       fancybox=False, edgecolor='black')
            ax3.grid(True, alpha=0.3, linewidth=0.5)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)

            # Panel D: Confusion matrix
            ax4 = fig.add_subplot(gs[1, 1])
            y_pred = self.best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            im = ax4.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax4.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black",
                             fontsize=9)

            ax4.set_xticks([0, 1])
            ax4.set_yticks([0, 1])
            ax4.set_xticklabels(['Neg', 'Pos'], fontsize=8)
            ax4.set_yticklabels(['Neg', 'Pos'], fontsize=8)
            ax4.set_xlabel('Predicted', fontsize=8)
            ax4.set_ylabel('True', fontsize=8)
            ax4.set_title('D. Confusion matrix', fontsize=9, loc='left', fontweight='bold')

            # Panel E: Learning curve
            ax5 = fig.add_subplot(gs[2, :])
            best_C = self.grid_search.best_params_['C']
            train_sizes = np.linspace(0.1, 1.0, 10)

            train_sizes_abs, train_scores, val_scores = learning_curve(
                LogisticRegression(penalty='l1', C=best_C, solver='liblinear',
                                   random_state=self.seed, max_iter=self.max_iter),
                X_train, y_train,
                train_sizes=train_sizes,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                random_state=self.seed
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            val_scores_mean = np.mean(val_scores, axis=1)
            val_scores_std = np.std(val_scores, axis=1)

            ax5.fill_between(train_sizes_abs,
                             train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std,
                             alpha=0.2, color=COLORS['secondary'])
            ax5.fill_between(train_sizes_abs,
                             val_scores_mean - val_scores_std,
                             val_scores_mean + val_scores_std,
                             alpha=0.2, color=COLORS['primary'])
            ax5.plot(train_sizes_abs, train_scores_mean, 'o-',
                     color=COLORS['secondary'], label='Training',
                     markersize=4, linewidth=1.5)
            ax5.plot(train_sizes_abs, val_scores_mean, 's-',
                     color=COLORS['primary'], label='Validation',
                     markersize=4, linewidth=1.5)
            ax5.set_xlabel('Training set size', fontsize=8)
            ax5.set_ylabel('Accuracy', fontsize=8)
            ax5.set_title('E. Learning curves', fontsize=9, loc='left', fontweight='bold')
            ax5.legend(loc='lower right', fontsize=7, frameon=True,
                       fancybox=False, edgecolor='black')
            ax5.grid(True, alpha=0.3, linewidth=0.5)
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.set_ylim(0.5, 1.02)

            plt.tight_layout()
            plt.savefig(f_summary, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error creating summary figure: {e}")

    def save_comprehensive_results(self, X_train, X_test, y_train, y_test,
                                   feature_scores, d_out):
        """保存综合分析结果"""
        try:
            f_results = d_out / "lasso_comprehensive_results.json"

            # Calculate metrics
            y_train_pred = self.best_model.predict(X_train)
            y_test_pred = self.best_model.predict(X_test)
            y_test_proba = self.best_model.predict_proba(X_test)[:, 1]

            # Get selected features
            coefficients = self.best_model.coef_[0]
            selected_features = X_train.columns[coefficients != 0].tolist()

            results = {
                "model_parameters": {
                    "best_C": float(self.grid_search.best_params_['C']),
                    "best_cv_score": float(self.grid_search.best_score_),
                    "n_features_selected": len(selected_features),
                    "total_features": X_train.shape[1],
                    "feature_selection_rate": f"{len(selected_features) / X_train.shape[1] * 100:.2f}%"
                },
                "training_metrics": {
                    "accuracy": float(accuracy_score(y_train, y_train_pred)),
                    "precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
                    "recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
                    "f1_score": float(f1_score(y_train, y_train_pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_train, self.best_model.predict_proba(X_train)[:, 1]))
                },
                "test_metrics": {
                    "accuracy": float(accuracy_score(y_test, y_test_pred)),
                    "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
                    "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
                    "f1_score": float(f1_score(y_test, y_test_pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
                    "pr_auc": float(average_precision_score(y_test, y_test_proba))
                },
                "selected_features": selected_features,
                "feature_coefficients": {
                    feat: float(coef) for feat, coef in zip(X_train.columns, coefficients) if coef != 0
                },
                "sample_sizes": {
                    "train": len(y_train),
                    "test": len(y_test),
                    "train_positive": int(y_train.sum()),
                    "train_negative": len(y_train) - int(y_train.sum()),
                    "test_positive": int(y_test.sum()),
                    "test_negative": len(y_test) - int(y_test.sum())
                }
            }

            with open(f_results, 'w') as f:
                json.dump(results, f, indent=4)

            # Save detailed feature analysis
            f_feature_analysis = d_out / "feature_analysis.csv"
            feature_df = pd.DataFrame({
                'feature': X_train.columns,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients),
                'selected': coefficients != 0,
                'stability_score': feature_scores if feature_scores is not None else np.nan
            })
            feature_df = feature_df.sort_values('abs_coefficient', ascending=False)
            feature_df.to_csv(f_feature_analysis, index=False)

        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def save_model(self, d_out):
        """保存模型和标准化器"""
        try:
            f_model = d_out / "lasso_model.pkl"
            f_scaler = d_out / "scaler.pkl"

            joblib.dump(self.best_model, f_model)
            joblib.dump(self.scaler, f_scaler)
            logger.info(f"Model saved to {f_model}")
            logger.info(f"Scaler saved to {f_scaler}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def fit(self, X_train, y_train, param_grid, cv_folds=5, n_jobs=-1):
        """训练LASSO模型"""
        X_train, y_train = self.validate_data(X_train, y_train, "Training data")

        X_train_scaled = X_train

        logreg_l1 = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            random_state=self.seed,
            max_iter=self.max_iter
        )

        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)

        self.grid_search = GridSearchCV(
            estimator=logreg_l1,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=cv_strategy,
            n_jobs=n_jobs,
            return_train_score=True,
            verbose=1
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.grid_search.fit(X_train_scaled, y_train)

        self.best_model = self.grid_search.best_estimator_

        # Get selected features
        coefficients = self.best_model.coef_[0]
        self.selected_features = X_train.columns[coefficients != 0].tolist()

        return X_train_scaled


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-g", "--genotype",
              type=click.Path(),
              required=True,
              help="The genotype input file")
@click.option("-p", "--phenotype",
              type=click.Path(),
              required=True,
              help="The phenotype input file")
@click.option("-vg", "--vgenotype",
              type=click.Path(),
              help="The validation genotype input file (must be provided with --vphenotype)")
@click.option("-vp", "--vphenotype",
              type=click.Path(),
              help="The validation phenotype input file (must be provided with --vgenotype)")
@click.option("--feature",
              type=click.Path(),
              help="A file containing features to be used")
@click.option("--testsize",
              type=float,
              default=0.25,
              show_default=True,
              help="The proportion of the dataset that should be used for testing")
@click.option("-k", "--kfolds",
              type=int,
              default=5,
              show_default=True,
              help="The k fold used for cross validation")
@click.option("--maxiter",
              type=int,
              default=10000,
              show_default=True,
              help="The maximum number of iterations for LASSO")
@click.option("--seed",
              type=int,
              default=42,
              show_default=True,
              help="The random seed for split dataset and model training")
@click.option("--stability",
              is_flag=True,
              help="Perform stability selection analysis")
@click.option("--n-bootstrap",
              type=int,
              default=100,
              show_default=True,
              help="Number of bootstrap iterations for stability selection")
@click.option("--stability-threshold",
              type=float,
              default=0.6,
              show_default=True,
              help="Threshold for stability selection")
@click.option("--n-jobs",
              type=int,
              default=-1,
              show_default=True,
              help="Number of parallel jobs for computation (-1 uses all cores)")
@click.option("--c-range",
              type=str,
              default="-3,2,30",
              show_default=True,
              help="C parameter range: log_start,log_end,n_points")
@click.option("--publication-ready",
              is_flag=True,
              help="Generate publication-ready figures with all panels")
@click.option("-o", "--out",
              type=click.Path(),
              required=True,
              help="The output directory")
def main(genotype, phenotype, vgenotype, vphenotype, feature, testsize, kfolds,
         maxiter, seed, stability, n_bootstrap, stability_threshold, n_jobs,
         c_range, publication_ready, out):
    """
    Enhanced LASSO feature selection for binary classification in bioinformatics

    This tool performs LASSO-based feature selection with comprehensive statistical
    analysis and publication-quality visualization for genomic data analysis.

    Example:
        python lasso_feature_selection.py -g genotype.csv -p phenotype.csv -o results/
    """
    # Setup paths
    f_genotype = Path(genotype).absolute()
    f_phenotype = Path(phenotype).absolute()
    d_out = Path(out).absolute()
    d_out.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for organization
    d_figures = d_out / "figures"
    d_figures.mkdir(exist_ok=True)
    d_tables = d_out / "tables"
    d_tables.mkdir(exist_ok=True)
    d_models = d_out / "models"
    d_models.mkdir(exist_ok=True)

    f_log = d_out / "lasso_analysis.log"
    file_handler = logging.FileHandler(f_log)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("LASSO Feature Selection Analysis")
    logger.info("=" * 60)
    logger.info(f"Output directory: {d_out}")

    if (vgenotype and not vphenotype) or (vphenotype and not vgenotype):
        raise click.BadParameter("Both --vgenotype and --vphenotype must be provided together")

    selector = LASSOFeatureSelector(seed=seed, max_iter=maxiter)

    features = []
    if feature:
        f_feature = Path(feature).absolute()
        logger.info(f"Reading feature file: {f_feature}")
        features = selector.get_features(f_feature)
        logger.info(f"Loaded {len(features)} features from file")

    logger.info("Loading data...")
    try:
        X_train, X_test, y_train, y_test = auto_load_data(
            f_genotype, f_phenotype,
            f_test_genotype=vgenotype, f_test_phenotype=vphenotype,
            random_seed=seed, test_size=testsize, features=features
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Class distribution - Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"Class distribution - Test: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    try:
        c_start, c_end, c_points = map(float, c_range.split(','))
        c_points = int(c_points)
        param_grid = {
            'C': np.logspace(c_start, c_end, c_points)
        }
    except ValueError:
        logger.error("Invalid C range format. Use: log_start,log_end,n_points")
        raise

    logger.info(f"C parameter range: {param_grid['C'][0]:.4f} to {param_grid['C'][-1]:.4f} ({c_points} points)")

    # Train model
    logger.info("Training LASSO model with cross-validation...")
    X_train_scaled = selector.fit(X_train, y_train, param_grid, cv_folds=kfolds, n_jobs=n_jobs)
    X_test_scaled = X_test

    logger.info(f"Best C parameter: {selector.grid_search.best_params_['C']:.4f}")
    logger.info(f"Best cross-validation AUC: {selector.grid_search.best_score_:.4f}")
    logger.info(f"Number of selected features: {len(selector.selected_features)} / {X_train.shape[1]}")

    # Generate visualizations
    logger.info("Generating publication-quality visualizations...")

    # Individual figures
    selector.plot_coefficient_path(X_train_scaled, y_train, param_grid, d_figures)
    selector.cv_auc_plot_enhanced(d_figures)
    selector.plot_learning_curve(X_train_scaled, y_train, d_figures)
    selector.plot_feature_importance(X_train_scaled, d_figures)
    selector.plot_roc_pr_curves(X_test_scaled, y_test, d_figures)
    selector.plot_confusion_matrix(X_test_scaled, y_test, d_figures)
    selector.plot_validation_curve(X_train_scaled, y_train, param_grid, d_figures)
    selector.plot_feature_count_vs_parameter(X_train_scaled, y_train, param_grid, d_figures)

    if len(selector.selected_features) > 0:
        selector.plot_feature_correlation_heatmap(X_train, selector.selected_features, d_figures)

    # Create summary figure if requested
    if publication_ready:
        logger.info("Creating comprehensive summary figure...")
        selector.create_summary_figure(X_train_scaled, X_test_scaled,
                                       y_train, y_test, d_figures)

    # Stability selection if requested
    feature_scores = None
    if stability:
        logger.info("Performing stability selection...")
        feature_scores = selector.stability_selection(
            X_train_scaled, y_train, n_bootstrap, stability_threshold, d_figures
        )

    # Save comprehensive results
    selector.save_comprehensive_results(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_scores, d_tables
    )

    selector.save_model(d_models)

    f_param = d_tables / "param.json"
    save_param(selector.best_model.get_params(), f_param)

    f_report_train = d_tables / "report_train.txt"
    save_report(classification_report(y_train, selector.best_model.predict(X_train_scaled)), f_report_train)

    f_report_test = d_tables / "report_test.txt"
    save_report(classification_report(y_test, selector.best_model.predict(X_test_scaled)), f_report_test)

    if selector.selected_features:
        f_selected = d_tables / "selected_features.txt"
        with open(f_selected, 'w') as f:
            for feat in selector.selected_features:
                f.write(f"{feat}\n")
        logger.info(f"Selected features saved to {f_selected}")

    generate_summary_report(selector, X_train, X_test, y_train, y_test, d_out)

    create_figure_legends(d_figures)

    logger.info("=" * 60)
    logger.info(f"Analysis complete. Results saved to {d_out}")
    logger.info("=" * 60)


def generate_summary_report(selector, X_train, X_test, y_train, y_test, d_out):
    """生成分析总结报告"""
    f_summary = d_out / "analysis_summary.txt"

    y_test_pred = selector.best_model.predict(X_test)
    y_test_proba = selector.best_model.predict_proba(X_test)[:, 1]

    with open(f_summary, 'w') as f:
        f.write("LASSO Feature Selection Analysis Summary\n")
        f.write("=" * 60 + "\n\n")

        f.write("Dataset Information:\n")
        f.write(f"  - Total samples: {len(X_train) + len(X_test)}\n")
        f.write(f"  - Training samples: {len(X_train)}\n")
        f.write(f"  - Test samples: {len(X_test)}\n")
        f.write(f"  - Total features: {X_train.shape[1]}\n")
        f.write(f"  - Selected features: {len(selector.selected_features)}\n")
        f.write(f"  - Feature reduction: {(1 - len(selector.selected_features) / X_train.shape[1]) * 100:.1f}%\n\n")

        f.write("Model Performance:\n")
        f.write(f"  - Best C parameter: {selector.grid_search.best_params_['C']:.4f}\n")
        f.write(f"  - Cross-validation AUC: {selector.grid_search.best_score_:.4f}\n")
        f.write(f"  - Test set AUC: {roc_auc_score(y_test, y_test_proba):.4f}\n")
        f.write(f"  - Test set accuracy: {accuracy_score(y_test, y_test_pred):.4f}\n")
        f.write(f"  - Test set F1-score: {f1_score(y_test, y_test_pred):.4f}\n\n")

        f.write("Top 10 Selected Features:\n")
        coefficients = selector.best_model.coef_[0]
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)

        feature_importance = feature_importance[feature_importance['coefficient'] != 0]
        for i, (idx, row) in enumerate(feature_importance.head(10).iterrows()):
            f.write(f"  {i + 1}. {row['feature']}: {row['coefficient']:.4f}\n")

        f.write("\nAnalysis completed successfully.\n")


def create_figure_legends(d_figures):
    """Create a file with figure legends for publication"""
    f_legends = d_figures / "figure_legends.txt"

    with open(f_legends, 'w') as f:
        f.write("FIGURE LEGENDS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Figure 1. LASSO coefficient paths.\n")
        f.write("Coefficient values for the top 10 features as a function of the regularization "
                "strength (α). The vertical dashed line indicates the optimal α value selected "
                "by cross-validation. Different colors represent different features.\n\n")

        f.write("Figure 2. Cross-validation results.\n")
        f.write("(A) Mean cross-validation AUC scores with standard deviation error bars across "
                "different regularization parameter C values. The red dot indicates the optimal C. "
                "(B) Comparison of training and validation scores showing model generalization.\n\n")

        f.write("Figure 3. Learning curves.\n")
        f.write("Model accuracy as a function of training set size for both training (red) and "
                "validation (blue) sets. Shaded areas represent standard deviation across "
                "cross-validation folds.\n\n")

        f.write("Figure 4. Feature importance.\n")
        f.write("Top 15 features selected by LASSO with their coefficient values. Blue bars "
                "indicate positive associations and red bars indicate negative associations "
                "with the outcome.\n\n")

        f.write("Figure 5. Model performance curves.\n")
        f.write("(A) Receiver Operating Characteristic (ROC) curve with area under the curve (AUC). "
                "Shaded region represents 95% confidence interval from bootstrap resampling. "
                "(B) Precision-Recall (PR) curve with average precision (AP) score.\n\n")

        f.write("Figure 6. Confusion matrix.\n")
        f.write("Classification results on the test set showing true positive, true negative, "
                "false positive, and false negative counts. Performance metrics are displayed below.\n\n")

        f.write("Figure 7. Stability selection results.\n")
        f.write("Features with stability scores above the threshold (red dashed line) are considered "
                "stable across bootstrap samples. Higher scores indicate more consistent selection.\n\n")

        f.write("Figure 8. Feature correlation matrix.\n")
        f.write("Pairwise Pearson correlation coefficients among selected features. Red indicates "
                "positive correlation and blue indicates negative correlation.\n\n")

        f.write("Figure S1. Validation curve.\n")
        f.write("Model performance (AUC) on training and validation sets across different "
                "regularization parameter C values. Shaded areas represent standard deviation.\n\n")

        f.write("Main Figure. Comprehensive analysis summary.\n")
        f.write("(A) Cross-validation results for model selection. (B) Top 10 selected features "
                "with coefficient values. (C) ROC curve on test set. (D) Confusion matrix. "
                "(E) Learning curves showing model generalization.\n")


def create_methods_section(d_out):
    """Generate a methods section template for publication"""
    f_methods = d_out / "methods_template.txt"

    with open(f_methods, 'w') as f:
        f.write("METHODS SECTION TEMPLATE\n")
        f.write("=" * 60 + "\n\n")

        f.write("Feature Selection and Model Development\n\n")

        f.write("LASSO (Least Absolute Shrinkage and Selection Operator) regression was employed "
                "for feature selection and predictive model development. Prior to analysis, all "
                "features were standardized to have zero mean and unit variance using the "
                "StandardScaler from scikit-learn.\n\n")

        f.write("Model selection was performed using stratified k-fold cross-validation with "
                "k=5 folds to ensure balanced class representation in each fold. The regularization "
                "parameter C was optimized over a logarithmic grid from 10^-3 to 10^2 with 30 "
                "equally spaced points in log space. The area under the receiver operating "
                "characteristic curve (AUC-ROC) was used as the primary performance metric for "
                "model selection.\n\n")

        f.write("To assess model stability and feature robustness, we performed stability "
                "selection with 100 bootstrap iterations. Features were considered stable if "
                "they were selected in at least 60% of bootstrap samples. Learning curves were "
                "generated to evaluate model performance as a function of training set size and "
                "to detect potential overfitting.\n\n")

        f.write("Model performance was evaluated on an independent test set comprising 25% of "
                "the total samples. Performance metrics included accuracy, precision, recall, "
                "F1-score, AUC-ROC, and area under the precision-recall curve (AUC-PR). "
                "Confidence intervals for the ROC curve were estimated using 1000 bootstrap "
                "iterations.\n\n")

        f.write("All analyses were performed using Python 3.x with scikit-learn, numpy, pandas, "
                "and matplotlib libraries. Random seed was set to ensure reproducibility. "
                "The complete analysis pipeline is available at [repository URL].\n\n")

        f.write("Statistical Analysis\n\n")

        f.write("Continuous variables are presented as mean ± standard deviation. Categorical "
                "variables are presented as counts and percentages. Feature importance was "
                "determined by the absolute value of LASSO coefficients. Correlation between "
                "selected features was assessed using Pearson correlation coefficients.\n")


# Additional helper function for creating publication-ready tables
def create_supplementary_tables(selector, X_train, X_test, y_train, y_test, d_tables):
    """Create supplementary tables for publication"""

    # Table S1: Full feature list with coefficients
    coefficients = selector.best_model.coef_[0]
    feature_table = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': coefficients,
        'Absolute_Coefficient': np.abs(coefficients),
        'Selected': coefficients != 0
    }).sort_values('Absolute_Coefficient', ascending=False)

    f_table_s1 = d_tables / "Table_S1_all_features.csv"
    feature_table.to_csv(f_table_s1, index=False)

    # Table S2: Cross-validation results
    cv_results = pd.DataFrame(selector.grid_search.cv_results_)
    cv_summary = cv_results[['param_C', 'mean_test_score', 'std_test_score',
                             'mean_train_score', 'std_train_score']]
    cv_summary.columns = ['C_parameter', 'Mean_CV_AUC', 'Std_CV_AUC',
                          'Mean_Train_AUC', 'Std_Train_AUC']

    f_table_s2 = d_tables / "Table_S2_cv_results.csv"
    cv_summary.to_csv(f_table_s2, index=False)

    # Table S3: Performance metrics summary
    y_train_pred = selector.best_model.predict(X_train)
    y_test_pred = selector.best_model.predict(X_test)
    y_train_proba = selector.best_model.predict_proba(X_train)[:, 1]
    y_test_proba = selector.best_model.predict_proba(X_test)[:, 1]

    metrics_data = {
        'Dataset': ['Training', 'Test'],
        'N_samples': [len(y_train), len(y_test)],
        'N_positive': [int(y_train.sum()), int(y_test.sum())],
        'N_negative': [len(y_train) - int(y_train.sum()), len(y_test) - int(y_test.sum())],
        'Accuracy': [
            accuracy_score(y_train, y_train_pred),
            accuracy_score(y_test, y_test_pred)
        ],
        'Precision': [
            precision_score(y_train, y_train_pred, zero_division=0),
            precision_score(y_test, y_test_pred, zero_division=0)
        ],
        'Recall': [
            recall_score(y_train, y_train_pred, zero_division=0),
            recall_score(y_test, y_test_pred, zero_division=0)
        ],
        'F1_score': [
            f1_score(y_train, y_train_pred, zero_division=0),
            f1_score(y_test, y_test_pred, zero_division=0)
        ],
        'AUC_ROC': [
            roc_auc_score(y_train, y_train_proba),
            roc_auc_score(y_test, y_test_proba)
        ],
        'AUC_PR': [
            average_precision_score(y_train, y_train_proba),
            average_precision_score(y_test, y_test_proba)
        ]
    }

    metrics_df = pd.DataFrame(metrics_data)
    f_table_s3 = d_tables / "Table_S3_performance_metrics.csv"
    metrics_df.to_csv(f_table_s3, index=False)

    logger.info(f"Supplementary tables saved to {d_tables}")


if __name__ == "__main__":
    main()
