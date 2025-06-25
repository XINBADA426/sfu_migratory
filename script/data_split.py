#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: CB Ren
# @Created Date:   2025/5/29 17:32
from pathlib import Path
import logging
import click

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, Union, List, Tuple
import matplotlib.pyplot as plt
from scipy import stats

import matplotlib.font_manager as fm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

#### Some setting ####
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
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Define color palette for publication (matching the provided script)
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


#### Some Functions ####
class FontToolsSubsetFilter(logging.Filter):
    def __init__(self, name=''):
        super().__init__(name)
        self.unwanted_phrases = [
            "subsetting not needed",
            "subsetted"
        ]

    def filter(self, record):
        # Only apply to 'fontTools.subset' logger
        if record.name == 'fontTools.subset':
            # Check if the message contains any of the unwanted phrases
            if any(phrase in record.getMessage() for phrase in self.unwanted_phrases):
                return False  # Suppress this log message
        return True  # Allow other messages


def create_figure_legends(d_figures):
    """Create a file with figure legends for publication"""
    f_legends = d_figures / "figure_legends.txt"

    with open(f_legends, 'w') as f:
        f.write("FIGURE LEGENDS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Figure 1. Dataset split summary.\n")
        f.write("Bar chart showing the number of samples in training and test sets. "
                "Values above bars indicate sample counts and percentages of the total dataset. "
                "Blue bars represent training set and rose bars represent test set.\n\n")

        f.write("Figure 2. Target variable distribution across splits.\n")
        f.write("(A) Absolute sample counts for each class in training (rose) and test (blue) sets. "
                "(B) Class proportions in training and test sets with chi-square test p-value "
                "indicating the statistical similarity of distributions. Error bars represent "
                "standard error of proportions.\n\n")

        f.write("Figure 3. Stratification variable distributions.\n")
        f.write("Comparison of stratification variable proportions between training (rose) and "
                "test (blue) sets. Each panel shows the distribution of a different stratification "
                "variable used to ensure balanced splits. Categories are shown on x-axis with "
                "proportions on y-axis.\n\n")

        f.write("Figure 4. Stratification balance assessment.\n")
        f.write("Proportion differences (Training - Test) for each category of stratification variables. "
                "Green bars indicate higher proportion in training set, red bars indicate higher "
                "proportion in test set. Dashed lines at ±5% indicate acceptable balance threshold. "
                "Values closer to zero indicate better balance between splits.\n\n")

        f.write("Supplementary Figure S1. Test size analysis.\n")
        f.write("Analysis of how different test set proportions affect (left) sample sizes and "
                "(right) maximum class proportion differences between training and test sets. "
                "The red dashed line indicates the 5% difference threshold. This analysis helps "
                "determine optimal test set size for maintaining class balance.\n\n")

        f.write("Note: All figures were generated using matplotlib with publication-ready settings. "
                "Statistical tests were performed using scipy.stats. Sample sizes (n) are indicated "
                "in figure titles or labels where applicable.\n")


def create_methods_section(d_out):
    """Generate a methods section template for publication"""
    f_methods = d_out / "methods_template.txt"

    with open(f_methods, 'w') as f:
        f.write("METHODS SECTION TEMPLATE\n")
        f.write("=" * 60 + "\n\n")

        f.write("Data Splitting and Stratification\n\n")

        f.write("The dataset was divided into training and test sets using a stratified random "
                "sampling approach to ensure representative distributions across all subsets. "
                "The splitting procedure was implemented using scikit-learn's train_test_split "
                "function with the following specifications:\n\n")

        f.write("• Test set proportion: 20% of the total dataset (or as specified)\n")
        f.write("• Random seed: Fixed at 42 for reproducibility\n")
        f.write("• Stratification strategy: Multi-variable stratification including target variable "
                "and additional covariates\n\n")

        f.write("Stratification Variables\n\n")

        f.write("To maintain balanced distributions across training and test sets, we employed "
                "a multi-condition stratification approach. The stratification included:\n\n")

        f.write("1. Target variable: Ensured proportional representation of all classes\n")
        f.write("2. Additional covariates: When provided, variables such as batch, site, or other "
                "technical/biological factors were included in the stratification scheme\n\n")

        f.write("For multi-variable stratification, a composite key was created by concatenating "
                "all stratification variables. This ensured that the combination of all specified "
                "conditions was proportionally represented in both training and test sets.\n\n")

        f.write("Quality Control and Validation\n\n")

        f.write("Prior to splitting, several quality control steps were performed:\n\n")

        f.write("• Sample alignment: Ensured that genotype and phenotype data contained identical "
                "samples in the same order\n")
        f.write("• Stratification feasibility: Verified that each stratification category contained "
                "sufficient samples (minimum 2) for splitting\n")
        f.write("• Class balance assessment: Chi-square tests were performed to confirm that class "
                "distributions were statistically similar between training and test sets\n\n")

        f.write("When stratification was not feasible due to small category sizes, the algorithm "
                "automatically fell back to simple random sampling with a warning message.\n\n")

        f.write("Data Output and Documentation\n\n")

        f.write("The split datasets were saved in tab-separated value (TSV) format with the "
                "following structure:\n\n")

        f.write("• X_train.tsv, X_test.tsv: Feature matrices with samples as rows and features as columns\n")
        f.write("• y_train.tsv, y_test.tsv: Target variables with sample identifiers\n")
        f.write("• train_samples.tsv, test_samples.tsv: Lists of sample identifiers for each set\n")
        f.write("• split_metadata.json: Comprehensive metadata including split ratios, sample counts, "
                "and class distributions\n\n")

        f.write("Statistical Analysis\n\n")

        f.write("Distribution comparisons between training and test sets were evaluated using:\n\n")

        f.write("• Chi-square tests for categorical variables to assess distribution similarity\n")
        f.write("• Proportion differences calculated as (train proportion - test proportion) for "
                "each category\n")
        f.write("• A 5% threshold was used to identify potentially imbalanced categories\n\n")

        f.write("Visualization\n\n")

        f.write("Publication-quality figures were generated to visualize:\n\n")

        f.write("• Overall split summary with sample counts and percentages\n")
        f.write("• Class distributions in training and test sets\n")
        f.write("• Stratification variable distributions across splits\n")
        f.write("• Balance assessment showing proportion differences\n\n")

        f.write("All visualizations were created using matplotlib with standardized formatting "
                "for publication readiness, including appropriate color schemes, font sizes, "
                "and figure dimensions.\n\n")

        f.write("Software and Dependencies\n\n")

        f.write("The analysis was performed using Python 3.x with the following key packages:\n")
        f.write("• pandas (v.X.X.X) for data manipulation\n")
        f.write("• scikit-learn (v.X.X.X) for train_test_split functionality\n")
        f.write("• numpy (v.X.X.X) for numerical operations\n")
        f.write("• matplotlib (v.X.X.X) for visualization\n")
        f.write("• scipy (v.X.X.X) for statistical tests\n\n")

        f.write("The complete code for data splitting and analysis is available at [repository URL].\n")


def create_publication_figures(stats_data: dict, output_dir: Path) -> None:
    """
    Create publication-quality figures for data split statistics

    Parameters:
    -----------
    stats_data : dict
        Dictionary containing statistics dataframes
    output_dir : Path
        Output directory for saving figures
    """
    figures_path = output_dir / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)

    # 1. Create split summary figure
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    split_summary = stats_data['split_summary']
    train_test_data = split_summary[split_summary['dataset'].isin(['train', 'test'])]

    bars = ax.bar(train_test_data['dataset'],
                  train_test_data['n_samples'],
                  color=[COLORS['primary'], COLORS['secondary']],
                  edgecolor='black',
                  linewidth=0.5,
                  alpha=0.8)

    # Add value labels on bars
    for bar, pct in zip(bars, train_test_data['percentage']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + max(train_test_data['n_samples']) * 0.01,
                f'{int(height)}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Number of samples', fontsize=9)
    ax.set_xlabel('Dataset', fontsize=9)
    ax.set_title('Dataset split summary', fontsize=10, pad=10)

    # Customize grid and spines (matching the style)
    ax.grid(True, alpha=0.3, linewidth=0.5, linestyle='-')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set y-axis limit with some padding
    ax.set_ylim(0, max(train_test_data['n_samples']) * 1.15)

    plt.tight_layout()
    fig.savefig(figures_path / 'Figure_1_split_summary.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Create target distribution figure if available
    if 'target_distribution' in stats_data:
        target_dist = stats_data['target_distribution']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

        # Subplot 1: Absolute counts
        x = np.arange(len(target_dist))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, target_dist['train_count'],
                        width, label='Training', color=COLORS['secondary'],
                        edgecolor='black', linewidth=0.5, alpha=0.8)
        bars2 = ax1.bar(x + width / 2, target_dist['test_count'],
                        width, label='Test', color=COLORS['primary'],
                        edgecolor='black', linewidth=0.5, alpha=0.8)

        ax1.set_xlabel('Class', fontsize=9)
        ax1.set_ylabel('Number of samples', fontsize=9)
        ax1.set_title('A. Sample distribution', fontsize=10, loc='left', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(target_dist['class'])
        ax1.legend(frameon=True, fancybox=False, edgecolor='black',
                   framealpha=0.9, loc='best')

        # Customize appearance
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.set_axisbelow(True)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Subplot 2: Proportions with chi-square test
        train_props = target_dist['train_proportion'].values
        test_props = target_dist['test_proportion'].values

        bars3 = ax2.bar(x - width / 2, train_props,
                        width, label='Training', color=COLORS['secondary'],
                        edgecolor='black', linewidth=0.5, alpha=0.8)
        bars4 = ax2.bar(x + width / 2, test_props,
                        width, label='Test', color=COLORS['primary'],
                        edgecolor='black', linewidth=0.5, alpha=0.8)

        ax2.set_xlabel('Class', fontsize=9)
        ax2.set_ylabel('Proportion', fontsize=9)
        ax2.set_title('B. Class balance', fontsize=10, loc='left', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(target_dist['class'])
        ax2.set_ylim(0, max(max(train_props), max(test_props)) * 1.2)

        # Perform chi-square test
        chi2, p_value = stats.chisquare(target_dist['test_count'],
                                        f_exp=target_dist['train_count'] *
                                              (target_dist['test_count'].sum() /
                                               target_dist['train_count'].sum()))

        # Add chi-square test result
        props = dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='black', linewidth=0.5, alpha=0.9)
        ax2.text(0.95, 0.95, f'χ² test\np = {p_value:.3f}',
                 transform=ax2.transAxes, ha='right', va='top',
                 fontsize=7, bbox=props)

        # Customize appearance
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        ax2.set_axisbelow(True)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()
        fig.savefig(figures_path / 'Figure_2_target_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Create stratification variable distributions
    strat_fig_num = 3
    for key in stats_data:
        if key.endswith('_distribution') and key != 'target_distribution':
            var_name = key.replace('_distribution', '')
            var_dist = stats_data[key]

            # Create figure with appropriate size based on number of categories
            n_categories = len(var_dist)
            fig_width = max(3.5, min(7, n_categories * 0.5))
            fig, ax = plt.subplots(figsize=(fig_width, 2.8))

            x = np.arange(len(var_dist))
            width = 0.35

            bars1 = ax.bar(x - width / 2, var_dist['train_proportion'],
                           width, label='Training', color=COLORS['secondary'],
                           edgecolor='black', linewidth=0.5, alpha=0.8)
            bars2 = ax.bar(x + width / 2, var_dist['test_proportion'],
                           width, label='Test', color=COLORS['primary'],
                           edgecolor='black', linewidth=0.5, alpha=0.8)

            ax.set_xlabel(var_name.capitalize(), fontsize=9)
            ax.set_ylabel('Proportion', fontsize=9)
            ax.set_title(f'{var_name.capitalize()} distribution across splits', fontsize=10, pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(var_dist['category'], rotation=45 if n_categories > 5 else 0,
                               ha='right' if n_categories > 5 else 'center')
            ax.legend(frameon=True, fancybox=False, edgecolor='black',
                      framealpha=0.9, loc='best')

            # Customize appearance
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(0, max(max(var_dist['train_proportion']), max(var_dist['test_proportion'])) * 1.2)

            plt.tight_layout()
            fig.savefig(figures_path / f'Figure_{strat_fig_num}_{var_name}_distribution.pdf',
                        dpi=300, bbox_inches='tight')
            plt.close()
            strat_fig_num += 1

    # 4. Create combined stratification balance plot if multiple variables
    if len([k for k in stats_data if k.endswith('_distribution')]) > 1:
        create_stratification_balance_plot(stats_data, figures_path, strat_fig_num)

    create_figure_legends(figures_path)
    logger.info(f"PDF figures saved to: {figures_path}")


def create_stratification_balance_plot(stats_data: dict, figures_path: Path, fig_num: int) -> None:
    """
    Create a balance plot showing the distribution differences across stratification variables
    """
    # Collect all stratification variables
    strat_vars = []
    for key in stats_data:
        if key.endswith('_distribution'):
            var_name = key.replace('_distribution', '')
            strat_vars.append({
                'name': var_name,
                'data': stats_data[key]
            })

    if len(strat_vars) < 2:
        return

    # Create figure with subplots
    n_vars = len(strat_vars)
    fig_width = min(7, n_vars * 2.5)
    fig, axes = plt.subplots(1, n_vars, figsize=(fig_width, 2.8))

    if n_vars == 2:
        axes = [axes]

    for idx, var in enumerate(strat_vars):
        ax = axes[idx] if n_vars > 1 else axes

        # Calculate proportion differences
        data = var['data']
        diff = data['train_proportion'] - data['test_proportion']

        # Create bar plot of differences
        colors_diff = [COLORS['success'] if d >= 0 else COLORS['quaternary'] for d in diff]
        bars = ax.bar(range(len(diff)), diff, color=colors_diff,
                      edgecolor='black', linewidth=0.5, alpha=0.8)

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add threshold lines
        ax.axhline(y=0.05, color=COLORS['neutral'], linestyle='--',
                   linewidth=0.5, alpha=0.5, label='±5% threshold')
        ax.axhline(y=-0.05, color=COLORS['neutral'], linestyle='--',
                   linewidth=0.5, alpha=0.5)

        ax.set_xlabel(var['name'].capitalize(), fontsize=9)
        if idx == 0:
            ax.set_ylabel('Proportion difference\n(Train - Test)', fontsize=9)
        ax.set_title(f'{chr(65 + idx)}. {var["name"].capitalize()}',
                     fontsize=10, loc='left', pad=10)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data['category' if 'category' in data.columns else 'class'],
                           rotation=45, ha='right', fontsize=5)

        # Customize appearance
        ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(frameon=True, fancybox=False, edgecolor='black',
                      framealpha=0.9, loc='best', fontsize=5)

    plt.tight_layout()
    fig.savefig(figures_path / f'Figure_{fig_num}_stratification_balance.pdf',
                dpi=300, bbox_inches='tight')
    plt.close()


def create_sample_size_analysis(X: pd.DataFrame, y: pd.Series,
                                output_dir: Path, test_sizes: List[float] = None) -> None:
    """
    Create analysis of how different test sizes affect class balance
    """
    if test_sizes is None:
        test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

    figures_path = output_dir / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)

    # Calculate class imbalance for different test sizes
    results = []
    for test_size in test_sizes:
        try:
            _, _, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            train_dist = y_train.value_counts(normalize=True)
            test_dist = y_test.value_counts(normalize=True)

            # Calculate maximum proportion difference
            max_diff = 0
            for cls in train_dist.index:
                diff = abs(train_dist[cls] - test_dist.get(cls, 0))
                max_diff = max(max_diff, diff)

            results.append({
                'test_size': test_size,
                'max_proportion_diff': max_diff,
                'n_train': len(y_train),
                'n_test': len(y_test)
            })
        except:
            continue

    if not results:
        return

    results_df = pd.DataFrame(results)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))

    # Plot 1: Sample sizes
    ax1.plot(results_df['test_size'], results_df['n_train'],
             'o-', color='#1f77b4', label='Train', markersize=4, linewidth=1)
    ax1.plot(results_df['test_size'], results_df['n_test'],
             'o-', color='#ff7f0e', label='Test', markersize=4, linewidth=1)
    ax1.set_xlabel('Test size proportion', fontsize=8)
    ax1.set_ylabel('Number of samples', fontsize=8)
    ax1.legend(frameon=False, fontsize=7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot 2: Class balance
    ax2.plot(results_df['test_size'], results_df['max_proportion_diff'],
             'o-', color='#2ca02c', markersize=4, linewidth=1)
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Test size proportion', fontsize=8)
    ax2.set_ylabel('Max class proportion\ndifference', fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(figures_path / 'test_size_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()


#### Enhanced balanced_train_test_split function ####
def balanced_train_test_split(
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        stratify_data: Optional[pd.DataFrame] = None,
        stratify_cols: Optional[Union[str, List[str]]] = None,
        include_y: bool = True,
        output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets with multi-condition stratified sampling

    Parameters:
    -----------
    X : pd.DataFrame
        Feature data
    y : pd.Series or pd.DataFrame
        Target variable
    test_size : float, default=0.2
        Proportion of test set
    random_state : int, optional
        Random seed
    stratify_data : pd.DataFrame, optional
        External data containing stratification columns
    stratify_cols : str or list of str, optional
        Column names for stratification (from stratify_data)
    include_y : bool, default=True
        Whether to include y in stratification conditions
    output_dir : str, optional
        Output directory for saving statistics

    Returns:
    --------
    X_train, X_test, y_train, y_test : Split datasets
    """
    # Ensure y is Series format
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        else:
            raise ValueError("y should be one-dimensional data")

    # Build stratification data
    stratify_components = []

    # 1. Whether to include target variable
    if include_y:
        stratify_components.append(y.astype(str))

    # 2. Add stratification columns from external data
    if stratify_data is not None and stratify_cols is not None:
        if isinstance(stratify_cols, str):
            stratify_cols = [stratify_cols]

        # Ensure indices match
        if not X.index.equals(stratify_data.index):
            common_idx = X.index.intersection(stratify_data.index)
            if len(common_idx) < len(X):
                raise ValueError(f"Index mismatch: X has {len(X)} samples, "
                                 f"but only {len(common_idx)} match with stratify_data")
            stratify_data = stratify_data.loc[X.index]

        for col in stratify_cols:
            if col in stratify_data.columns:
                stratify_components.append(stratify_data[col].astype(str))
            else:
                raise ValueError(f"Column '{col}' not found in stratification data")

    # Combine all stratification conditions
    stratify_combined = None
    if len(stratify_components) > 0:
        if len(stratify_components) == 1:
            stratify_combined = stratify_components[0]
        else:
            # Create combined key
            stratify_combined = stratify_components[0].str.cat(
                stratify_components[1:], sep='_'
            )

        # Check stratification feasibility
        value_counts = stratify_combined.value_counts()
        min_samples = value_counts.min()
        n_test_samples = int(len(X) * test_size)

        # Check if each category has enough samples
        if min_samples < 2:
            logger.warning(f"Some category combinations have only {min_samples} sample(s), cannot stratify")
            logger.warning(f"Problematic categories:\n{value_counts[value_counts < 2]}")
            stratify_combined = None
        elif min_samples < n_test_samples / len(value_counts):
            logger.warning("Some categories may not have enough samples to ensure representation in test set")

    # Perform data split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_combined
        )
    except ValueError as e:
        logger.warning(f"Stratified sampling failed: {str(e)}")
        logger.info("Falling back to random sampling...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )

    # Collect statistics for saving
    stats_data = {}

    # Basic split statistics
    stats_data['split_summary'] = pd.DataFrame({
        'dataset': ['train', 'test', 'total'],
        'n_samples': [len(X_train), len(X_test), len(X)],
        'percentage': [len(X_train) / len(X) * 100, len(X_test) / len(X) * 100, 100.0]
    })

    # Print distribution information
    logger.info("Dataset split results:")
    logger.info(f"Training set: {len(X_train)} samples ({len(X_train) / len(X) * 100:.1f}%)")
    logger.info(f"Test set: {len(X_test)} samples ({len(X_test) / len(X) * 100:.1f}%)")

    # Target variable distribution
    if include_y:
        logger.info("Target variable distribution:")
        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()

        # Create distribution dataframe
        y_dist_data = []
        for label in train_dist.index:
            train_prop = train_dist[label]
            test_prop = test_dist.get(label, 0)
            logger.info(f"  Class {label}: Train {train_prop:.3f}, Test {test_prop:.3f}")

            y_dist_data.append({
                'class': label,
                'train_count': int(y_train.value_counts()[label]),
                'train_proportion': train_prop,
                'test_count': int(y_test.value_counts().get(label, 0)),
                'test_proportion': test_prop
            })

        stats_data['target_distribution'] = pd.DataFrame(y_dist_data)

    # Other stratification variable distributions
    if stratify_data is not None and stratify_cols:
        for col in stratify_cols if isinstance(stratify_cols, list) else [stratify_cols]:
            if col in stratify_data.columns:
                logger.debug(f"{col} distribution:")
                train_indices = X_train.index
                test_indices = X_test.index
                train_dist = stratify_data.loc[train_indices, col].value_counts(normalize=True).sort_index()
                test_dist = stratify_data.loc[test_indices, col].value_counts(normalize=True).sort_index()

                # Create distribution dataframe for this column
                col_dist_data = []
                for label in train_dist.index:
                    train_prop = train_dist[label]
                    test_prop = test_dist.get(label, 0)
                    logger.debug(f"  {label}: Train {train_prop:.3f}, Test {test_prop:.3f}")

                    col_dist_data.append({
                        'category': label,
                        'train_count': int(stratify_data.loc[train_indices, col].value_counts()[label]),
                        'train_proportion': train_prop,
                        'test_count': int(stratify_data.loc[test_indices, col].value_counts().get(label, 0)),
                        'test_proportion': test_prop
                    })

                stats_data[f'{col}_distribution'] = pd.DataFrame(col_dist_data)

    # Save statistics to files if output_dir is provided
    if output_dir:
        output_path = Path(output_dir)
        stats_path = output_path / 'split_statistics'
        stats_path.mkdir(parents=True, exist_ok=True)

        for name, df in stats_data.items():
            df.to_csv(stats_path / f'{name}.tsv', sep='\t', index=False)
            logger.info(f"Saved {name} to {stats_path / f'{name}.tsv'}")

        # Also save a combined Excel file for convenience
        with pd.ExcelWriter(stats_path / 'split_statistics.xlsx') as writer:
            for name, df in stats_data.items():
                df.to_excel(writer, sheet_name=name, index=False)
        logger.info(f"Saved combined statistics to {stats_path / 'split_statistics.xlsx'}")

        create_publication_figures(stats_data, output_path)
        create_sample_size_analysis(X, y, output_path)

    return X_train, X_test, y_train, y_test


def save_split_data(X_train, X_test, y_train, y_test, output_dir):
    """Save split data to specified directory with enhanced metadata"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save training set as TSV
    X_train.to_csv(output_path / "X_train.tsv", index=True, sep='\t')
    y_train.to_csv(output_path / "y_train.tsv", index=True, header=True, sep='\t')

    # Save test set as TSV
    X_test.to_csv(output_path / "X_test.tsv", index=True, sep='\t')
    y_test.to_csv(output_path / "y_test.tsv", index=True, header=True, sep='\t')

    logger.info(f"Data saved to: {output_path}")
    logger.info(f"  - X_train.tsv: {X_train.shape}")
    logger.info(f"  - X_test.tsv: {X_test.shape}")
    logger.info(f"  - y_train.tsv: {y_train.shape}")
    logger.info(f"  - y_test.tsv: {y_test.shape}")

    # Save metadata
    metadata = {
        'split_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X_train) + len(X_test),
            'train_ratio': len(X_train) / (len(X_train) + len(X_test)),
            'test_ratio': len(X_test) / (len(X_train) + len(X_test)),
            'n_features': X_train.shape[1],
            'n_classes': len(y_train.unique())
        },
        'class_distribution': {
            'train': y_train.value_counts().to_dict(),
            'test': y_test.value_counts().to_dict()
        }
    }

    import json
    with open(output_path / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


#### Main ####
@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-g", "--genotype",
              type=click.Path(exists=True),
              required=True,
              help="The genotype input file (TSV format)")
@click.option("-p", "--phenotype",
              type=click.Path(exists=True),
              required=True,
              help="The phenotype input file (TSV format)")
@click.option("-s", "--stratify-file",
              type=click.Path(exists=True),
              required=False,
              help="File containing additional stratification information (TSV format, e.g., batch, site)")
@click.option("--testsize",
              type=float,
              default=0.2,
              show_default=True,
              help="The proportion of the dataset that should be used for testing")
@click.option("--seed",
              type=int,
              default=42,
              show_default=True,
              help="The random seed for split dataset")
@click.option("--stratify-cols",
              type=str,
              multiple=True,
              help="Column names for stratified sampling from stratify-file (can be specified multiple times)")
@click.option("--include-y/--no-include-y",
              default=True,
              show_default=True,
              help="Whether to include target variable in stratification")
@click.option("-o", "--out",
              type=click.Path(),
              required=True,
              help="The output directory")
def main(genotype, phenotype, stratify_file, testsize, seed, stratify_cols, include_y, out):
    """
    Split genotype and phenotype data into training and test sets with stratification support.

    This tool reads genotype and phenotype data files (TSV format) and splits them into training
    and test sets while maintaining the distribution of specified variables through stratified sampling.
    Publication-quality PDF figures are generated for visualization.
    """
    out = Path(out).absolute()
    #### 日志设定
    ft_subset_logger = logging.getLogger('fontTools.subset')
    ft_subset_logger.setLevel(logging.ERROR)
    ft_subset_logger.addFilter(FontToolsSubsetFilter())

    try:
        logger.info(f"Reading genotype data: {genotype}")
        df_genotype = pd.read_csv(genotype, index_col=0, sep='\t')
        logger.info(f"Genotype data shape: {df_genotype.shape}")

        logger.info(f"Reading phenotype data: {phenotype}")
        df_phenotype = pd.read_csv(phenotype, index_col=0, sep='\t')
        logger.info(f"Phenotype data shape: {df_phenotype.shape}")

        if not df_genotype.index.equals(df_phenotype.index):
            common_samples = df_genotype.index.intersection(df_phenotype.index)
            logger.warning(f"Genotype and phenotype data samples do not fully match")
            logger.warning(f"Common samples: {len(common_samples)}")

            # Keep only common samples
            df_genotype = df_genotype.loc[common_samples]
            df_phenotype = df_phenotype.loc[common_samples]
            logger.info(f"Filtered data shape: Genotype {df_genotype.shape}, Phenotype {df_phenotype.shape}")

        # Read stratification file if provided
        df_stratify = None
        if stratify_file:
            logger.info(f"Reading stratification file: {stratify_file}")
            df_stratify = pd.read_csv(stratify_file, index_col=0, sep='\t')
            logger.info(f"Stratification data shape: {df_stratify.shape}")
            logger.info(f"Available columns: {', '.join(df_stratify.columns)}")

            # Check sample overlap
            common_with_stratify = df_genotype.index.intersection(df_stratify.index)
            if len(common_with_stratify) < len(df_genotype):
                logger.warning(f"Only {len(common_with_stratify)} out of {len(df_genotype)} samples "
                               f"have stratification information")
                # Filter all data to common samples
                df_genotype = df_genotype.loc[common_with_stratify]
                df_phenotype = df_phenotype.loc[common_with_stratify]
                df_stratify = df_stratify.loc[common_with_stratify]
                logger.info(f"Final data shape after filtering: {df_genotype.shape}")

        # Convert stratify_cols to list (if specified)
        stratify_cols_list = list(stratify_cols) if stratify_cols else None

        # Validate stratify_cols if stratify_file is provided
        if stratify_cols_list and df_stratify is not None:
            missing_cols = [col for col in stratify_cols_list if col not in df_stratify.columns]
            if missing_cols:
                raise ValueError(f"Columns not found in stratification file: {', '.join(missing_cols)}")

        # Perform data split
        logger.info("Starting data split...")
        X_train, X_test, y_train, y_test = balanced_train_test_split(
            df_genotype,
            df_phenotype,
            test_size=testsize,
            random_state=seed,
            stratify_data=df_stratify,
            stratify_cols=stratify_cols_list,
            include_y=include_y,
            output_dir=out  # Pass output directory for statistics
        )

        # Save results
        save_split_data(X_train, X_test, y_train, y_test, out)

        # Also save the indices for reference as TSV
        indices_path = Path(out)
        pd.DataFrame({'sample_id': X_train.index, 'set': 'train'}).to_csv(
            indices_path / "train_samples.tsv", index=False, sep='\t'
        )
        pd.DataFrame({'sample_id': X_test.index, 'set': 'test'}).to_csv(
            indices_path / "test_samples.tsv", index=False, sep='\t'
        )
        create_methods_section(out)

        logger.info("Data split completed successfully!")

    except Exception as e:
        logger.error(f"\033[31mError during processing: {str(e)}\033[0m")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()
