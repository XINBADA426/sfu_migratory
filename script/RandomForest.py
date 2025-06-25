#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Ming Jia
# @Created Date:   2024/10/8 12:18
import matplotlib as mpl

mpl.use("Agg")

import pickle
from pathlib import Path
import logging
import click
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier

from utils import get_features, NumpyEncoder, auto_load_data
from figures import plot_roc_pr_curves, plot_confusion_matrix, plot_learning_curve

logger = logging.getLogger(__file__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


#### Some Function ####
# 1. 交叉验证结果详细分析
def detailed_cv_analysis(gc, kfolds):
    """详细分析交叉验证结果"""
    cv_results = pd.DataFrame(gc.cv_results_)

    # 获取每折的得分
    cv_scores = []
    for i in range(kfolds):
        cv_scores.append(cv_results[f'split{i}_test_score'].iloc[gc.best_index_])

    cv_stats = {
        "交叉验证各折得分": cv_scores,
        "交叉验证平均得分": np.mean(cv_scores),
        "交叉验证标准差": np.std(cv_scores),
        "交叉验证得分范围": f"{np.min(cv_scores):.4f} - {np.max(cv_scores):.4f}"
    }

    return cv_stats


def comprehensive_model_evaluation(best_model, x_train, y_train, x_test, y_test, cv_stats):
    """综合模型性能评估"""

    y_train_pred = best_model.predict(x_train)
    y_test_pred = best_model.predict(x_test)

    train_score = best_model.score(x_train, y_train)
    test_score = best_model.score(x_test, y_test)

    train_report = classification_report(y_train, y_train_pred, digits=4)
    test_report = classification_report(y_test, y_test_pred, digits=4)

    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    if len(np.unique(y_test)) == 2:
        train_prob = best_model.predict_proba(x_train)[:, 1]
        test_prob = best_model.predict_proba(x_test)[:, 1]
        train_auc = roc_auc_score(y_train, train_prob)
        test_auc = roc_auc_score(y_test, test_prob)
    else:
        train_auc = test_auc = None

    performance_summary = {
        "=== 交叉验证结果 ===": "",
        "CV平均得分": f"{cv_stats['交叉验证平均得分']:.4f}",
        "CV标准差": f"{cv_stats['交叉验证标准差']:.4f}",
        "CV得分范围": cv_stats['交叉验证得分范围'],
        "CV各折得分": cv_stats['交叉验证各折得分'],

        "=== 训练集性能 ===": "",
        "训练集准确率": f"{train_score:.4f}",
        "训练集AUC": f"{train_auc:.4f}" if train_auc else "N/A",

        "=== 测试集性能（最终评估）===": "",
        "测试集准确率": f"{test_score:.4f}",
        "测试集AUC": f"{test_auc:.4f}" if test_auc else "N/A",

        "=== 过拟合分析 ===": "",
        "训练集与测试集准确率差异": f"{abs(train_score - test_score):.4f}",
        "过拟合程度": "高" if abs(train_score - test_score) > 0.1 else (
            "中等" if abs(train_score - test_score) > 0.05 else "低"),
    }

    return {
        'summary': performance_summary,
        'train_report': train_report,
        'test_report': test_report,
        'train_cm': train_cm,
        'test_cm': test_cm
    }


def improved_evaluation_section(gc, kfolds, best_model, x_train, y_train, x_test, y_test, d_out):
    """改进的评估部分，替换你原来的代码"""

    cv_stats = detailed_cv_analysis(gc, kfolds)

    evaluation_results = comprehensive_model_evaluation(
        best_model, x_train, y_train, x_test, y_test, cv_stats
    )

    logger.info("=== 模型性能综合报告 ===")
    for key, value in evaluation_results['summary'].items():
        if value == "":
            logger.info(key)
        else:
            logger.info(f"{key}: {value}")

    f_comprehensive_report = d_out.joinpath("comprehensive_evaluation.txt")
    with open(f_comprehensive_report, "w", encoding='utf-8') as OUT:
        print("=== 模型性能综合评估报告 ===\n", file=OUT)

        for key, value in evaluation_results['summary'].items():
            if value == "":
                print(key, file=OUT)
            else:
                print(f"{key}: {value}", file=OUT)

        print("\n=== 训练集详细分类报告 ===", file=OUT)
        print(evaluation_results['train_report'], file=OUT)

        print("\n=== 测试集详细分类报告 ===", file=OUT)
        print(evaluation_results['test_report'], file=OUT)

        print("\n=== 训练集混淆矩阵 ===", file=OUT)
        print(evaluation_results['train_cm'], file=OUT)

        print("\n=== 测试集混淆矩阵 ===", file=OUT)
        print(evaluation_results['test_cm'], file=OUT)

    return evaluation_results


def create_performance_visualization(evaluation_results, cv_stats, d_out):
    """创建性能可视化图表"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].bar(range(len(cv_stats['交叉验证各折得分'])), cv_stats['交叉验证各折得分'])
    axes[0, 0].axhline(y=cv_stats['交叉验证平均得分'], color='r', linestyle='--', label='Mean Score')
    axes[0, 0].set_title('Cross-validation Fold Scores')
    axes[0, 0].set_xlabel('Fold Number')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()

    sns.heatmap(evaluation_results['train_cm'], annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
    axes[0, 1].set_title('Training Set Confusion Matrix')

    sns.heatmap(evaluation_results['test_cm'], annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
    axes[1, 0].set_title('Test Set Confusion Matrix')

    performance_data = {
        'Cross-Validation': cv_stats['交叉验证平均得分'],
        'Train': float(evaluation_results['summary']['训练集准确率']),
        'Test': float(evaluation_results['summary']['测试集准确率'])
    }

    bars = axes[1, 1].bar(performance_data.keys(), performance_data.values())
    axes[1, 1].set_title('Model Performance Comparison')
    axes[1, 1].set_ylabel('Accuracy')

    for bar, value in zip(bars, performance_data.values()):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(d_out.joinpath('performance_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Performance visualization saved to: {d_out.joinpath('performance_visualization.png')}")


#### Main ####
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
              help="The validation genotype input file(must be provided with --vphenotype)")
@click.option("-vp", "--vphenotype",
              type=click.Path(),
              help="The validation phenotype input file(must be provided with --vgenotype)")
@click.option("--feature",
              type=click.Path(),
              help="A file containing features to be used")
@click.option("--testsize",
              type=float,
              default=0.25,
              show_default=True,
              help="The proportion of the dataset that should be used for testing(only when --vgenotype or --vphenotype is not given)")
@click.option("--seed",
              type=int,
              default=42,
              show_default=True,
              help="The random seed for split dataset and xgb model training")
@click.option("--n_estimators",
              default=None,
              show_default=True,
              help="决策树的数量，范围为[0, +∞]，默认为100")
@click.option("--max_depth",
              default=None,
              show_default=True,
              help="最大深度，整数类型，范围为[0, +∞]，默认无穷大")
@click.option("--min_samples_split",
              default=None,
              show_default=True,
              help="分裂一个内部节点所需的最小样本数，整数类型，范围为[0, +∞]，默认2")
@click.option("--min_samples_leaf",
              default=None,
              show_default=True,
              help="叶子节点最少样本数，整数类型，范围为[0, +∞]，默认1")
@click.option("--scoring",
              default="roc_auc",
              show_default=True,
              help="The scoring method for cross validation and plot curves")
@click.option("-k", "--kfolds",
              type=int,
              default=5,
              show_default=True,
              help="The k fold used for cross validation")
@click.option("--n_bootstrap",
              type=int,
              default=1000,
              show_default=True,
              help="Number of bootstrap iterations for stability selection")
@click.option("-o", "--out",
              type=click.Path(),
              required=True,
              help="The output dir")
def randomforest(genotype, phenotype, vgenotype, vphenotype, feature, testsize, seed, n_estimators, max_depth,
                 min_samples_split, min_samples_leaf, scoring, kfolds, n_bootstrap, out):
    """
    使用随机森林进行模型构建

    如果不提供验证集，则会自动按照指定规则从原始数据集分割出训练集和测试集。
    """
    f_genotype = Path(genotype).absolute()
    f_phenotype = Path(phenotype).absolute()
    d_out = Path(out).absolute()
    d_out.mkdir(parents=True, exist_ok=True)

    features = []
    if feature:
        f_feature = Path(feature).absolute()
        logger.info(f"Read the feature file: {f_feature}")
        features = get_features(f_feature)

    X_train, X_test, y_train, y_test = auto_load_data(f_genotype, f_phenotype,
                                                      f_test_genotype=vgenotype, f_test_phenotype=vphenotype,
                                                      random_seed=seed, test_size=testsize, features=features)

    n_class = len(np.unique(y_train))
    original_features = X_train.columns.tolist()

    X_train_scale = X_train
    X_test_scale = X_test

    logger.info("Parameter optimization using grid search")
    param_grid = {}
    if n_estimators:
        if '_' in n_estimators:
            # 给定的取值为一个范围start_stop_step
            param_grid["n_estimators"] = list(np.arange(*[int(i) for i in n_estimators.split("_")]))
        elif ',' in n_estimators:
            # 给定的取值为一个列表1,2,3,4,5
            param_grid["n_estimators"] = [int(i) for i in n_estimators.split(",")]
        else:
            # 给定的取值为一个单个值
            param_grid["n_estimators"] = [int(n_estimators)]
    if max_depth:
        if '_' in max_depth:
            param_grid["max_depth"] = np.arange(*[int(i) for i in max_depth.split("_")])
        elif ',' in max_depth:
            param_grid["max_depth"] = [int(i) for i in max_depth.split(",")]
        else:
            param_grid["max_depth"] = [int(max_depth)]
    if min_samples_split:
        if '_' in min_samples_split:
            param_grid["min_samples_split"] = np.arange(*[int(i) for i in min_samples_split.split("_")])
        elif ',' in min_samples_split:
            param_grid["min_samples_split"] = [int(i) for i in min_samples_split.split(",")]
        else:
            param_grid["min_samples_split"] = [int(min_samples_split)]
    if min_samples_leaf:
        if '_' in min_samples_leaf:
            param_grid["min_samples_leaf"] = np.arange(*[int(i) for i in min_samples_leaf.split("_")])
        elif ',' in min_samples_leaf:
            param_grid["min_samples_leaf"] = [int(i) for i in min_samples_leaf.split(",")]
        else:
            param_grid["min_samples_leaf"] = [int(min_samples_leaf)]

    param_num = 1
    for i in param_grid:
        param_num *= len(param_grid[i])
    logger.info(f"参数空间大小: {param_num}")

    logger.info("Get the best parameters")
    model = RandomForestClassifier(random_state=seed)
    stratified_cv = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=seed)
    gc = GridSearchCV(model,
                      param_grid,
                      cv=stratified_cv,
                      scoring=scoring,
                      n_jobs=-1,
                      return_train_score=True)
    gc.fit(X_train_scale, y_train)

    best_params = gc.best_params_
    best_model = gc.best_estimator_

    best_index = gc.best_index_
    train_score_cv = gc.cv_results_['mean_train_score'][best_index]
    val_score_cv = gc.cv_results_['mean_test_score'][best_index]

    print(f"\n最佳模型参数组合: {best_params}")
    print(f"交叉验证训练集平均得分: {train_score_cv:.4f}")
    print(f"交叉验证验证集平均得分: {val_score_cv:.4f}")

    y_pred = best_model.predict(X_test_scale)
    test_score = best_model.score(X_test_scale, y_test)

    train_score_full = best_model.score(X_train_scale, y_train)

    info_score = {
        "完整训练集得分": train_score_full,
        "测试集得分": test_score,
    }
    for i, j in info_score.items():
        print(f"{i}: {j}")

    f_score = d_out.joinpath("score.tsv")
    logger.info(f"保存得分信息到文件: {f_score}")
    with open(f_score, "w") as OUT:
        for k, v in info_score.items():
            print(f"{k}\t{v}", file=OUT)

    f_param = d_out.joinpath("param.json")
    logger.info(f"保存参数到文件:{f_param}")
    with open(f_param, "w") as OUT:
        json.dump(best_params, OUT, cls=NumpyEncoder)

    class_report = classification_report(y_test, y_pred, digits=4)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f_confusion_matrix = d_out.joinpath("confusion_matrix.tsv")
    logger.info(f"保存混淆矩阵到文件:{f_confusion_matrix}")
    with open(f_confusion_matrix, "w") as OUT:
        print(conf_matrix, file=OUT)

    f_report = d_out.joinpath("report.txt")
    logger.info(f"保存报告到文件:{f_report}")
    with open(f_report, "w") as OUT:
        print(class_report, file=OUT)

    f_model = d_out.joinpath("model.pkl")
    logger.info(f"保存模型到文件:{f_model}")
    pickle.dump(best_model, open(f_model, "wb"))

    logger.info("Learning curve Plot")
    f_learning_curve_plot = d_out.joinpath("learning_curve.pdf")
    plot_learning_curve(best_model, X_train_scale, y_train, f_learning_curve_plot,
                        scoring=scoring,
                        cv=stratified_cv,
                        n_jobs=-1,
                        title='RandomForest')

    logger.info("ROC & PR curves Plot")
    if n_class > 2:
        pass
        # TODO: 多分类
        # phenotype_test_one_hot = label_binarize(
        #     df_phenotype_test, classes=np.arange(n_class)
        # )
        # phenotype_test_one_hot_hat = model.predict_proba(df_genotype_test)
        # fpr, tpr, _ = roc_curve(
        #     phenotype_test_one_hot, phenotype_test_one_hot_hat.ravel()
        # )
        # auc = roc_auc_score(fpr, tpr, average="macro")
    else:
        f_roc_pr_plot = d_out.joinpath("performance_curves.pdf")
        plot_roc_pr_curves(best_model, X_test_scale, y_test, f_roc_pr_plot, seed=seed, n_bootstraps=n_bootstrap)

    logger.info("Confusion matrix Plot")
    f_confusion_matrix_plot = d_out.joinpath("confusion_matrix.pdf")
    plot_confusion_matrix(best_model, X_test_scale, y_test, f_confusion_matrix_plot, title="RandomForest")


if __name__ == "__main__":
    randomforest()
