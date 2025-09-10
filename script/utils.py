#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: CB Ren
# @Created Date:   2024/10/8 0:19
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import json
import itertools

from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score

logger = logging.getLogger(__file__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class NumpyEncoder(json.JSONEncoder):
    """
    用于将numpy数组转换为json格式
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def get_features(f_feature_list):
    """
    从特征文件中读取特征列表
    :param f_feature_list: 特征文件路径
    """
    res = []
    with open(f_feature_list, "r") as IN:
        for line in IN:
            res.append(line.strip())
    return res


def binary_classification_plot(fpr, tpr, auc_val, d_out: Path):
    """
    绘制二分类ROC曲线

    :param fpr: 假阳性率
    :param tpr: 真阳性率
    :param auc_val: AUC值
    :param d_out: 输出目录
    """
    figure, axis = plt.subplots(figsize=(7, 7))

    axis.plot([0, 1], [0, 1], "k--", label="Random classifier")
    axis.plot(fpr, tpr, "#D32421", linewidth=2, label=f"ROC curve {auc_val:.4}")
    axis.grid()
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("ROC Curve")
    axis.legend(loc="lower right")
    f_pdf = d_out.joinpath("roc_curve.pdf")
    plt.savefig(f_pdf, bbox_inches="tight")
    f_png = d_out.joinpath("roc_curve.png")
    plt.savefig(f_png, dpi=300, bbox_inches="tight")


def train_test_binary_roc_plot(train, test, d_out: Path, title="ROC Curve"):
    """
    绘制训练集和测试集ROC曲线

    :param fpr: 假阳性率
    :param tpr: 真阳性率
    :param auc_val: AUC值
    :param d_out: 输出目录
    """
    figure, axis = plt.subplots(figsize=(7, 7))

    axis.plot([0, 1], [0, 1], "k--", label="Random classifier")
    axis.plot(train["fpr"], train["tpr"], "#D32421", linewidth=2, label=f"ROC curve Train {train['auc']:.4}")
    axis.plot(test["fpr"], test["tpr"], "#2771A7", linewidth=2, label=f"ROC curve Test {test['auc']:.4}")
    axis.grid()
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend(loc="lower right")
    f_pdf = d_out.joinpath("roc_curve.pdf")
    plt.savefig(f_pdf, bbox_inches="tight")
    f_png = d_out.joinpath("roc_curve.png")
    plt.savefig(f_png, dpi=300, bbox_inches="tight")


def align_dataframes(df_genotype, df_phenotype):
    """
    确保两个数据框的行名顺序一致

    :param df_genotype: 基因型数据框
    :param df_phenotype: 表型数据框
    :return: 行名顺序一致的两个数据框
    """
    # 获取两个数据框共有的行名
    common_indices = df_genotype.index.intersection(df_phenotype.index)

    if len(common_indices) < len(df_genotype.index) or len(common_indices) < len(df_phenotype.index):
        logger.warning(f"有 {len(df_genotype.index) - len(common_indices)} 个样本在基因型中但不在表型中")
        logger.warning(f"有 {len(df_phenotype.index) - len(common_indices)} 个样本在表型中但不在基因型中")

    # 使用共有的行名并按相同顺序排列两个数据框
    df_genotype_aligned = df_genotype.loc[common_indices]
    df_phenotype_aligned = df_phenotype.loc[common_indices]

    # 验证行名是否一致
    assert all(df_genotype_aligned.index == df_phenotype_aligned.index), "行名不一致"

    return df_genotype_aligned, df_phenotype_aligned


def auto_load_data(f_genotype, f_phenotype, f_test_genotype=None, f_test_phenotype=None, random_seed=42, test_size=0.3,
                   features=[]):
    """
    自动加载数据，如果只提供了一个基因型文件和表型文件，那么会自动划分训练集和测试集，如果提供了两个基因型文件和表型文件，那么会使用这两个文件作为训练集和测试集

    :param f_genotype: 基因型文件路径
    :param f_phenotype: 表型文件路径
    :param f_test_genotype: 测试集基因型文件
    :param f_test_phenotype: 测试集表型文件
    :param random_seed: 划分训练集和测试集的随机种子
    :param test_size: 测试集的占比
    """
    if (f_test_genotype == None and f_test_phenotype != None) or (f_test_genotype != None and f_test_phenotype == None):
        logger.error("Test set must be provided both genotypes and phenotypes")
        exit(-1)

    logger.info(f"Read the genotype file: {f_genotype}")
    df_genotype = pd.read_csv(f_genotype, index_col=0, sep="\t")
    logger.info(f"Read the phenotype file: {f_phenotype}")
    df_phenotype = pd.read_csv(f_phenotype, index_col=0, sep="\t")

    df_genotype, df_phenotype = align_dataframes(df_genotype, df_phenotype)

    if len(features) == 0:
        features = list(df_genotype.columns)
    df_genotype = df_genotype.loc[:, features]

    if f_test_genotype == None and f_test_phenotype == None:
        from sklearn.model_selection import train_test_split

        logger.info("Split the dataset into training and testing sets")
        x_train, x_test, y_train, y_test = (
            train_test_split(df_genotype, df_phenotype, test_size=test_size, random_state=random_seed)
        )
        return x_train, x_test, y_train, y_test
    else:
        logger.info(f"Read the test genotype file: {f_test_genotype}")
        df_genotype_test = pd.read_csv(f_test_genotype, index_col=0, sep="\t")
        df_genotype_test = df_genotype_test.loc[:, features]
        logger.info(f"Read the test phenotype file: {f_test_phenotype}")
        df_phenotype_test = pd.read_csv(f_test_phenotype, index_col=0, sep="\t")

        df_genotype_test, df_phenotype_test = align_dataframes(df_genotype_test, df_phenotype_test)

        return df_genotype, df_genotype_test, df_phenotype.values.ravel(), df_phenotype_test.values.ravel()


def plot_learning_curves(estimator, title, X, y, ylim=None, cv=5, n_jobs=-1, scoring="accuracy",
                         train_sizes=np.linspace(.1, 1.0, 10), out='./'):
    """
    绘制模型的学习曲线

    :param estimator: 模型
    :param title: 图片标题
    :param X: 特征
    :param y: 表型
    :param ylim: y轴范围
    :param cv: 交叉验证折数
    :param n_jobs: 并行计算
    :param train_sizes: 训练集大小
    :param out: 输出目录
    """
    # 数据处理
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            cv=cv,
                                                            n_jobs=n_jobs,
                                                            scoring=scoring,
                                                            train_sizes=train_sizes,
                                                            error_score=np.nan)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # 绘图
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.set_title(title, fontweight="bold", fontsize=16)
    if ylim is not None:
        axis.set_ylim(*ylim)
    else:
        axis.set_ylim([0.0, 1.1])
    axis.set_xlabel("Training examples", fontweight="bold", fontsize=14)
    axis.set_ylabel(f"{scoring} score", fontweight="bold", fontsize=14)
    axis.grid()
    axis.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                      alpha=0.1, color="#D32421")
    axis.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                      color="#2771A7")
    axis.plot(train_sizes, train_scores_mean, 'o-', color="#D32421", label="Training score")
    axis.plot(train_sizes, test_scores_mean, 'o-', color="#2771A7", label="Cross-validation score")

    axis.legend(loc="best")

    # 保存
    d_out = Path(out).absolute()
    f_pdf = d_out.joinpath(f"{scoring}.learning_curves.pdf")
    plt.savefig(f_pdf, bbox_inches="tight")
    f_png = d_out.joinpath(f"{scoring}.learning_curves.png")
    plt.savefig(f_png, dpi=300, bbox_inches="tight")


def pi_importance_stat(model, x, y, d_out, repeat=10, seed=42, top=10, color="#D32421", loss_function=None):
    """
    排列特征重要性统计

    :param model: 模型
    :param x: 特征
    :param y: 表型
    :param d_out: 输出目录
    :param repeat: 重复次数
    :param seed: 随机种子
    :param top: 只绘制重要性排名前top个特征
    :param color: 柱状图颜色与盒型图的颜色
    :param loss_function: 损失函数, 可选: 'loss_mse','loss_root_mean_square','loss_classification_error','loss_log_loss','loss_1_auc'
    """

    # 各种损失函数
    def rmse(y_true, y_pred):
        """
       定义均方根误差（RMSE）作为损失函数

        :param y_true: 真实值
        :param y_pred: 预测值
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # 创建自定义评分器
    # TODO: 添加其他损失函数
    if loss_function == "loss_root_mean_square":
        scorer = make_scorer(rmse, greater_is_better=False)
    else:
        scorer = None

    try:
        if scorer:
            result = permutation_importance(model, x, y, n_repeats=repeat, random_state=seed, scoring=scorer, n_jobs=-1)
        else:
            result = permutation_importance(model, x, y, n_repeats=repeat, random_state=seed, n_jobs=-1)
    except Exception as e:
        logger.error(f"Error calculating permutation importance: {e}")
        return

    # 获取特征重要性
    importances = result.importances
    importances_mean = result.importances_mean
    std = result.importances_std
    feature_names = x.columns

    # 对特征重要性进行排序
    indices = np.argsort(importances_mean)[::-1]
    indices_top = indices[:top]

    # 将特征重要性保存到文件中
    f_importances = f"{d_out}/pi_importances.tsv"
    with open(f_importances, "w") as OUT:
        # print(*["Feature", "Importance"], sep='\t', file=OUT)
        for f, idx in enumerate(indices):
            print(*[feature_names[idx], *importances[idx]], sep='\t', file=OUT)
    f_importance_mean = f"{d_out}/pi_importances_mean.tsv"
    with open(f_importance_mean, "w") as OUT:
        print(*["Feature", "Importance Mean", "Std"], sep='\t', file=OUT)
        for f, idx in enumerate(indices):
            print(*[feature_names[idx], importances_mean[idx], std[idx]], sep='\t', file=OUT)

    # 绘制特征重要性并保存图像
    f_pdf = f"{d_out}/pi_importances.pdf"
    f_png = f"{d_out}/pi_importances.png"
    figure, axis = plt.subplots(figsize=(7, 12))
    axis.barh(range(len(indices_top)),
              importances_mean[indices_top],
              color=color,
              alpha=0.5,
              height=0.5)
    axis.boxplot(importances[indices_top].T,
                 positions=np.arange(len(indices_top)),
                 widths=0.1,
                 patch_artist=True,
                 showfliers=False,
                 boxprops=dict(facecolor=color, color=color),
                 medianprops=dict(color=color),
                 whiskerprops=dict(color=color),
                 capprops=dict(visible=False),
                 vert=False)
    axis.invert_yaxis()
    axis.set_title(f"Feature Importances (Top {top})", fontweight="bold", fontsize=18, pad=15)
    axis.set_yticks(range(len(indices_top)), labels=feature_names[indices_top])
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.grid(axis='x', which='major', linestyle='--', linewidth=0.2, alpha=0.7, color='grey')
    axis.set_xlabel(loss_function, fontweight="bold", fontsize=14, labelpad=15)
    axis.tick_params(axis='y', pad=10, length=0)
    axis.tick_params(axis='x', pad=10, length=0)
    plt.savefig(f_pdf, bbox_inches='tight')
    plt.savefig(f_png, dpi=300, bbox_inches='tight')


def multi_roc_plot(data, d_out):
    """
    多个模型的ROC曲线绘制

    :param data: A dict with model name as key and a dict of {'fpr': fpr, 'tpr': tpr, 'auc': auc} as value
    :param d_out: The output directory
    """
    figure, axis = plt.subplots(figsize=(7, 7))

    # TODO: 颜色处理
    colors = ["#D32421", "#2771A7", "#27A727", "#A727A7", "#A72727"]

    axis.plot([0, 1], [0, 1], "k--", label="Random classifier")
    index = 0
    for model_name, model_data in data.items():
        fpr = model_data["fpr"]
        tpr = model_data["tpr"]
        auc_val = model_data["auc"]
        axis.plot(fpr, tpr, colors[index], linewidth=2, label=f"{model_name} {auc_val:.4}")
        index += 1
    axis.grid()
    axis.set_xlabel("False Positive Rate", fontweight="bold", fontsize=14, labelpad=15)
    axis.set_ylabel("True Positive Rate", fontweight="bold", fontsize=14, labelpad=15)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("ROC Curve", fontweight="bold", fontsize=16, pad=15)
    axis.legend(loc="lower right")
    f_pdf = d_out.joinpath("multi_roc_curve.pdf")
    plt.savefig(f_pdf, bbox_inches="tight")
    f_png = d_out.joinpath("multi_roc_curve.png")
    plt.savefig(f_png, dpi=300, bbox_inches="tight")


def save_param(best_params, f_out):
    """
    保存模型最佳参数

    :param best_params: 最佳参数
    :param f_out: 输出文件
    """
    logger.info(f"保存参数到文件:{f_out}")
    with open(f_out, "w") as OUT:
        json.dump(best_params, OUT, cls=NumpyEncoder)


def save_report(class_report, f_out):
    """
    保存模型报告

    :param class_report: 分类报告
    :param f_out: 输出文件
    """
    logger.info(f"保存报告到文件:{f_out}")
    with open(f_out, "w") as OUT:
        print(class_report, file=OUT)


def plot_confusion_matrix(cm, d_out, classes=["Healthy", "Disease"], title='Confusion matrix'):
    """
    绘制混淆矩阵

    :param cm: 混淆矩阵
    :param d_out: 输出目录
    :param classes: 类别名称
    :param title: 标题
    """
    figure, axis = plt.subplots(figsize=(7, 7))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title(title, fontweight="bold", fontsize=16, pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    axis.set_xticks(tick_marks, classes, fontweight="bold", fontsize=14, rotation=45)
    axis.set_yticks(tick_marks, classes, fontweight="bold", fontsize=14)

    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axis.text(j, i,
                  format(cm[i, j], '.0f'),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > threshold else "black")

    axis.set_xlabel('Predicted label', fontweight="bold", fontsize=14, labelpad=15)
    axis.set_ylabel('True label', fontweight="bold", fontsize=14, labelpad=15)

    f_pdf = d_out.joinpath("confusion_matrix.pdf")
    plt.savefig(f_pdf, bbox_inches="tight")
    f_png = d_out.joinpath("confusion_matrix.png")
    plt.savefig(f_png, dpi=300, bbox_inches="tight")


def evaluate_models_on_test(models_dict, X_test, y_test, d_out):
    d_out = Path(d_out).absolute()
    d_out.mkdir(parents=True, exist_ok=True)

    results = []

    unique_labels = np.unique(y_test)
    print(f"\nTest set label distribution:")
    print(f"Unique labels: {unique_labels}")
    print(f"Class counts: {np.bincount(y_test)}")
    print(f"Class proportions: {np.bincount(y_test) / len(y_test)}")
    print("\n")

    for model_name, model in models_dict.items():
        logger.debug(f"Evaluating model: {model_name}")

        try:
            y_pred = model.predict(X_test)

            logger.debug(f"Predicted label distribution: {np.bincount(y_pred)}")

            unique_pred = np.unique(y_pred)
            if len(unique_pred) == 1:
                logger.warning(f"{model_name} predicted only class {unique_pred[0]}")

            accuracy = round(accuracy_score(y_test, y_pred), 3)

            from sklearn.metrics import classification_report

            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            precision = round(report['weighted avg']['precision'], 3)
            recall = round(report['weighted avg']['recall'], 3)
            f1 = round(report['weighted avg']['f1-score'], 3)

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.debug(f"Confusion matrix:\n{cm}")

            logger.debug(f"\nClassification report:")
            logger.debug(classification_report(y_test, y_pred, zero_division=0))

            results.append({
                'Model': model_name,
                'Precision (weighted)': precision,
                'Recall (weighted)': recall,
                'F1-score (weighted)': f1,
                'Accuracy': accuracy,
            })

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'Model': model_name,
                'Precision (weighted)': 'Error',
                'Recall (weighted)': 'Error',
                'F1-score (weighted)': 'Error',
                'Accuracy': 'Error'
            })

    df_results = pd.DataFrame(results)
    f_out = d_out.joinpath('model_evaluation_results.tsv')
    df_results.to_csv(f_out, index=False, sep="\t")
    logger.info(f"Results saved to: {f_out}")

    return df_results
