#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Ming Jia
# @Created Date:   2024/10/8 0:01
import pickle
from pathlib import Path
import logging
import click
import numpy as np
import json
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB

from utils import get_features, NumpyEncoder, auto_load_data

from figures import plot_roc_pr_curves, plot_confusion_matrix, plot_learning_curve

logger = logging.getLogger(__file__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
              help="The random seed for split dataset")
@click.option("--nb_type",
              type=click.Choice(['multinomial', 'gaussian', 'bernoulli', 'complement']),
              default='bernoulli',
              show_default=True,
              help="朴素贝叶斯分类器类型")
@click.option("--alpha",
              default=None,
              show_default=True,
              help="加性平滑参数（拉普拉斯平滑），范围为[0, +∞]，默认为1.0。值越大，平滑程度越高")
@click.option("--fit_prior",
              type=click.Choice(['True', 'False']),
              default='True',
              show_default=True,
              help="是否学习类的先验概率。如果为False，将使用均匀先验")
@click.option("--class_prior",
              default=None,
              show_default=True,
              help="类的先验概率，格式为逗号分隔的浮点数，如'0.2,0.8'")
@click.option("--binarize",
              default=None,
              show_default=True,
              help="二值化阈值（仅用于BernoulliNB）。如果为None，假设输入已经是二进制的")
@click.option("--var_smoothing",
              default=None,
              show_default=True,
              help="方差平滑参数（仅用于GaussianNB），范围为[0, +∞]，默认为1e-9")
@click.option("--norm",
              type=click.Choice(['True', 'False']),
              default='False',
              show_default=True,
              help="是否对计数进行二阶规范化（仅用于ComplementNB）")
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
def naive_bayes(genotype, phenotype, vgenotype, vphenotype, feature, testsize, seed, nb_type, alpha, fit_prior,
                class_prior, binarize, var_smoothing, norm, scoring, kfolds, n_bootstrap, out):
    """
    使用朴素贝叶斯进行模型构建

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
                                                      random_seed=seed,
                                                      test_size=testsize,
                                                      features=features)

    n_class = len(np.unique(y_train))

    X_train_scale = X_train
    X_test_scale = X_test

    logger.info("Parameter optimization using grid search")
    param_grid = {}

    # 根据不同的朴素贝叶斯类型设置参数
    if nb_type == 'multinomial':
        base_model = MultinomialNB()
        if alpha:
            if '_' in alpha:
                param_grid["alpha"] = np.arange(*[float(i) for i in alpha.split("_")])
            elif ',' in alpha:
                param_grid["alpha"] = [float(i) for i in alpha.split(",")]
            else:
                param_grid["alpha"] = [float(alpha)]

        if fit_prior:
            param_grid["fit_prior"] = [fit_prior == 'True']

    elif nb_type == 'gaussian':
        base_model = GaussianNB()
        if var_smoothing:
            if '_' in var_smoothing:
                param_grid["var_smoothing"] = np.logspace(*[float(i) for i in var_smoothing.split("_")])
            elif ',' in var_smoothing:
                param_grid["var_smoothing"] = [float(i) for i in var_smoothing.split(",")]
            else:
                param_grid["var_smoothing"] = [float(var_smoothing)]

    elif nb_type == 'bernoulli':
        base_model = BernoulliNB()
        if alpha:
            if '_' in alpha:
                param_grid["alpha"] = np.arange(*[float(i) for i in alpha.split("_")])
            elif ',' in alpha:
                param_grid["alpha"] = [float(i) for i in alpha.split(",")]
            else:
                param_grid["alpha"] = [float(alpha)]

        if binarize:
            if '_' in binarize:
                param_grid["binarize"] = np.arange(*[float(i) for i in binarize.split("_")])
            elif ',' in binarize:
                param_grid["binarize"] = [float(i) for i in binarize.split(",")]
            else:
                param_grid["binarize"] = [float(binarize)]

        if fit_prior:
            param_grid["fit_prior"] = [fit_prior == 'True']

    elif nb_type == 'complement':
        base_model = ComplementNB()
        if alpha:
            if '_' in alpha:
                param_grid["alpha"] = np.arange(*[float(i) for i in alpha.split("_")])
            elif ',' in alpha:
                param_grid["alpha"] = [float(i) for i in alpha.split(",")]
            else:
                param_grid["alpha"] = [float(alpha)]

        if norm:
            param_grid["norm"] = [norm == 'True']

        if fit_prior:
            param_grid["fit_prior"] = [fit_prior == 'True']

    # 处理class_prior参数（适用于除GaussianNB外的所有类型）
    if class_prior and nb_type != 'gaussian':
        prior_values = [float(i) for i in class_prior.split(",")]
        param_grid["class_prior"] = [prior_values]

    # 如果参数网格为空，添加默认参数
    if not param_grid:
        param_grid = {'fit_prior': [True]}  # 默认参数

    param_num = 1
    for i in param_grid:
        param_num *= len(param_grid[i])
    logger.info(f"参数空间大小: {param_num}")

    logger.info("Get the best parameters")
    stratified_cv = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=seed)
    gc = GridSearchCV(base_model,
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
                        title=f'Naive Bayes ({nb_type})')

    logger.info("ROC & PR curves Plot")
    if n_class > 2:
        # TODO: 多分类
        pass
    else:
        f_roc_pr_plot = d_out.joinpath("performance_curves.pdf")
        plot_roc_pr_curves(best_model, X_test_scale, y_test, f_roc_pr_plot, seed=seed, n_bootstraps=n_bootstrap)

    logger.info("Confusion matrix Plot")
    f_confusion_matrix_plot = d_out.joinpath("confusion_matrix.pdf")
    plot_confusion_matrix(best_model, X_test_scale, y_test, f_confusion_matrix_plot,
                          title=f"Naive Bayes ({nb_type})")


if __name__ == "__main__":
    naive_bayes()
