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

from xgboost.sklearn import XGBClassifier

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
              help="The random seed for split dataset and xgb model training")
@click.option("--n_estimators",
              default=None,
              show_default=True,
              help="弱分类器的数量，范围为[0, +∞]，默认为100")
@click.option("--learning_rate",
              default=None,
              show_default=True,
              help="学习速率，控制每次迭代更新权重的步长，较小的值意味着更多的迭代次数")
@click.option("--colsample_bytree",
              default=None,
              show_default=True,
              help="列采样比率，用于随机选择特征列，范围在[0,1]之间，默认为1.0表示所有列都被采样")
@click.option("--gamma",
              default=None,
              show_default=True,
              help="节点分裂所需的最小损失函数下降值，范围在[0,+∞]。值越大，算法越保守")
@click.option("--reg_lambda",
              default=None,
              show_default=True,
              help="L2正则化项参数，用于惩罚过拟合，范围在[0,+∞]。值越大，算法越保守")
@click.option("--min_child_weight",
              default=None,
              show_default=True,
              help="叶子节点的最小权重，用于防止过拟合")
@click.option("--max_depth",
              default=None,
              show_default=True,
              help="最大深度，整数类型，范围为[0, +∞]，默认无穷大")
@click.option("--reg_alpha",
              default=None,
              show_default=True,
              help="L1正则化项参数，用于惩罚过拟合，范围在[0,+∞]。值越大，算法越保守")
@click.option("--subsample",
              default=None,
              show_default=True,
              help="行采样比率，用于随机选择样本行，范围在[0,1]之间，默认为1.0表示所有行都被采样")
@click.option("--objective",
              # type=click.Choice(['regression', 'binary:logistic', 'multi:softprob']),
              default="binary:logistic",
              show_default=True,
              help="学习任务于学习目标，可选取回归、二分类或多类别分类")
@click.option("--scale_pos_weight",
              default=None,
              show_default=True,
              help="正负样本比例，用于平衡正负样本的重要性，范围在[0,+∞]。值越大，正负样本的权重越大")
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
def xgboost(genotype, phenotype, vgenotype, vphenotype, feature, testsize, seed, n_estimators, learning_rate,
            colsample_bytree, gamma, reg_lambda, min_child_weight, max_depth, reg_alpha, subsample, objective,
            scale_pos_weight, scoring, kfolds, n_bootstrap, out):
    """
    使用XGBoost进行模型构建

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

    if learning_rate:
        if '_' in learning_rate:
            param_grid["learning_rate"] = np.arange(*[float(i) for i in learning_rate.split("_")])
        elif ',' in learning_rate:
            param_grid["learning_rate"] = [float(i) for i in learning_rate.split(",")]
        else:
            param_grid["learning_rate"] = [float(learning_rate)]

    if colsample_bytree:
        if '_' in colsample_bytree:
            param_grid["colsample_bytree"] = np.arange(*[float(i) for i in colsample_bytree.split("_")])
        elif ',' in colsample_bytree:
            param_grid["colsample_bytree"] = [float(i) for i in colsample_bytree.split(",")]
        else:
            param_grid["colsample_bytree"] = [float(colsample_bytree)]

    if gamma:
        if '_' in gamma:
            param_grid["gamma"] = np.arange(*[float(i) for i in gamma.split("_")])
        elif ',' in gamma:
            param_grid["gamma"] = [float(i) for i in gamma.split(",")]
        else:
            param_grid["gamma"] = [float(gamma)]

    if reg_lambda:
        if '_' in reg_lambda:
            param_grid["reg_lambda"] = np.arange(*[float(i) for i in reg_lambda.split("_")])
        elif ',' in reg_lambda:
            param_grid["reg_lambda"] = [float(i) for i in reg_lambda.split(",")]
        else:
            param_grid["reg_lambda"] = [float(reg_lambda)]

    if min_child_weight:
        if '_' in min_child_weight:
            param_grid["min_child_weight"] = np.arange(*[float(i) for i in min_child_weight.split("_")])
        elif ',' in min_child_weight:
            param_grid["min_child_weight"] = [float(i) for i in min_child_weight.split(",")]
        else:
            param_grid["min_child_weight"] = [float(min_child_weight)]

    if max_depth:
        if '_' in max_depth:
            param_grid["max_depth"] = np.arange(*[int(i) for i in max_depth.split("_")])
        elif ',' in max_depth:
            param_grid["max_depth"] = [int(i) for i in max_depth.split(",")]
        else:
            param_grid["max_depth"] = [int(max_depth)]

    if reg_alpha:
        if '_' in reg_alpha:
            param_grid["reg_alpha"] = np.arange(*[float(i) for i in reg_alpha.split("_")])
        elif ',' in reg_alpha:
            param_grid["reg_alpha"] = [float(i) for i in reg_alpha.split(",")]
        else:
            param_grid["reg_alpha"] = [float(reg_alpha)]

    if subsample:
        if '_' in subsample:
            param_grid["subsample"] = np.arange(*[float(i) for i in subsample.split("_")])
        elif ',' in subsample:
            param_grid["subsample"] = [float(i) for i in subsample.split(",")]
        else:
            param_grid["subsample"] = [float(subsample)]

    if ',' in objective:
        param_grid["objective"] = [obj for obj in objective.split(",")]
    else:
        param_grid["objective"] = [objective]

    if scale_pos_weight:
        if '_' in scale_pos_weight:
            param_grid["scale_pos_weight"] = np.arange(*[float(i) for i in scale_pos_weight.split("_")])
        elif ',' in scale_pos_weight:
            param_grid["scale_pos_weight"] = [float(i) for i in scale_pos_weight.split(",")]
        else:
            param_grid["scale_pos_weight"] = [float(scale_pos_weight)]

    param_num = 1
    for i in param_grid:
        param_num *= len(param_grid[i])
    logger.info(f"参数空间大小: {param_num}")

    logger.info("Get the best parameters")
    model = XGBClassifier(random_state=seed)
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
                        title='XGBoost')

    logger.info("ROC & PR curves Plot")
    if n_class > 2:
        # TODO: 多分类
        pass
    else:
        f_roc_pr_plot = d_out.joinpath("performance_curves.pdf")
        plot_roc_pr_curves(best_model, X_test_scale, y_test, f_roc_pr_plot, seed=seed, n_bootstraps=n_bootstrap)

    logger.info("Confusion matrix Plot")
    f_confusion_matrix_plot = d_out.joinpath("confusion_matrix.pdf")
    plot_confusion_matrix(best_model, X_test_scale, y_test, f_confusion_matrix_plot, title="XGBoost")


if __name__ == "__main__":
    xgboost()
