#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: CB Ren
# @Created Date:   2024/10/19 8:45
from pathlib import Path
import logging
import click
import pickle
import json
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from utils import get_features, auto_load_data, NumpyEncoder
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
              help="The random seed for split dataset and mplc model training")
@click.option("--hidden_layer_sizes",
              type=str,
              help="隐藏层的数量和每层的神经元数")
@click.option("--activation",
              type=str,
              help="激活函数")
@click.option("--solver",
              type=str,
              help="权重优化的算法")
@click.option("--alpha",
              type=str,
              help="L2正则化参数")
@click.option("--learning_rate_init",
              type=str,
              help="初始学习率(仅用于'sgd'或'adam')")
@click.option("--max_iter",
              type=str,
              help="最大迭代次数")
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
def main(genotype, phenotype, vgenotype, vphenotype, feature, testsize, seed, hidden_layer_sizes, activation, solver,
         alpha, learning_rate_init, max_iter, scoring, kfolds, n_bootstrap, out):
    """
    使用多层感知分类器（MLPC）进行模型构建

    如果不提供测试集，则会自动按照指定规则从原始数据集分割出训练集和测试集。
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

    X_train_scale = X_train
    X_test_scale = X_test

    logger.info("Parameter optimization using grid search")
    param_grid = {}

    if hidden_layer_sizes:
        if ',' in hidden_layer_sizes:
            hidden_layer_size = []
            for i in hidden_layer_sizes.strip().split(','):
                num_layer, num_neuron = i.strip().split('_')
                num_layer = int(num_layer)
                if num_neuron != "":
                    num_neuron = int(num_neuron)
                    hidden_layer_size.append((num_layer, num_neuron))
                else:
                    hidden_layer_size.append((num_layer,))
        else:
            hidden_layer_size = []
            num_layer, num_neuron = hidden_layer_sizes.strip().split('_')
            num_layer = int(num_layer)
            if num_neuron != "":
                num_neuron = int(num_neuron)
                hidden_layer_size.append((num_layer, num_neuron))
            else:
                hidden_layer_size.append((num_layer,))
        param_grid['hidden_layer_sizes'] = hidden_layer_size

    if activation:
        if ',' in activation:
            activations = [i.strip() for i in activation.strip().split(',')]
        else:
            activations = [activation.strip()]
        param_grid['activation'] = activations

    if solver:
        if ',' in solver:
            solvers = [i.strip() for i in solver.strip().split(',')]
        else:
            solvers = [solver.strip()]
        param_grid['solver'] = solvers

    if alpha:
        if '_' in alpha:
            alphas = list(np.arange(*[float(i) for i in alpha.split("_")]))
        elif ',' in alpha:
            alphas = [float(i) for i in alpha.strip().split(",")]
        else:
            alphas = [float(alpha)]
        param_grid['alpha'] = alphas

    if learning_rate_init:
        if '_' in learning_rate_init:
            lrs = list(np.arange(*[float(i) for i in learning_rate_init.split("_")]))
        elif ',' in learning_rate_init:
            lrs = [float(i) for i in learning_rate_init.strip().split(",")]
        else:
            lrs = [float(learning_rate_init)]
        param_grid['learning_rate_init'] = lrs

    if max_iter:
        if '_' in max_iter:
            max_iters = list(np.arange(*[int(i) for i in max_iter.split("_")]))
        elif ',' in max_iter:
            max_iters = [int(i) for i in max_iter.stirp().split(",")]
        else:
            max_iters = [int(max_iter)]
        param_grid['max_iter'] = max_iters

    param_num = 1
    for i in param_grid:
        param_num *= len(param_grid[i])
    logger.info(f"参数空间大小: {param_num}")

    logger.info("Get the best parameters")
    model = MLPClassifier(random_state=seed)
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
                        title='MLP')

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
    plot_confusion_matrix(best_model, X_test_scale, y_test, f_confusion_matrix_plot, title="MLP")


if __name__ == "__main__":
    main()
