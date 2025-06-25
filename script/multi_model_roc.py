#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: CB Ren
# @Created Date:   2024/10/21 22:01
from pathlib import Path
import logging
import click
import pickle

from utils import auto_load_data, get_features, evaluate_models_on_test
from figures import multi_roc_pr_plot

logger = logging.getLogger(__file__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-vg", "--vgenotype",
              type=click.Path(),
              help="The validation genotype input file(must be provided with --vphenotype)")
@click.option("-vp", "--vphenotype",
              type=click.Path(),
              help="The validation phenotype input file(must be provided with --vgenotype)")
@click.option("--feature",
              type=click.Path(),
              help="A file containing features to be used")
@click.option("-dt", "--decisiontree",
              type=click.Path(),
              help="Decisiontree model")
@click.option("-rf", "--randomforest",
              type=click.Path(),
              help="Random forest model")
@click.option("-svm", "--svm",
              type=click.Path(),
              help="Random SVM model")
@click.option("-xgb", "--xgboost",
              type=click.Path(),
              help="Random XGBoost model")
@click.option("-mlp", "--mlpclassifier",
              type=click.Path(),
              help="Random MLPClassifier model")
@click.option("-lr", "--logisticregression",
              type=click.Path(),
              help="Logistic Regression model")
@click.option("-nb", "--naivebayes",
              type=click.Path(),
              help="NaiveBayes model")
@click.option("-g", "--genotype",
              type=click.Path(),
              help="The genotype input file")
@click.option("-p", "--phenotype",
              type=click.Path(),
              help="The phenotype input file")
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
@click.option("--n_bootstrap",
              type=int,
              default=1000,
              show_default=True,
              help="Number of bootstrap iterations for stability selection")
@click.option("--show_ci",
              type=click.BOOL,
              default=False,
              show_default=True,
              help="Whether to show confidence interval for ROC curve")
@click.option("-o", "--out",
              type=click.Path(),
              required=True,
              help="The output dir")
def main(vgenotype, vphenotype, feature, decisiontree, randomforest, svm, xgboost, mlpclassifier, logisticregression,
         naivebayes, genotype, phenotype, testsize, seed, n_bootstrap, show_ci, out):
    """
    多种机器学习模型预测结果ROC曲线绘制
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

    X_train_scale = X_train
    X_test_scale = X_test

    data_roc = {}
    if decisiontree:
        f_dt = Path(decisiontree).absolute()
        logger.info(f"Load Decision Tree model from {f_dt}.")
        model = pickle.load(open(f_dt, 'rb'))
        data_roc["Decision Tree"] = model
    if randomforest:
        f_rf = Path(randomforest).absolute()
        logger.info(f"Load RandomForest model from {f_rf}.")
        model = pickle.load(open(f_rf, 'rb'))
        data_roc["Random Forest"] = model
    if svm:
        f_svm = Path(svm).absolute()
        logger.info(f"Load SVC model from {f_svm}.")
        model = pickle.load(open(f_svm, 'rb'))
        data_roc["SVM"] = model
    if xgboost:
        f_xgb = Path(xgboost).absolute()
        logger.info(f"Load XGBoost model from {f_xgb}.")
        model = pickle.load(open(f_xgb, 'rb'))
        data_roc["XGBoost"] = model
    if mlpclassifier:
        f_mlpc = Path(mlpclassifier).absolute()
        logger.info(f"Load MLPClassifier model from {f_mlpc}.")
        model = pickle.load(open(f_mlpc, 'rb'))
        data_roc["MLP"] = model
    if logisticregression:
        f_lr = Path(logisticregression).absolute()
        logger.info(f"Load Logistic Regression model from {f_lr}.")
        model = pickle.load(open(f_lr, 'rb'))
        data_roc["Logistic Regression"] = model
    if naivebayes:
        f_nb = Path(naivebayes).absolute()
        logger.info(f"Load Naive Bayes model from {f_nb}.")
        model = pickle.load(open(f_nb, 'rb'))
        data_roc["Naive Bayes"] = model

    if len(data_roc) == 0:
        raise ValueError("No model is loaded.")

    logger.info(f"Plot ROC curve.")
    multi_roc_pr_plot(data_roc, X_test_scale, y_test, d_out, seed=seed, n_bootstraps=n_bootstrap, show_ci=show_ci)
    logger.info(f"Stat the models")
    evaluate_models_on_test(data_roc, X_test_scale, y_test, d_out)


if __name__ == "__main__":
    main()
