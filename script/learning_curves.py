#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Ming Jia
# @Created Date:   2024/10/16 9:54
import json
from pathlib import Path
import logging
import click
import pandas as pd

from utils import get_features, plot_learning_curves

logger = logging.getLogger(__file__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-g', '--genotype',
              required=True,
              type=click.Path(),
              help="The genotype input file")
@click.option('-p', '--phenotype',
              required=True,
              type=click.Path(),
              help="The phenotype input file")
@click.option('-m', '--model',
              required=True,
              type=click.Choice(['rf', 'svm', 'xgb', 'mlp'], case_sensitive=False),
              help="The name of the machine learning model")
@click.option('--parameters',
              type=click.Path(),
              help="A json parameters file for the model")
@click.option('--feature',
              type=click.Path(),
              help="A file containing features to be used")
@click.option("--seed",
              type=int,
              default=42,
              show_default=True,
              help="The random seed for split dataset and train")
@click.option("-o", "--out",
              type=click.Path(),
              required=True,
              help="The output dir")
def main(genotype, phenotype, model, parameters, feature, seed, out):
    """
    学习曲线绘制
    """
    f_genotype = Path(genotype).absolute()
    f_phenotype = Path(phenotype).absolute()
    d_out = Path(out).absolute()
    d_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Read the genotype file: {f_genotype}")
    df_genotype = pd.read_csv(f_genotype, index_col=0, sep="\t")
    logger.info(f"Read the phenotype file: {f_phenotype}")
    df_phenotype = pd.read_csv(f_phenotype, index_col=0, sep="\t")

    if feature:
        f_feature = Path(feature).resolve()
        logger.info(f"Read the feature file: {f_feature}")
        features = get_features(f_feature)
        df_genotype = df_genotype.loc[:, features]

    params = {}
    if parameters:
        f_parameter = Path(parameters).resolve()
        logger.info(f"Read the parameter file: {f_parameter}")
        with open(f_parameter, "r") as IN:
            params = json.load(IN)
    if "random_state" not in params:
        params["random_state"] = seed

    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(**params)
        title = "Random Forest"
    elif model == "svm":
        from sklearn.svm import SVC
        estimator = SVC(**params)
        title = "SVM"
    elif model == "xgb":
        from xgboost.sklearn import XGBClassifier
        estimator = XGBClassifier(**params)
        title = "XGBoost"
    elif model == "mlp":
        from sklearn.neural_network import MLPClassifier
        estimator = MLPClassifier(**params)
        title = "MLP"
    else:
        raise ValueError("Invalid model name")

    logger.info(f"Start to plot the learning curve")
    plot_learning_curves(estimator, title, df_genotype, df_phenotype.values.ravel(), out=d_out)


if __name__ == "__main__":
    main()
