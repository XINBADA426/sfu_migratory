#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Chaobo Ren
# @Date: 2024-09-26 07:43:26
# @Description: "使用vcf文件进行群体预测"
# Path: /data1/NFS/home/rcb/project/YYR2024082901
import pickle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    mean_squared_error,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import gzip
import click
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Functions
def get_features(f_feature_list):
    """
    从特征文件中读取特征列表
    """
    res = set()
    with open(f_feature_list, "r") as IN:
        for line in IN:
            res.add(line.strip())
    return list(res)


def binary_classification_plot(fpr, tpr, auc_val, d_out: Path):
    """
    绘制二分类ROC曲线

    :param fpr: 假阳性率
    :param tpr: 真阳性率
    :param auc_val: AUC值
    :param d_out: 输出目录
    """
    figure, axis = plt.subplots(figsize=(10, 8))

    axis.plot([0, 1], [0, 1], "k--", label="Random classifier")
    axis.plot(fpr, tpr, "r", linewidth=3, label=f"ROC curve {auc_val:.4}")
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


# Main
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """
    使用vcf文件进行群体预测的工具
    """
    pass


@click.command()
@click.option(
    "-v", "--vcf", type=click.Path(), required=True, help="The vcf input file"
)
@click.option(
    "-n",
    "--name",
    type=click.Choice(["id", "chrpos"]),
    required=True,
    help="Use the ID column or chromosome + position as name",
)
@click.option(
    "--feature",
    type=click.Path(),
    help="A file containing features to be used",
)
@click.option(
    "-m",
    "--miss",
    type=int,
    default=0,
    show_default=True,
    help="Regard the missing genotypes as 0(纯合参考) or 1(杂合) or 2(纯合突变)",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(),
    required=True,
    help="The output file name",
)
def vcfconvert(vcf, name, feature, miss, out):
    """
    将vcf文件转为后续机器学习所需要的格式
    """
    f_vcf = Path(vcf).absolute()
    f_out = Path(out).absolute()
    d_out = f_out.parent
    d_out.mkdir(parents=True, exist_ok=True)

    # 1. 读取vcf文件
    logger.info(f"Read the vcf file: {f_vcf}")
    info = {}
    if f_vcf.suffix == ".gz":
        handle = gzip.open(f_vcf, "rt")
    else:
        handle = open(f_vcf, "r")
    for line in handle:
        if line.startswith("##"):
            continue
        elif line.startswith("#"):
            samples = line.strip().split("\t")[9:]
        else:
            arr = line.strip().split("\t")

            if name == "id":
                key = arr[2]
            elif name == "chrpos":
                key = "__".join([arr[0], arr[1]])
            else:
                raise ValueError(f"Unknown name option: {name}")
            info[key] = []

            for i in arr[9:]:
                gt = i.strip().split(":")[0]
                if gt == "0/0":
                    info[key].append(0)
                elif gt == "0/1":
                    info[key].append(1)
                elif gt == "1/1":
                    info[key].append(2)
                else:
                    info[key].append(miss)
    handle.close()

    # 2. 将info结合sample信息转为data.frame
    vcf_data = pd.DataFrame(info, index=samples)

    if feature:
        f_feature = Path(feature).absolute()
        logger.info(f"Read the feature file: {f_feature}")
        features = get_features(f_feature)
        vcf_data = vcf_data.loc[:, features]

    logger.info(f"Save result to {f_out}")
    vcf_data.to_csv(f_out, index=True, index_label="Sample", sep="\t")


@click.command()
@click.option(
    "-g", "--genotype", type=click.Path(), required=True, help="The genotype input file"
)
@click.option(
    "-p",
    "--phenotype",
    type=click.Path(),
    required=True,
    help="The phenotype input file",
)
@click.option(
    "--alpha",
    type=float,
    default=0,
    show_default=True,
    help="The alpha used by LASSO, if 0, will be selected automatically",
)
@click.option(
    "--maxiter",
    type=int,
    default=10000,
    show_default=True,
    help="The maximum number of iterations for LASSO",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="The random seed for split dataset and LASSO",
)
@click.option(
    "--testsize",
    type=float,
    default=0.25,
    show_default=True,
    help="The proportion of the dataset that should be used for testing",
)
@click.option(
    "-pfx",
    "--prefix",
    type=click.Path(),
    required=True,
    help="The output prefix",
)
def lasso(genotype, phenotype, alpha, maxiter, seed, testsize, prefix):
    """
    使用lasso进行特征选择
    """
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error

    f_genotype = Path(genotype).absolute()
    f_phenotype = Path(phenotype).absolute()
    f_feature = Path(f"{prefix}.feature.list").absolute()
    f_report = Path(f"{prefix}.report.txt").absolute()
    d_out = f_feature.parent
    d_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Read the genotype file: {f_genotype}")
    df_genotype = pd.read_csv(f_genotype, index_col=0, sep="\t")

    logger.info(f"Read the phenotype file: {f_phenotype}")
    df_phenotype = pd.read_csv(f_phenotype, index_col=0, sep="\t")

    logger.info("Split the dataset into training and testing sets")
    df_genotype_train, df_genotype_test, df_phenotype_train, df_phenotype_test = (
        train_test_split(
            df_genotype, df_phenotype, test_size=testsize, random_state=seed
        )
    )

    if alpha == 0:
        logger.info("Get the best alpha value")
        # 定义alpha的范围
        param_grid = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
        # 使用GridSearchCV进行交叉验证
        lasso = Lasso(max_iter=10000)
        grid_search = GridSearchCV(
            lasso, param_grid, cv=5, scoring="neg_mean_squared_error"
        )
        grid_search.fit(df_genotype_train, df_phenotype_train)
        # 输出最佳alpha值
        best_alpha = grid_search.best_params_["alpha"]
        logger.info(f"最佳alpha值: {best_alpha}")
    else:
        best_alpha = alpha

    logger.info("Lasso regression for feature selection")
    model = Lasso(alpha=best_alpha, max_iter=maxiter, random_state=seed).fit(
        df_genotype_train, df_phenotype_train
    )
    df_phenotype_test_pred = model.predict(df_genotype_test)
    train_score = model.score(df_genotype_train, df_phenotype_train)
    test_score = model.score(df_genotype_test, df_phenotype_test)
    val_mse = mean_squared_error(df_phenotype_test, df_phenotype_test_pred)

    logger.info(f"训练集score: {train_score:.2f}")
    logger.info(f"测试集score: {test_score:.2f}")
    logger.info(f"验证集上的均方误差: {val_mse:.2f}")
    logger.info(f"保存报告到文件:{f_report}")
    with open(f_report, "w") as OUT:
        print(f"Best alpha: {best_alpha}", file=OUT)
        print(f"Training score: {train_score:.2f}", file=OUT)
        print(f"Testing score: {test_score:.2f}", file=OUT)
        print(f"Validation MSE: {val_mse:.2f}", file=OUT)

    logger.info(f"保存选择的特征结果到文件:{f_feature}")
    retained_features = df_genotype.columns[model.coef_ != 0]
    with open(f_feature, "w") as OUT:
        print(*retained_features, sep="\n", file=OUT)


@click.command()
@click.option(
    "-g", "--genotype", type=click.Path(), required=True, help="The genotype input file"
)
@click.option(
    "-p",
    "--phenotype",
    type=click.Path(),
    required=True,
    help="The phenotype input file",
)
@click.option(
    "--feature",
    type=click.Path(),
    help="A file containing features to be used",
)
@click.option(
    "--testsize",
    type=float,
    default=0.1,
    show_default=True,
    help="The proportion of the dataset that should be used for testing",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="The random seed for split dataset and random forest",
)
@click.option(
    "--n_estimators",
    type=int,
    help="Number of trees in the forest, if not specified, it will be automaticly determined",
)
@click.option(
    "--max_depth",
    type=int,
    help="Maximum depth of each tree, if not specified, it will be automaticly determined",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(),
    required=True,
    help="The output dir",
)
def randomforest(
    genotype, phenotype, feature, testsize, seed, n_estimators, max_depth, out
):
    """
    使用随机森林进行模型构建
    """
    from sklearn.ensemble import RandomForestClassifier

    f_genotype = Path(genotype).absolute()
    f_phenotype = Path(phenotype).absolute()
    d_out = Path(out).absolute()
    d_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Read the genotype file: {f_genotype}")
    df_genotype = pd.read_csv(f_genotype, index_col=0, sep="\t")
    if feature:
        f_feature = Path(feature).absolute()
        logger.info(f"Read the feature file: {f_feature}")
        features = get_features(f_feature)
        df_genotype = df_genotype.loc[:, features]

    logger.info(f"Read the phenotype file: {f_phenotype}")
    df_phenotype = pd.read_csv(f_phenotype, index_col=0, sep="\t")

    logger.info("Split the dataset into training and testing sets")
    df_genotype_train, df_genotype_test, df_phenotype_train, df_phenotype_test = (
        train_test_split(
            df_genotype, df_phenotype, test_size=testsize, random_state=seed
        )
    )

    param_grid = {
        "n_estimators": [50, 100, 200, 400, 800, 1600],
        "max_depth": [3, 5, 7, 9, 11],
    }
    if n_estimators and max_depth:
        best_n_estimators = n_estimators
        best_max_depth = max_depth
    else:
        logger.info("Get the best parameters")
        if n_estimators:
            param_grid["n_estimators"] = [n_estimators]
        if max_depth:
            param_grid["max_depth"] = [max_depth]
        model = RandomForestClassifier(random_state=seed)
        gc = GridSearchCV(model, param_grid, cv=5)
        gc.fit(df_genotype_train, df_phenotype_train.values.ravel())
        best_n_estimators = gc.best_params_["n_estimators"]
        best_max_depth = gc.best_params_["max_depth"]

    logger.info(
        f"Use n_estimators = {best_n_estimators} max_depth = {best_max_depth} to do randomforest classification"
    )
    model = RandomForestClassifier(
        n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=seed
    )
    model.fit(df_genotype_train, df_phenotype_train.values.ravel())
    df_phenotype_test_pred = model.predict(df_genotype_test)
    train_score = model.score(
        df_genotype_train, df_phenotype_train.values.ravel())
    test_score = model.score(
        df_genotype_test, df_phenotype_test.values.ravel())
    val_mse = mean_squared_error(df_phenotype_test, df_phenotype_test_pred)
    info_score = {
        "训练集得分": train_score,
        "测试集得分": test_score,
        "验证集上的均方误差": val_mse,
    }
    logger.info(info_score)

    # 结果保存
    f_score = d_out.joinpath("score.tsv")
    logger.info(f"保存得分信息到文件: {f_score}")
    with open(f_score, "w") as OUT:
        for k, v in info_score.items():
            print(f"{k}\t{v}", file=OUT)

    f_param = d_out.joinpath("param.json")
    logger.info(f"保存参数到文件:{f_param}")
    with open(f_param, "w") as OUT:
        json.dump({"n_estimators": best_n_estimators,
                  "max_depth": best_max_depth}, OUT)

    f_report = d_out.joinpath("report.txt")
    logger.info(f"保存报告到文件:{f_report}")
    report = classification_report(df_phenotype_test, df_phenotype_test_pred)
    with open(f_report, "w") as OUT:
        print(report, file=OUT)

    f_model = d_out.joinpath("model.pkl")
    logger.info(f"保存模型到文件:{f_model}")
    pickle.dump(model, open(f_model, "wb"))

    logger.info("ROC Plot")
    n_class = df_phenotype.iloc[:, 0].nunique()
    if n_class > 2:
        # TODO: 多分类
        phenotype_test_one_hot = label_binarize(
            df_phenotype_test, classes=np.arange(n_class)
        )
        phenotype_test_one_hot_hat = model.predict_proba(df_genotype_test)
        fpr, tpr, _ = roc_curve(
            phenotype_test_one_hot, phenotype_test_one_hot_hat.ravel()
        )
        # auc = roc_auc_score(fpr, tpr, average="macro")
    else:
        # 二分类
        phenotype_test_prob = model.predict_proba(df_genotype_test)[:, 1]
        fpr, tpr, _ = roc_curve(
            df_phenotype_test.values.ravel(), phenotype_test_prob)
        auc_val = roc_auc_score(
            df_phenotype_test.values.ravel(), phenotype_test_prob)
        binary_classification_plot(fpr, tpr, auc_val, d_out)


@click.command()
@click.option(
    "-g",
    "--genotype",
    type=click.Path(),
    required=True,
    help="The genotype input file"
)
@click.option(
    "-p",
    "--phenotype",
    type=click.Path(),
    required=True,
    help="The phenotype input file",
)
@click.option(
    "--feature",
    type=click.Path(),
    help="A file containing features to be used",
)
@click.option(
    "--testsize",
    type=float,
    default=0.1,
    show_default=True,
    help="The proportion of the dataset that should be used for testing",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="The random seed for split dataset",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(),
    required=True,
    help="The output dir",
)
def xgboost(genotype, phenotype, feature, testsize, seed, out):
    """
    使用XGBoost进行模型构建
    """
    import xgboost as xgb

    f_genotype = Path(genotype).absolute()
    f_phenotype = Path(phenotype).absolute()
    d_out = Path(out).absolute()
    d_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Read the genotype file: {f_genotype}")
    df_genotype = pd.read_csv(f_genotype, index_col=0, sep="\t")
    if feature:
        f_feature = Path(feature).absolute()
        logger.info(f"Read the feature file: {f_feature}")
        features = get_features(f_feature)
        df_genotype = df_genotype.loc[:, features]

    logger.info(f"Read the phenotype file: {f_phenotype}")
    df_phenotype = pd.read_csv(f_phenotype, index_col=0, sep="\t")

    logger.info("Split the dataset into training and testing sets")
    df_genotype_train, df_genotype_test, df_phenotype_train, df_phenotype_test = (
        train_test_split(
            df_genotype, df_phenotype, test_size=testsize, random_state=seed
        )
    )

    # # 使用 LabelEncoder 将类别标签转换为数值型
    # label_encoder = LabelEncoder()
    # phenotype_encoded = label_encoder.fit_transform(df_phenotype.values.ravel())

    logger.info("XGBoost classification")
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
    )
    model.fit(df_genotype, phenotype_encoded)

    logger.info("Performing cross-validation")
    cv_scores = cross_val_score(
        model, df_genotype, phenotype_encoded, cv=5
    )  # 5折交叉验证
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean cross-validation score: {cv_scores.mean()}")

    logger.info(f"保存模型到文件:{f_out}")
    pickle.dump(model, open(f_out, "wb"))


@click.command()
@click.option(
    "-g", "--genotype", type=click.Path(), required=True, help="The genotype input file"
)
@click.option(
    "-p",
    "--phenotype",
    type=click.Path(),
    required=True,
    help="The phenotype input file",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(),
    required=True,
    help="The output file name",
)
def svm(genotype, phenotype, out):
    """
    使用支持向量机进行分类
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report

    f_genotype = Path(genotype).absolute()
    f_phenotype = Path(phenotype).absolute()
    f_out = Path(out).absolute()
    d_out = f_out.parent
    d_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Read the genotype file: {f_genotype}")
    df_genotype = pd.read_csv(f_genotype, index_col=0, sep="\t")

    logger.info(f"Read the phenotype file: {f_phenotype}")
    df_phenotype = pd.read_csv(f_phenotype, index_col=0, sep="\t")

    # logger.info("Split the dataset into training and testing sets")
    # df_genotype_train, df_genotype_test, df_phenotype_train, df_phenotype_test = (
    #     train_test_split(df_genotype, df_phenotype, test_size=0.2, random_state=42)
    # )

    logger.info("SVM classification")
    model = SVC(kernel="linear")  # 使用线性核
    model.fit(df_genotype, df_phenotype.values.ravel())

    # logger.info("Evaluate the model")
    # predictions = model.predict(df_genotype_test)
    # report = classification_report(df_phenotype_test, predictions)
    # logger.info(f"Classification Report:\n{report}")

    logger.info("Performing cross-validation")
    cv_scores = cross_val_score(
        model, df_genotype, df_phenotype.values.ravel(), cv=5
    )  # 5折交叉验证
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean cross-validation score: {cv_scores.mean()}")

    logger.info(f"保存模型到文件:{f_out}")
    pickle.dump(model, open(f_out, "wb"))


@click.command()
@click.option(
    "-g", "--genotype", type=click.Path(), required=True, help="The genotype input file"
)
@click.option(
    "-p",
    "--phenotype",
    type=click.Path(),
    required=True,
    help="The phenotype input file",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(),
    required=True,
    help="The output file name",
)
def nn(genotype, phenotype, out):
    """
    使用神经网络进行分类
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report

    f_genotype = Path(genotype).absolute()
    f_phenotype = Path(phenotype).absolute()
    f_out = Path(out).absolute()
    d_out = f_out.parent
    d_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Read the genotype file: {f_genotype}")
    df_genotype = pd.read_csv(f_genotype, index_col=0)

    logger.info(f"Read the phenotype file: {f_phenotype}")
    df_phenotype = pd.read_csv(f_phenotype, index_col=0)

    logger.info("Split the dataset into training and testing sets")
    df_genotype_train, df_genotype_test, df_phenotype_train, df_phenotype_test = (
        train_test_split(df_genotype, df_phenotype,
                         test_size=0.2, random_state=42)
    )

    logger.info("Neural Network classification")
    model = MLPClassifier(
        hidden_layer_sizes=(100,), max_iter=1000, random_state=42
    )  # 100个神经元的隐藏层
    model.fit(df_genotype_train, df_phenotype_train.values.ravel())

    logger.info("Evaluate the model")
    predictions = model.predict(df_genotype_test)
    report = classification_report(
        df_phenotype_test, predictions, zero_division=1)
    logger.info(f"Classification Report:\n{report}")

    logger.info("Performing cross-validation")
    cv_scores = cross_val_score(
        model, df_genotype, df_phenotype.values.ravel(), cv=5
    )  # 5折交叉验证
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean cross-validation score: {cv_scores.mean()}")

    logger.info(f"保存模型到文件:{f_out}")
    pickle.dump(model, open(f_out, "wb"))


@click.command()
@click.option(
    "-g", "--genotype", type=click.Path(), required=True, help="The genotype input file"
)
@click.option(
    "-p",
    "--phenotype",
    type=click.Path(),
    required=True,
    help="The phenotype input file",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(),
    required=True,
    help="The output file name",
)
def naivebayes(genotype, phenotype, out):
    """
    使用朴素贝叶斯进行分类
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report

    f_genotype = Path(genotype).absolute()
    f_phenotype = Path(phenotype).absolute()
    f_out = Path(out).absolute()
    d_out = f_out.parent
    d_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Read the genotype file: {f_genotype}")
    df_genotype = pd.read_csv(f_genotype, index_col=0)

    logger.info(f"Read the phenotype file: {f_phenotype}")
    df_phenotype = pd.read_csv(f_phenotype, index_col=0)

    logger.info("Split the dataset into training and testing sets")
    df_genotype_train, df_genotype_test, df_phenotype_train, df_phenotype_test = (
        train_test_split(df_genotype, df_phenotype,
                         test_size=0.2, random_state=42)
    )

    logger.info("Naive Bayes classification")
    model = GaussianNB()
    model.fit(df_genotype_train, df_phenotype_train.values.ravel())

    logger.info("Evaluate the model")
    predictions = model.predict(df_genotype_test)
    report = classification_report(df_phenotype_test, predictions)
    logger.info(f"Classification Report:\n{report}")

    logger.info("Performing cross-validation")
    cv_scores = cross_val_score(
        model, df_genotype, df_phenotype.values.ravel(), cv=5
    )  # 5折交叉验证
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean cross-validation score: {cv_scores.mean()}")

    logger.info(f"保存模型到文件:{f_out}")
    pickle.dump(model, open(f_out, "wb"))


@click.command()
@click.option(
    "-g", "--genotype", type=click.Path(), required=True, help="The genotype input file"
)
@click.option(
    "-p",
    "--phenotype",
    type=click.Path(),
    required=True,
    help="The phenotype input file",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(),
    required=True,
    help="The output file name",
)
def decisiontree(genotype, phenotype, out):
    """
    使用决策树进行分类
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report

    f_genotype = Path(genotype).absolute()
    f_phenotype = Path(phenotype).absolute()
    f_out = Path(out).absolute()
    d_out = f_out.parent
    d_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Read the genotype file: {f_genotype}")
    df_genotype = pd.read_csv(f_genotype, index_col=0)

    logger.info(f"Read the phenotype file: {f_phenotype}")
    df_phenotype = pd.read_csv(f_phenotype, index_col=0)

    logger.info("Split the dataset into training and testing sets")
    df_genotype_train, df_genotype_test, df_phenotype_train, df_phenotype_test = (
        train_test_split(df_genotype, df_phenotype,
                         test_size=0.2, random_state=42)
    )

    logger.info("Decision Tree classification")
    model = DecisionTreeClassifier(random_state=42)  # 可以根据需要调整参数
    model.fit(df_genotype_train, df_phenotype_train.values.ravel())

    logger.info("Evaluate the model")
    predictions = model.predict(df_genotype_test)
    report = classification_report(df_phenotype_test, predictions)
    logger.info(f"Classification Report:\n{report}")

    logger.info("Performing cross-validation")
    cv_scores = cross_val_score(
        model, df_genotype, df_phenotype.values.ravel(), cv=5
    )  # 5折交叉验证
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean cross-validation score: {cv_scores.mean()}")

    logger.info(f"保存模型到文件:{f_out}")
    pickle.dump(model, open(f_out, "wb"))


@click.command()
@click.option(
    "-g", "--genotype", type=click.Path(), required=True, help="The genotype input file"
)
@click.option(
    "-p",
    "--phenotype",
    type=click.Path(),
    required=True,
    help="The phenotype input file",
)
@click.option(
    "-m", "--model", type=click.Path(), required=True, help="The trained model file"
)
@click.option(
    "-pfx", "--prefix", type=str, required=True, help="Prefix for output files"
)
def forcast(genotype, phenotype, model, prefix):
    """
    使用训练好的模型进行预测并绘制ROC曲线
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, classification_report, roc_auc_score
    from sklearn.preprocessing import label_binarize

    import pickle
    import pandas as pd

    f_genotype = Path(genotype).absolute()
    f_phenotype = Path(phenotype).absolute()
    f_model = Path(model).absolute()

    logger.info(f"Read the genotype file: {f_genotype}")
    df_genotype = pd.read_csv(f_genotype, index_col=0, sep="\t")

    logger.info(f"Read the phenotype file: {f_phenotype}")
    df_phenotype = pd.read_csv(f_phenotype, index_col=0, sep="\t")

    logger.info(f"Load the trained model from: {f_model}")
    with open(f_model, "rb") as f:
        model = pickle.load(f)

    logger.info("Make predictions")
    # predictions_proba = model.predict_proba(df_genotype)  # 获取所有类别的概率
    predictions = model.predict(df_genotype)

    # 输出评估结果到文件
    report_file = f"{prefix}_classification_report.txt"
    with open(report_file, "w") as f:
        report = classification_report(df_phenotype, predictions)
        f.write(report)
    logger.info(f"Classification report saved to: {report_file}")

    # # 假设 df_phenotype 是类别标签
    # y_bin = label_binarize(df_phenotype, classes=[0, 1])  # 根据实际类别调整
    # #  计算多分类的ROC AUC
    # roc_auc = roc_auc_score(y_bin, predictions_proba, multi_class="ovr")

    # # 绘制 ROC 曲线并保存
    # plt.figure()
    # plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    # plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="lower right")
    # plt.grid()

    # roc_file = f"{prefix}_roc_curve.png"
    # plt.savefig(roc_file)
    # plt.close()
    # logger.info(f"ROC curve saved to: {roc_file}")

    logger.info("Predictions and ROC curve plotted successfully.")


cli.add_command(vcfconvert)
cli.add_command(lasso)
cli.add_command(randomforest)
cli.add_command(xgboost)
cli.add_command(svm)
cli.add_command(nn)
cli.add_command(naivebayes)
cli.add_command(decisiontree)
cli.add_command(forcast)


if __name__ == "__main__":
    cli()
