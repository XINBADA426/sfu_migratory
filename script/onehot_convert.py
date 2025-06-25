#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: CB Ren
# @Created Date:   2025/6/21 20:15
from pathlib import Path
import logging
import click

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__file__)
logger.addHandler(logging.NullHandler())

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-g', '--genotype',
              required=True,
              type=click.Path(),
              help="The genotype file input")
@click.option('-o', '--out',
              required=True,
              type=click.Path(),
              help="The output file after onehot convert")
def main(genotype, out):
    """
    将输入数据进行独热编码
    """
    f_genotype = Path(genotype).absolute()
    f_out = Path(out).absolute()

    logger.info(f"Read the genotype file: {f_genotype}")
    df_genotype = pd.read_csv(f_genotype, index_col=0, sep="\t")

    sample_names = df_genotype.index

    logger.info(f"Convert the features to onehot")
    logger.info(f"Original shape: {df_genotype.shape}")

    encoded_dfs = []
    feature_names = []

    for col in df_genotype.columns:
        unique_values = df_genotype[col].unique()
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        col_data = df_genotype[col].values.reshape(-1, 1)
        encoded = encoder.fit_transform(col_data).astype(int)

        categories = encoder.categories_[0]
        new_col_names = [f"{col}_{cat}" for cat in categories]
        feature_names.extend(new_col_names)

        encoded_dfs.append(pd.DataFrame(encoded, columns=new_col_names))

    df_encoded = pd.concat(encoded_dfs, axis=1)

    df_encoded.index = sample_names

    logger.info(f"OneHot encoded shape: {df_encoded.shape}")
    logger.info(f"Number of features after encoding: {len(feature_names)}")

    logger.info(f"Save the onehot result to {f_out}")
    df_encoded.to_csv(f_out, sep="\t")


if __name__ == "__main__":
    main()
