#! /usr/bin/env python
# coding=utf-8
# Copyright 2022 Bofeng Huang

import os
import re
import time
from pathlib import Path
from typing import Optional, Union

import fire
import jiwer
import numpy as np
import pandas as pd

# from myscripts.text.normalize_french import FrenchTextNormalizer
from myscripts.text.normalize_french_zaion import FrenchTextNormalizer
from myscripts.text.compute_wer import compute_wer

text_normalizer = FrenchTextNormalizer()

def eval_results(result_df, outdir, id_column_name, target_column_name, prediction_column_name, do_ignore_words=False):

    def norm_func(s):
        # NB
        return text_normalizer(
            s, do_lowercase=True, do_ignore_words=do_ignore_words, symbols_to_keep="'-", do_standardize_numbers=True
        )

    result_df[target_column_name] = result_df[target_column_name].map(norm_func)
    result_df[prediction_column_name] = result_df[prediction_column_name].map(norm_func)

    # filtering out empty targets
    result_df = result_df[result_df[target_column_name] != ""]

    result_df[target_column_name] = result_df[target_column_name].str.split()
    targets = result_df.set_index(id_column_name)[target_column_name].to_dict()

    result_df[prediction_column_name] = result_df[prediction_column_name].str.split()
    predictions = result_df.set_index(id_column_name)[prediction_column_name].to_dict()

    out_dir_ = f"{outdir}/wer_summary_without_fillers" if do_ignore_words else f"{outdir}/wer_summary"
    compute_wer(targets, predictions, out_dir_, do_print_top_wer=True, do_catastrophic=True)


def main(input_file_path, outdir, id_column_name="ID", target_column_name="wrd", prediction_column_name="prediction"):
    result_df = pd.read_csv(input_file_path, sep="\t")
    # result_df = pd.read_csv(input_file_path)

    # ! real single "nan" in dekuple
    result_df[target_column_name] = result_df[target_column_name].fillna("nan")

    # fill empty hyp
    result_df[prediction_column_name] = result_df[prediction_column_name].fillna("")

    eval_results(result_df, outdir, id_column_name, target_column_name, prediction_column_name, do_ignore_words=False)
    eval_results(result_df, outdir, id_column_name, target_column_name, prediction_column_name, do_ignore_words=True)


if __name__ == "__main__":
    fire.Fire(main)
