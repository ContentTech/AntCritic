#! /usr/bin/env python
# -*- coding:utf-8 -*-
import csv
import argparse
import h5py
import pandas as pd
from tqdm import tqdm


def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-f', '--file', type=str, help='input file')
    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))
    return args


def idx2set(idx):
    if idx < 8000:
        cnt = "train"
    elif 8000 <= idx < 9000:
        cnt = "test"
    else:
        cnt = "val"
    return cnt


def resplit(csv_paths):
    dfs = []
    for csv_path in csv_paths:
        df_tmp = pd.read_csv(csv_path)
        dfs.append(df_tmp)
    df = pd.concat(dfs)
    df.reset_index()
    res = {'train': df.head(0), 'val': df.head(0), 'test': df.head(0)}
    for row_idx in tqdm(range(len(df))):
        row = df.iloc[row_idx]
        tags = eval(row['tags'])
        if len(tags) > 400:
            print(f"Too Long Passage for Row {row_idx}")
            continue
        res[idx2set(row_idx)].loc[res[idx2set(row_idx)].shape[0]] = row
    res['train'].to_csv("/mnt/fengyao.hjj/transformers/data/pgc/0506/train.1.csv", index=False)
    res['val'].to_csv("/mnt/fengyao.hjj/transformers/data/pgc/0506/dev.1.csv", index=False)
    res['test'].to_csv("/mnt/fengyao.hjj/transformers/data/pgc/0506/test.1.csv", index=False)


if __name__=="__main__":
    args = cmd()
    csv_paths = ["/mnt/fengyao.hjj/transformers/data/pgc/0506/train.collect.csv",
     "/mnt/fengyao.hjj/transformers/data/pgc/0506/dev.collect.csv",
     "/mnt/fengyao.hjj/transformers/data/pgc/0506/test.collect.csv"]
    resplit(csv_paths)