import numpy as np
import os
import scipy.io as sio
import pandas as pd
from matplotlib import pyplot as plt
import sys


def prepare_plot(save_plot, df, timeseries_index, label):
    if save_plot:
        plt.figure(figsize=(12,8))
        plt.plot(df[['mBP']], c='r', linewidth=1, label='mBP')
        plt.plot(df[['HR']], c='b', linewidth=1, label='HR')
        plt.ylim(20, 180)
        plt.grid()
        plt.legend(loc='upper right')
        plt.title(str(timeseries_index) + " " + label)
        plt.savefig(str(timeseries_index) + "_" + label)
        plt.close()


def combine_mat_arrays(split):
    combined = None
    for element in split[0][0]:
        combined = element[0][0] if combined is None else np.concatenate([combined, element[0][0]])
    return combined


def process_matlab_file(matfile, save_plot, timeseries_index, label):
    hr = combine_mat_arrays(matfile['BeatToBeat']['HR'])
    bp = combine_mat_arrays(matfile['BeatToBeat']['mBP'])
    mapping = {'HR': hr, 'mBP': bp}
    df = pd.DataFrame(data=mapping)

    # interpolating data to avoid NaN values
    df = df.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

    # outliers removal
    repeat_times = 5
    for i in range(repeat_times):
        df_copy = df.copy()
        df_copy = (df - df.mean())/df.std()
        for column_name in df_copy:
            column = df_copy[column_name]
            outliers = column.rolling(window=31, center=True).median().fillna(method='bfill').fillna(method='ffill')
            diff = np.abs(column - outliers)
            outlier_ids = diff > 2 / (i+1)
            df[column_name][outlier_ids] = np.NaN
        df = df.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

    prepare_plot(save_plot, df, timeseries_index, label)
    return df


def convert_all(input_folders, save_plots):
    dataset_BP = pd.DataFrame()
    dataset_HR = pd.DataFrame()

    for input_folder in input_folders:
        for filename in os.listdir(input_folder):
            matfile = sio.loadmat(input_folder+filename)
            timeseries_index = int(filename[1:-4])
            label = 'Synkope' if 'Synkope' in input_folder else 'Nosynkope'
            df = process_matlab_file(matfile, save_plots, timeseries_index, label)
            if not df.isnull().values.any():
                print("Processing" + input_folder + filename + " Shape:" + str(df.shape))
                df_HR = pd.DataFrame(columns=[filename[1:-4]], data=[label] + df['HR'].values.tolist())
                df_BP = pd.DataFrame(columns=[filename[1:-4]], data=[label] + df['mBP'].values.tolist())
                dataset_BP = pd.concat([dataset_BP, df_BP], axis=1)
                dataset_HR = pd.concat([dataset_HR, df_HR], axis=1)
            else:
                print(input_folder+filename+" was rejected!")
    dataset_BP.to_csv('BP.csv', index=False)
    dataset_HR.to_csv('HR.csv', index=False)


convert_all(['./Mat/Synkope/', './Mat/No finding/'], True)
