import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts import utils
from sklearn.model_selection import train_test_split

def load(clip=False):
    fallalld = pd.read_pickle(r'data/FallAllD.pkl')
    fallalld_waist = fallalld[fallalld['Device']=='Waist']
    fallalld_waist = fallalld_waist.reset_index().drop(columns=['index'])
    ADL_ids = [i for i in range(13,43)]
    fall_ids = [i for i in range(101,136)]
    adls_dict = {id: 0 for id in ADL_ids}
    falls_dict = {id: 1 for id in fall_ids}
    activity_dict = {**adls_dict, **falls_dict}
    fallalld_waist['target'] = fallalld_waist['ActivityID'].replace(activity_dict)
    # drop rows with activities outside the chosen ones
    fallalld_waist.drop(fallalld_waist[fallalld_waist['target']>1].index, inplace=True)
    fallalld_waist.drop(columns=['Gyr', 'Mag', 'Bar', 'TrialNo', 'Device'], inplace=True)
    fallalld_waist['accel_g'] = fallalld_waist['Acc'].apply(
         g_from_LSB).apply(utils.magnitude).apply(reshape_arr)
    if clip:
         fallalld_waist['accel_g'] = fallalld_waist['accel_g'].apply(clip_arr)
    fallalld_waist.drop(columns=['Acc'], inplace=True)
    return fallalld_waist

def clip_arr(arr):
    return np.clip(arr, -2,2)

def g_from_LSB(arr, sensitivity=0.244):
    # Acceleration (g) = Raw data (LSB) * Sensitivity (mg/LSB) / 1000 (mg/g)
	return (arr * sensitivity)/1000

def reshape_arr(arr):
	return np.reshape(arr, (1,-1))

def get_X_y(df, winsize=7, clip=True):
    freq = 238
    X = np.zeros([df.shape[0], winsize*freq])
    # start 1 sec before the fall
    start = int(df['accel_g'][0].size/2) - freq
    end = start + (freq * winsize)
    for i, row in enumerate(df['accel_g']):
        if clip:
             row = np.clip(row, 0, 8)
        X[i] = row[:, start:end]
    y = np.array(df['target'], dtype='uint8')
    return X, y