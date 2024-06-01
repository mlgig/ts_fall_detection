import os
import mat73
# import utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import resample
from scripts import utils
from sklearn.model_selection import train_test_split

def load():
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
    return fallalld_waist


def g_from_LSB(arr, sensitivity=0.244):
    # Acceleration (g) = Raw data (LSB) * Sensitivity (mg/LSB) / 1000 (mg/g)
	return (arr * sensitivity)/1000

def reshape_arr(arr):
	return np.reshape(arr, (1,-1))

def get_X_y(df):
    X = np.zeros([df.shape[0], df['Accel'][0].size])
    for i, row in enumerate(df['Accel']):
        X[i] = row
    y = np.array(df['target'], dtype='uint8')
    return X, y

def subject_train_test_split(test=0.3, random_state=0):
    df = load()
    df['Accel'] = df['Acc'].apply(g_from_LSB).apply(
        utils.magnitude).apply(reshape_arr)
    df.drop(columns=['Acc'], inplace=True)
    subjects = df['SubjectID'].unique()
    train_set, test_set = train_test_split(subjects, random_state=random_state)
    test_df = df[df['SubjectID']==test_set[0]]
    df.drop(df[df['SubjectID']==test_set[0]].index, inplace=True)
    for id in test_set[1:]:
        this_df = df[df['SubjectID']==id]
        test_df = pd.concat([test_df, this_df], ignore_index=True)
        df.drop(this_df.index, inplace=True)
        df.reset_index().drop(columns=['index'], inplace=True)
    X_train, y_train = get_X_y(df)
    X_test, y_test = get_X_y(test_df)
    return X_train, y_train, X_test, y_test