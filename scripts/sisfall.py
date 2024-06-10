import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import resample
from scripts.utils import magnitude, visualize_falls_adls
from sklearn.model_selection import train_test_split

def load():
    # sisfall = pd.read_pickle(r'data/SisFall.pkl')
    sisfall = pd.read_pickle(r'data/SisFall_MMA8451Q.pkl').reset_index().drop(columns=['index'])
    sisfall.drop(columns=['TrialNo'], inplace=True)
    sisfall = sisfall[sisfall['Duration (s)'] > 12]
    return sisfall


def get_g(arr):
    # Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
    range = 8
    res = 14 # specified by authors
    return ((2*range)/2**res)*np.array(arr)

def get_candidate_windows(ts, freq, target, thresh=1.4, step=1,
                          prefall=1, fall=1, postfall=5):
    ts = ts.flatten()
    total_duration = prefall + fall + postfall
    sample_window_size = int(freq*total_duration)
    required_length = int(freq*(total_duration))
    candidate_windows = []
    y = []
    if target==1:
        start = freq*2
        end = start + (freq*step) + 1
    else:
        start = 0
        end = len(ts)
    for j in range(start, end, freq*step):
        candidate_window = ts[j:j+sample_window_size]
        if len(candidate_window) < required_length:
            break
        main_window = candidate_window[freq:2*freq]
        if len(main_window) == freq*step:
            if max(main_window) >= thresh:
                candidate_windows.append(candidate_window)
                y.append(target)
    return candidate_windows, y

def get_X_y(df, freq=200):
    X = []
    y = []
    for i, row in df.iterrows():
        cw, targets = get_candidate_windows(
            row['Accel'], freq=freq,target=row['Target'])
        X.extend(cw)
        y.extend(targets)
    X = np.array(X)
    y = np.array(y, dtype='uint8')
    return X, y

def train_test_subjects_split(test_size=0.3, random_state=0, visualize=True):
    df = load()
    df['Accel'] = df['Acc'].apply(get_g).apply(magnitude)
    df.drop(columns=['Acc'], inplace=True)
    subjects = df['SubjectID'].unique()
    print(f'{len(subjects)} subjects')
    train_set, test_set = train_test_split(subjects, test_size=test_size, random_state=random_state)
    test_df = df[df['SubjectID']==test_set[0]]
    df.drop(df[df['SubjectID']==test_set[0]].index, inplace=True)
    for id in test_set[1:]:
        this_df = df[df['SubjectID']==id]
        test_df = pd.concat([test_df, this_df], ignore_index=True)
        df.drop(this_df.index, inplace=True)
        df.reset_index().drop(columns=['index'], inplace=True)
    X_train, y_train = get_X_y(df)
    X_test, y_test = get_X_y(test_df)
    print(f"Train set: X: {X_train.shape}, y: {y_train.shape}\
    ([ADLs, Falls])", np.bincount(y_train))
    print(f"Test set: X: {X_test.shape}, y: {y_test.shape}\
    ([ADLs, Falls])", np.bincount(y_test))
    if visualize:
        visualize_falls_adls(X_train, y_train)
        visualize_falls_adls(X_test, y_test, set="test")
    return X_train, y_train, X_test, y_test