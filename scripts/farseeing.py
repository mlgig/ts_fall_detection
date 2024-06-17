import os
import mat73
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.model_selection import train_test_split

def load(clip=False):
    farseeing = pd.read_pickle(r'data/farseeing.pkl').reset_index().drop(columns=['index'])
    return farseeing

def sample_adls(X_train, y_train, adl_samples):
    # Group falls and ADLs together to sample from ADLs
    X_train_y_train = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1)
    X_train_falls = X_train_y_train[X_train_y_train[:,-1]==1]
    X_train_ADLs = X_train_y_train[X_train_y_train[:,-1]==0]
    np.random.seed(5)
    rng = np.random.default_rng()
    rng.shuffle(X_train_ADLs)
    # Select <adl_samples> ADLs
    X_train_ADLs_with_labels = X_train_ADLs[:adl_samples]
    X_train_rejoined = np.concatenate([X_train_ADLs_with_labels,
                                    X_train_falls], axis=0)
    rng.shuffle(X_train_rejoined) # shuffle again
    # recover X_train and y_train
    X_train = X_train_rejoined[:,:-1]
    y_train = X_train_rejoined[:,-1].astype(int)
    return X_train, y_train

def expand_for_ts(X_train, X_test):
    X_train = np.array(X_train)[:, np.newaxis, :]
    X_test = np.array(X_test)[:, np.newaxis, :]
    return X_train, X_test

def get_X_y(df, prefall=1, fall=1, postfall=5):
    X = []
    y = []
    for i, row in df.iterrows():
        fall_point = row['fall_point']
        freq = row['freq']
        accel = row['accel_mag']
        # Take out the fall signal
        before = int(fall_point-(freq*prefall))
        after = int(fall_point+(freq*(fall+postfall)))
        fall_signal = accel[before:after]
        if freq==20:
            # resample to 100 Hz
            new_length = int(100*len(fall_signal)/20)
            fall_signal = resample(fall_signal, new_length)
        X.append(fall_signal)
        y.append(1)
        prefall_signal = accel[:before]
        # Segment prefall signal
        cw, targets = get_candidate_windows(
            prefall_signal, freq=freq, target=0, prefall=prefall,
            fall=fall, postfall=postfall, thresh=1.4)
        # print('adl', len(cw), len(targets))
        X.extend(cw)
        y.extend(targets)
    X = np.array(X)
    y = np.array(y, dtype='uint8')
    return X, y

def get_candidate_windows(ts, freq, target, prefall,
                fall, postfall, thresh=1.08, step=1):
    ts = np.array(ts).flatten()
    X = []
    y = []
    total_duration = prefall + fall + postfall
    sample_window_size = int(freq*total_duration)
    required_length = int(freq*(total_duration))
    freq_100_length = int(100*(total_duration))
    # resample to match signals of 100Hz if necessary
    resample_to_100Hz = freq_100_length != required_length
    for j in range(0, len(ts), freq*step):
        potential_window = ts[j:j+sample_window_size]
        if len(potential_window) < required_length:
            break
        main_window = potential_window[freq:2*freq]
        if len(main_window) == freq*step:
            if max(main_window) >= thresh:
                selected_window = potential_window
                if resample_to_100Hz:
                    selected_window = resample(selected_window, freq_100_length)
                X.append(selected_window)
                y.append(target)
    return X, y

# uncomment this function to load farseeing into a dataframe
# def load_in_df():
#     cols = ['SubjectID', 'FallID', 'freq', 'accel', 'accel_mag', 'fall_point']
#     df = pd.DataFrame(columns=cols)
#     signal_files, _, meta = load_signal_files()
#     df_dict = {}
#     for sf in tqdm(signal_files):
#         if sf == "F_00002186-05-2013-11-23-18-25-04.mat":
#             continue
#         fall_id = '-'.join(sf.split("_")[1].split("-")[:2])
#         df_dict['FallID'] = fall_id
#         row = meta[meta['Randomnumber']==fall_id]
#         if row['Sensor_location'].item() != 'L5':
#             continue
#         df_dict['SubjectID'] = fall_id.split("-")[0]
#         df_dict['freq'] = row['Sample_rate_Hz'].item()
#         signal = mat73.loadmat(f'data/FARSEEING/signals/{sf}')
#         accel = signal['tmp'][:,2:5]/9.8
#         df_dict['accel'] = [list(accel)]
#         accel_magnitude = magnitude(np.clip(accel, -2, 2))
#         df_dict['accel_mag'] = [list(accel_magnitude)]
#         fall_indicator = signal['tmp'][:,11]
#         df_dict['fall_point'] = np.where(fall_indicator!=0)[0][0]
#         this_df = pd.DataFrame(data=df_dict)
#         df = pd.concat([df, this_df], ignore_index=True)
#     return df