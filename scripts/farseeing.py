import os
import mat73
# import utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import resample

def get_description(sensor_location=None):
    cols = ['Randomnumber', 'Setting', 'Sensor_location', 'Sample_rate_Hz',
        'Pre_fall_activity_reported', 'fall_direction_reported']
    meta = pd.read_excel(r'data/FARSEEING/description.xlsx', engine='openpyxl', usecols=cols)
    if sensor_location == 'L5':
        meta = meta[meta['Sensor_location']=='L5']
    if sensor_location == 'Thigh':
        meta = meta[meta['Sensor_location']=='Thigh']
  
    return meta

def load_signal_files(show_plot=False, save_plot=False):
    meta = get_description()
    signal_files = os.listdir(r'data/FARSEEING/signals')
    falls_dict = {}
    for sf in signal_files:
        p, ID = sf[2:].split('-')[:2]
        if p in falls_dict:
            falls_dict[p].append(sf)
        else:
            falls_dict[p] = [sf]

    if show_plot or save_plot:
        fig, axs = plt.subplots(2,3, figsize=(12,6), dpi=150, sharey='row')
        for sf in signal_files:
            fall_id = '-'.join(sf.split("_")[1].split("-")[:2])
            row = meta[meta['Randomnumber']==fall_id]
            freq = row['Sample_rate_Hz'].item()
            signal = mat73.loadmat(f'data/FARSEEING/signals/{sf}')
            time = signal['tmp'][:,0]
            accel = signal['tmp'][:,2:5]/9.8
            accel_norm = np.clip(accel, -2, 2)
            labels = ['x','y','z']
            for i, sig in enumerate(accel_norm.T):
                if freq == 20: # resample to make freq=100Hz
                    sig = resample(sig, 120000)
                if row['Sensor_location'].item() == 'L5':
                    axs[0,i].plot(sig)
                else:
                    axs[1,i].plot(sig)
                axs[0,i].set_title(f'Acceleration {labels[i]}')
        for ax in axs.flatten():
            ymin, ymax = ax.get_ylim()
            ax.scatter(60000, ymin, label='fall')
        fig.supylabel(r'Accel (g)')
        fig.supxlabel('Time in seconds')
        axs[0,0].set_ylabel('Sensor on L5')
        axs[1,0].set_ylabel('Sensor on Thigh')
        fig.tight_layout()
        plt.legend()
        if save_plot:
            plt.savefig('/figs/signal_files.pdf')
        if show_plot:
            plt.show()
        
    return signal_files, falls_dict, meta

def train_test_subjects_split(prefall=1, fall=1, postfall=25.5, thresh=1.4,
                              adl_samples=None, visualize=True):
    signal_files, falls_dict, meta = load_signal_files()
    subjects = list(falls_dict.keys())
    # Have 65 subjects for training, 27 for testing
    test_set = ['74827807', '74905787', '75240038', '76573505', '79232001', '79336438', '79666043', '79761947', '80061866', '87486959', '88051353', '89647122', '91923026', '91943076', '92097726', '92680167', '93169462', '93807530', '95030446', '95205003', '95253031', '96201346', '96856291', '97085274', '97097674', '97946301', '98843998']
    # train_set = list(set(subjects) - set(test_set))
    X_train = []; y_train = [];
    X_test = []; y_test = [];

    for sf in signal_files:
        if sf == "F_00002186-05-2013-11-23-18-25-04.mat":
            continue
        fall_id = '-'.join(sf.split("_")[1].split("-")[:2])
        row = meta[meta['Randomnumber']==fall_id]
        if row['Sensor_location'].item() != 'L5':
            continue
        subject = fall_id.split("-")[0]
        test = subject in test_set
        freq = row['Sample_rate_Hz'].item()
        signal = mat73.loadmat(f'data/FARSEEING/signals/{sf}')
        time = signal['tmp'][:,0]
        accel = signal['tmp'][:,2:5]/9.8
        accel_magnitude = magnitude(np.clip(accel, -2, 2))
        fall_indicator = signal['tmp'][:,11]
        fall_point = np.where(fall_indicator!=0)[0][0]
        # window_size = prefall + fall + postfall 
        # end_with_padding = len(accel_magnitude) - int(freq * window_size)
        # extract the fall <prefall> sec before, and <fall+postfall> secs after
        before = int(fall_point-(freq*prefall))
        after = int(fall_point+(freq*(fall+postfall)))
        fall_signal = accel_magnitude[before:after]
        prefall_signal = accel_magnitude[:before]
        # Segment fall_signal
        X_train, X_test, y_train, y_test = get_windows(
            X_train, X_test, y_train, y_test, fall_signal, freq, target=1,
            test=test, prefall=prefall, fall=fall, postfall=postfall)
        # Segment prefall_signal
        X_train, X_test, y_train, y_test = get_windows(
            X_train, X_test, y_train, y_test, prefall_signal, freq, target=0, thresh=thresh, test=test, prefall=prefall, fall=fall, postfall=postfall)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    if adl_samples is not None:
        X_train, y_train = sample_adls(X_train, y_train, adl_samples)
    print(f"Train set: X: {X_train.shape}, y: {y_train.shape}\
    ([ADLs, Falls])", np.bincount(y_train))
    print(f"Test set: X: {X_test.shape}, y: {y_test.shape}\
    ([ADLs, Falls])", np.bincount(y_test))
    if visualize:
        visualize_falls_adls(X_train, y_train)
        visualize_falls_adls(X_test, y_test, set="test")
    return X_train, y_train, X_test, y_test

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

def visualize_falls_adls(X, y, set="train", save=True):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=150,
                        sharey=True, layout='tight')
    fallers = y.astype(bool)
    falls = X[fallers]
    adls = X[fallers == False]
    axs[0].plot(adls.T, color='lightblue')
    axs[0].plot(adls.mean(axis=0), color='blue', label='mean sample')
    axs[0].set_title('ADL samples')
    axs[0].set_ylabel('Accel magnitude (g)')
    
    axs[1].plot(falls.T, color='lightblue')
    axs[1].plot(falls.mean(axis=0), color='blue', label='mean sample')
    axs[1].set_title('Fall samples')
    
    fig.suptitle(f"Mean ADLs and fall samples in the {set} set")
    axs[1].legend()
    if save:
        plt.savefig('figs/adls_vs_falls.pdf', bbox_inches='tight')
    plt.show()

def expand_for_ts(X_train, X_test):
    X_train = np.array(X_train)[:, np.newaxis, :]
    X_test = np.array(X_test)[:, np.newaxis, :]
    return X_train, X_test

def get_windows(X_train, X_test, y_train, y_test,
    ts, freq, target, thresh=1.08, step=1, test=False, pip=False,
    prefall=1, fall=1, postfall=25.5):
    # Main fall_window = 1 sec, prefall window = 1 sec
    # postfall window = 1 sec, recovery window = 24.5 secs
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
                if test:
                    X_test.append(selected_window)
                    y_test.append(target)
                else:
                    X_train.append(selected_window)
                    y_train.append(target)
    return X_train, X_test, y_train, y_test

def magnitude(arr):
    x, y, z = arr.T
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    magnitude -= min(magnitude)
    return magnitude

# if __name__ == '__main__':
#     X_train, y_train, X_test, y_test = train_test_subjects_split()