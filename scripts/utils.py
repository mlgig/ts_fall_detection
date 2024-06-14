# import pywt
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.metrics import f1_score
import time, timeit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.preprocessing import LabelEncoder
from torch.utils import data
from scripts import farseeing, fallalld, sisfall


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def predict_eval(model, X_in=None, y_in=None, starttime=None, adapt_threshold=False):
    target_names = ['ADL', 'Fall']
    model_name, clf = model
    has_proba = hasattr(clf, 'predict_proba')
    if X_in is not None:
        X_train, X_test = X_in
        y_train, y_test = y_in
    print("classifier:", model_name)
    if starttime is None:
        starttime = timeit.default_timer()
    clf.fit(X_train, y_train)
    if has_proba:
        probs = clf.predict_proba(X_test)[:, 1]
        train_probs = clf.predict_proba(X_train)[:, 1]
    else:
        if adapt_threshold:
            adapt_threshold = False
            print("Setting adapt_threshold=False since chosen classifier has no predict_proba() method")
    if adapt_threshold:
        thresholds = np.arange(0, 1, 0.001)
        # evaluate each threshold
        scores = [f1_score(y_train, to_labels(train_probs, t)) for t in thresholds]
        # get best threshold
        ix = np.argmax(scores)
        print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
        y_pred = to_labels(probs, thresholds[ix])
    else:
        y_pred = clf.predict(X_test)
    print("Time to train + test (sec):", timeit.default_timer() - starttime)
    if has_proba:
        print(f'AUC: {np.round(roc_auc_score(y_test, probs), 2)}')
    else:
        print("Skipping AUC since chosen classifier has no predict_proba() method")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.grid(False)
    plt.show()
    print(classification_report(y_test, y_pred, target_names=target_names))

def point_line_distance(line, p):
    """Calculates the distance between a point and a line defined by two points.

    Args:
    line => p1, p2
        p1: First point on the line (tuple or list).
        p2: Second point on the line (tuple or list).
    p: The point to find the distance from (tuple or list).

    Returns:
    The distance between the point and the line.
    """
    p1, p2 = line
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p.T
    # Distance from (p3) to the line defined by (p1) and (p2)
    numerator = abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1)
    denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return numerator / denominator

def get_pips(y, k=None, visualize=True, **kwargs):
    # set k to half the length of y by default
    if k is None:
        k = max(3, int(len(y)/2))
    assert k > 2, "k must be greater than 2"
    default_kwargs = {
        'xlabel': '', 'ylabel': '',
        'figsize': (10,3),
        'dpi': 150,
        'title': 'Perceptually Important Points'
    }
    kwargs = {**default_kwargs, **kwargs}
    x = np.arange(0, len(y)).reshape(-1,1)
    y_arr = np.array(y).reshape(-1,1)
    points = np.concatenate([x, y_arr], axis=1)
    line = (0, len(y)-1) # initial line
    pips = find_pips(points, line, k)
    pips.sort()
    if visualize:
        visualize_pips(y, pips, **kwargs)
    return y[pips]

def visualize_pips(y, pips, **kwargs):
    plt.figure(figsize=kwargs['figsize'], dpi=kwargs['dpi'])
    plt.plot(y, label='Time Series')
    plt.plot(pips, y[pips], 'x', label='PIPs')
    plt.xlabel(kwargs['xlabel'])
    plt.ylabel(kwargs['ylabel'])
    plt.title(kwargs['title'])
    plt.legend()
    plt.grid(True)
    plt.show()

def find_pips(points, line, k, pips_list=None):
    if pips_list is None: 
        pips_list = [0, len(points)-1] # initial pips
    # select split point
    window = points[line[0]:line[1]]
    if len(pips_list) <= k and len(window) > 2:
        line_coords = (points[line[0]], points[line[1]])
        distances_arr = point_line_distance(line_coords, window)
        new_pip = np.argmax(distances_arr)
        pips_list.append(new_pip)
        # make new lines
        left_line = (line[0], new_pip)
        right_line = (new_pip, line[1])
        left_is_longer = len(left_line) >= len(right_line)
        new_line = left_line if left_is_longer else right_line
        return find_pips(points, new_line, k, pips_list)
    else:
        return pips_list

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
    end = len(ts) - int(freq * total_duration)
    count = 0
    for j in range(0, len(ts), freq*step):
        # potential_window = ts[j-int(freq*prefall):j+int(postfall*freq)]
        potential_window = ts[j:j+sample_window_size]
        if len(potential_window) < required_length:
            break
        main_window = potential_window[freq:2*freq]
        if len(main_window) == freq*step:
            if max(main_window) >= thresh:
                selected_window = potential_window
                count+=1
                # if len(selected_window) < required_length:
                    # excluded.append((selected_window, "selected_window", "not long enough"))
                    # continue
                if resample_to_100Hz:
                    selected_window = resample(selected_window, freq_100_length)
                if pip:
                    selected_window = get_pips(selected_window,
                        k=pip, visualize=False)
                    selected_window = resample(selected_window, pip)
                if test:
                    X_test.append(selected_window)
                    y_test.append(target)
                else:
                    X_train.append(selected_window)
                    y_train.append(target)
                # n_windows += 1
            # else:
                # excluded.append((main_window, "main_window", "max < 1.4"))
        # else:
            # excluded.append((main_window, "main_window", "wrong length"))
    print(f'target: {target}, count: {count}')
    return X_train, X_test, y_train, y_test

def magnitude(arr):
    x, y, z = arr.T.astype('float')
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    magnitude -= min(magnitude)
    return magnitude

def visualize_samples(X_train, y_train, X_test, y_test, dataset):
    X = np.vstack(X_train, X_test)
    y = np.vstack(y_train, y_test)
    visualize_falls_adls(X, y, dataset=dataset)


def visualize_falls_adls(X, y, dataset="train", save=True):
    fig, axs = plt.subplots(1, 2, figsize=(6, 2), dpi=200,
                        sharey=True, layout='tight')
    # remove dims with size 1
    X = np.squeeze(X)
    y = np.squeeze(y)
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
    fig.suptitle(f"Mean ADLs and fall samples in the {dataset} dataset")
    axs[1].legend()
    if save:
        plt.savefig(f'figs/{dataset}_mean_samples.eps', format='eps',
                    bbox_inches='tight')
    plt.show()

def resample_to(arr, old_f, new_f=100):
    new_list = []
    old_len = arr.shape[-1]
    for sample in arr:
        resampled = resample(sample, int(new_f*(old_len/old_f)))
        new_list.append(resampled)
    new_arr = np.array(new_list)
    return new_arr

def get_freq(dataset):
    if dataset==farseeing:
        return 100
    elif dataset==fallalld:
        return 238
    else:
        return 200

def train_test_subjects_split(dataset, test_size=0.3, random_state=0, visualize=False, clip=False, new_freq=None, split=True):
    df = dataset.load(clip=clip)
    subjects = df['SubjectID'].unique()
    # print(f'{len(subjects)} subjects')
    if split==False:
        X, y = dataset.get_X_y(df)
        if new_freq is not None and new_freq!=get_freq(dataset):
            X = resample_to(X, old_f=get_freq(dataset),
                            new_f=new_freq)
        return X, y
    else:
        train_set, test_set = train_test_split(subjects, test_size=test_size, random_state=random_state)
        test_df = df[df['SubjectID']==test_set[0]]
        df.drop(df[df['SubjectID']==test_set[0]].index, inplace=True)
        for id in test_set[1:]:
            this_df = df[df['SubjectID']==id]
            test_df = pd.concat([test_df, this_df], ignore_index=True)
            df.drop(this_df.index, inplace=True)
            df.reset_index().drop(columns=['index'], inplace=True)
        X_train, y_train = dataset.get_X_y(df)
        X_test, y_test = dataset.get_X_y(test_df)
        if resample:
            X_train = resample_to(X, old_f=get_freq(dataset),
                                  new_f=new_freq)
            X_test = resample_to(X, old_f=get_freq(dataset),
                                 new_f=new_freq)
        print(f"Train set: X: {X_train.shape}, y: {y_train.shape}\
        ([ADLs, Falls])", np.bincount(y_train))
        print(f"Test set: X: {X_test.shape}, y: {y_test.shape}\
        ([ADLs, Falls])", np.bincount(y_test))
        if visualize:
            visualize_falls_adls(X_train, y_train)
            visualize_falls_adls(X_test, y_test, dataset="test")
        return X_train, X_test, y_train, y_test