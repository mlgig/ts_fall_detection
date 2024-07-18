# import pywt
from cProfile import label
from curses import window
import numpy as np
import pandas as pd
from math import isnan, sqrt
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.signal import resample
from sklearn.metrics import f1_score
import time, timeit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.preprocessing import scale
from torch import layout
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

def colorlist2(c1, c2, num):
    l = np.linspace(0, 1, num)
    a = np.abs(np.array(c1) - np.array(c2))
    m = np.min([c1, c2], axis=0)
    s = np.sign(np.array(c2) - np.array(c1)).astype(int)
    s[s == 0] = 1
    r = np.sqrt(np.c_[(l * a[0] + m[0])[::s[0]],
                      (l * a[1] + m[1])[::s[1]], (l * a[2] + m[2])[::s[2]]])
    return r

def color_plot(x, y):
    ynorm = (y - y.min()) / (y.max() - y.min())
    cmap = LinearSegmentedColormap.from_list(
        "", colorlist2((1, 0, 0), (0, 0, 1), 100))
    colors = [cmap(k) for k in ynorm[:-1]]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
    lc = LineCollection(segments, colors=colors, linewidth=2)
    lc.set_array(x)
    return lc

def plot_all_samples(X, ax, freq=100):
    X = np.squeeze(X) if X.ndim > 2 else X
    # X = np.nan_to_num(X)
    # if X.min() != -1:
    #     X = scale(X)
    ax.plot(X.T, color='lightblue', alpha=0.5)
    # lc = color_plot(x, mean_vals)
    mean_vals = X.mean(axis=0)
    # x = np.arange(len(mean_vals))
    # tiled = np.tile(mean_vals, (400,1))
    # print(tiled.min(), tiled.max())
    # print(tiled)
    # norm = plt.Normalize(-1, 1)
    # im = ax.imshow(tiled, cmap='coolwarm', alpha=0.5, norm=norm)
    # ax.imshow(tiled, cmap='coolwarm', alpha=0.5, norm=norm)
    ax.plot(mean_vals, color='blue', label='mean attribution profile')
    # ax.add_collection(lc)
    # ax.autoscale()
    # plt.show()
    # return im

def visualize_falls_adls(X, y, dataset="train", save=True):
    fig, axs = plt.subplots(1, 2, figsize=(6, 2), dpi=200,
                        sharey=True, layout='tight')
    # remove dims with size 1
    X = np.squeeze(X)
    y = np.squeeze(y)
    fallers = y.astype(bool)
    falls = X[fallers]
    adls = X[fallers == False]
    plot_all_samples(adls, ax=axs[0])
    axs[0].set_title('ADL samples')
    axs[0].set_ylabel('Accel magnitude (g)')
    
    plot_all_samples(falls, ax=axs[1])
    axs[1].set_title('Fall samples')
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

def train_test_subjects_split(dataset, test_size=0.3, random_state=0, visualize=False, clip=False, new_freq=None, split=True, show_test=False):
    df = dataset.load(clip=clip)
    subjects = df['SubjectID'].unique()
    print(f'{len(subjects)} subjects')
    resample = new_freq is not None and new_freq!=get_freq(dataset)
    if split==False:
        X, y = dataset.get_X_y(df)
        if resample:
            X = resample_to(X, old_f=get_freq(dataset),
                            new_f=new_freq)
        return X, y
    else:
        train_set, test_set = train_test_split(subjects, test_size=test_size, random_state=random_state)
        if show_test:
            print(f'Test set -> {len(test_set)} of {len(subjects)} subjects: {test_set}.')
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
            X_train = resample_to(X_train, old_f=get_freq(dataset),
                                  new_f=new_freq)
            X_test = resample_to(X_test, old_f=get_freq(dataset),
                                 new_f=new_freq)
        print(f"Train set: X: {X_train.shape}, y: {y_train.shape}\
        ([ADLs, Falls])", np.bincount(y_train))
        print(f"Test set: X: {X_test.shape}, y: {y_test.shape}\
        ([ADLs, Falls])", np.bincount(y_test))
        if visualize:
            visualize_falls_adls(X_train, y_train)
            visualize_falls_adls(X_test, y_test, dataset="test")
        return X_train, X_test, y_train, y_test
    
def summary_visualization(dfs, model_type):
    dataset_names = ['FARSEEING', 'FallAllD', 'SisFall']
    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(1,3, figsize=(9, 4), dpi=400,
                            sharey=True, layout='tight')
    for d, df in enumerate(dfs):
        df.sort_values(by='model', inplace=True)
        sns.boxplot(data=df, x='model', y='f1-score', width=0.5, ax=axs[d], linewidth=1,
                    palette='tab10')
        axs[d].grid(axis='y')
        axs[d].set_title(dataset_names[d])
        axs[d].set_xlabel('')
        # axs[d].grid(axis='both')
        if d != 0:
            axs[d].set_ylabel('')
        plt.setp(axs[d].get_xticklabels(), rotation=45, ha='right')
    sns.despine()
    plt.savefig(f'figs/{model_type}_summary_boxplot.eps', format='eps', bbox_inches='tight')
    plt.show()

def ts_vs_tabular_summary(all_dfs):
    dataset_names = ['FARSEEING', 'FallAllD', 'SisFall']
    # add dataset names to each df
    # concatenate all results for each dataset
    farseeing_cv_df, farseeing_cv_df_ts, fallalld_cv_df, fallalld_cv_df_ts, sisfall_cv_df, sisfall_cv_df_ts = all_dfs
    farseeing_all_df = pd.concat([df.assign(
        dataset=dataset_names[0]) for df in [farseeing_cv_df.assign(type='Tabular Models'),
        farseeing_cv_df_ts.assign(type='Time Series Models')]],
        ignore_index=True)
    fallalld_all_df = pd.concat([df.assign(
        dataset=dataset_names[1]) for df in [fallalld_cv_df.assign(type='Tabular Models'),
        fallalld_cv_df_ts.assign(type='Time Series Models')]],
        ignore_index=True)
    sisfall_all_df = pd.concat([df.assign(
        dataset=dataset_names[2]) for df in [sisfall_cv_df.assign(type='Tabular Models'),
        sisfall_cv_df_ts.assign(type='Time Series Models')]],
        ignore_index=True)
    all_results_df = pd.concat([farseeing_all_df, fallalld_all_df, sisfall_all_df], ignore_index=True)
    all_results_df.to_csv('results/all_results.csv')
    all_results_df.drop(all_results_df[all_results_df['f1-score']==0].index, inplace=True)
    plt.rcParams.update({'font.size': 13})
    plt.figure(figsize=(10, 3), dpi=400)
    # plt.rcParams.update({'font.size': 16})
    sns.boxplot(data=all_results_df, x='type', y='f1-score', hue='dataset', width=0.3)
    # plt.xticks(rotation=45, ha='right')
    plt.grid()
    plt.xlabel('')
    sns.despine()
    plt.savefig('figs/ts_vs_tabular_boxplot_summary.eps', format='eps', bbox_inches='tight')
    plt.show()

def cross_dataset_summary(df):
    # df = pd.concat(dfs, ignore_index=True)
    plt.rcParams.update({'font.size': 13})
    melted = df.drop(columns=['runtime', 'window_size', 'auc', 'specificity']).melt(
        id_vars=["trainset", "model"])
    
    plt.figure(figsize=(9, 3), dpi=400)
    order=['FARSEEING', 'FallAllD', 'FallAllD+', 'SisFall','SisFall+', 'All']
    # melted.replace({'FallAllD+FARSEEING':'FallAllD+',
    #                 'SisFall+FARSEEING':'SisFall+'}, inplace=True)
    sns.boxplot(melted, x='trainset', y='value', hue='variable', width=0.5, palette="tab10", order=order)
    plt.grid(axis='both')
    plt.xlabel('Training Set', labelpad=10)
    plt.ylabel('score')
    sns.despine()
    # plt.legend(loc=9, ncols=3)
    plt.savefig('figs/cross_dataset_boxplot_summary.pdf', bbox_inches='tight')
    plt.show()

def plot_window_size_ablation(window_metrics=None):
    if window_metrics is None:
        window_metrics = pd.read_csv('results/window_size_ablation.csv')
    fig, axs = plt.subplots(2,3, figsize=(9,4), dpi=(400),
                        sharex='col', layout='tight')
    titles = [f'Test time/sample ($\mu$s)', 'AUC',
            'Precision', 'Recall', 'Specificity', f'F$_1$ score']
    for i, col in enumerate(window_metrics.columns[2:]):
        ax = axs.flat[i]
        sns.lineplot(data=window_metrics, x='window_size', y=col, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(titles[i])
    axs[0,0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.supxlabel('Total window size in seconds')
    plt.savefig('figs/window_size_ablation.pdf', bbox_inches='tight')
    plt.show()