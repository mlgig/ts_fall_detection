import os, time, timeit, json, mat73, argparse
import numpy as np, pandas as pd
from scipy.signal import resample
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt, seaborn as sns
sns.set_style("ticks")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scripts import farseeing
# from torch import nn

from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics import ConfusionMatrixDisplay

# from scripts.hydra import Hydra, SparseScaler

from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.deep_learning import FCNClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import DrCIFClassifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.convolution_based import RocketClassifier, HydraClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier
# from aeon.transformations.collection.shapelet_based import RandomShapeletTransform

# from sktime.transformations.panel.rocket import Rocket, MiniRocket, MultiRocket
# from sktime.transformations.panel.catch22 import Catch22


def run_tabular_models(X_train, y_train, X_test, y_test, window_size=2750):

    tabular_models = {
        'Logistic': LogisticRegression(random_state=0, n_jobs=-1, solver='newton-cg'),
        'RandomForest': RandomForestClassifier(n_estimators=150, random_state=0),
        'KNN': KNeighborsClassifier(),
        'Ridge': RidgeClassifier(random_state=0),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=150, max_features=0.1,
                                           criterion="entropy", n_jobs=-1,
                                           random_state =0)
    }

    metrics_df = pd.DataFrame(columns=['model', 'window_size', 'runtime', 'precision', 'recall', 'f1-score'])
    for model in tabular_models.items():
        metrics = predict_eval(model, window_size, X_in=(X_train, X_test),
                               y_in=(y_train, y_test))
        these_metrics = pd.DataFrame(data=metrics)
        metrics_df = pd.concat([metrics_df, these_metrics], ignore_index=True)
    
    return metrics_df

def run_ts_models(X_train, y_train, X_test, y_test, window_size=2750):

    ts_models = {
        'Hydra': HydraClassifier(random_state=0, n_jobs=-1),
        'Rocket': RocketClassifier(random_state=0, n_jobs=-1),
        'MultiRocketHydra': MultiRocketHydraClassifier(random_state=0, n_jobs=-1),
        # 'InceptionTime': InceptionTimeClassifier(n_epochs=100, random_state=0),
        'FCN': FCNClassifier(n_epochs=100, random_state=0),
        #'TimeSeriesKNN': KNeighborsTimeSeriesClassifier(n_jobs=-1),
        'Catch22': Catch22Classifier(random_state=0, n_jobs=-1),
        'QUANT': QUANTClassifier(random_state=0),
        'DrCIF': DrCIFClassifier(random_state=0, n_jobs=-1),
    }

    metrics_df = pd.DataFrame(columns=['model', 'window_size', 'runtime', 'precision', 'recall', 'f1-score'])
    for model in ts_models.items():
        metrics = predict_eval(model, window_size, X_in=(X_train, X_test),
                               y_in=(y_train, y_test))
        these_metrics = pd.DataFrame(data=metrics)
        metrics_df = pd.concat([metrics_df, these_metrics], ignore_index=True)

    return metrics_df

def plot_metrics(df, x='model', pivot='f1-score', compare='metrics', **kwargs):
    default_kwargs = {'figsize': (6,2), 'rot': 0}
    kwargs = {**default_kwargs, **kwargs}
    if compare=='metrics':
        w = max(df['window_size'])
        window_df = df[df['window_size']==w].drop(columns=['window_size', 'runtime'])
        window_df.plot(kind='bar', x='model', **kwargs)
    elif compare=='window_size':
        crosstab = df.pivot_table(pivot, ['model'], 'window_size')
        crosstab.plot(kind='bar', rot=0, **kwargs)
        plt.grid()
        plt.xlabel('')
        plt.ylabel('')
        sns.despine()
    else:
        df.plot(kind='bar', x='model', y=compare)
    plt.legend(loc=9, ncol=3, bbox_to_anchor=(0.5,1.3), title=compare)
# def run_hydra(X_train, y_train, X_test, y_test, final_clf, **kwargs):
#     X_train, X_test = farseeing.expand_for_ts(X_train, X_test)
#     X_train_torch = torch.tensor(X_train, dtype=torch.float32)
#     X_test_torch = torch.tensor(X_test, dtype=torch.float32)

#     start = timeit.default_timer()
#     hydra_transform = Hydra(X_train_torch.shape[-1], seed=0)
#     X_training_transform = hydra_transform(X_train_torch)
#     X_test_transform = hydra_transform(X_test_torch)
#     scaler = SparseScaler()
#     X_training_transform = scaler.fit_transform(X_training_transform)
#     X_test_transform = scaler.transform(X_test_transform)
#     metrics = predict_eval(final_clf, kwargs['window_size'],
#                            X_in=(X_training_transform, X_test_transform),
#                            y_in=(y_train, y_test), starttime=start)
#     return metrics

# def add_final_clf(transform):
#     clfs = {
#         'RidgeCV': RidgeClassifierCV(alphas = np.logspace(-3, 3, 10)),
#         'ExtraTrees': ExtraTreesClassifier(
#             n_estimators=150, max_features=0.1, criterion="entropy", n_jobs=-1, random_state=0)
#     }
#     models = []
#     for clf_name, clf in clfs.items():
#         models.append((f'{transform}+{clf_name}', clf))
    
#     return models

# def run_hydra(X_train, y_train, X_test, y_test, window_size):
#     X_train, X_test = farseeing.expand_for_ts(X_train, X_test)
#     X_train_torch = torch.tensor(X_train, dtype=torch.float32)
#     X_test_torch = torch.tensor(X_test, dtype=torch.float32)
#     hydra_models = add_final_clf('Hydra')
    
#     metrics_df = pd.DataFrame(columns=['model', 'window_size', 'runtime', 'precision', 'recall', 'f1-score'])
#     for hydra_model in hydra_models:
#         start = timeit.default_timer()
#         hydra_transform = Hydra(X_train_torch.shape[-1], seed=40)
#         X_training_transform = hydra_transform(X_train_torch)
#         X_test_transform = hydra_transform(X_test_torch)

#         scaler = SparseScaler()
#         X_training_transform = scaler.fit_transform(X_training_transform)
#         X_test_transform = scaler.transform(X_test_transform)
#         metrics = predict_eval(hydra_model, window_size,
#                            X_in=(X_training_transform, X_test_transform),
#                            y_in=(y_train, y_test), starttime=start)
#         this_df = pd.DataFrame(data=metrics)
#         metrics_df = pd.concat([metrics_df, this_df])    
#     return metrics_df            

# def run_rocket(X_train, y_train, X_test, y_test, window_size):
#     rocket_models = add_final_clf('Rocket')
#     for rocket_model in rocket_models:
#         start = timeit.default_timer()
#         rocket = Rocket(random_state=0).fit(X_train)
#         X_train_transform = rocket.transform(X_train)
#         X_test_transform = rocket.transform(X_test)
#         metrics = predict_eval(rocket_model, window_size,
#                            X_in=(X_train_transform, X_test_transform),
#                            y_in=(y_train, y_test), starttime=start)
#         this_df = pd.DataFrame(data=metrics)
#         metrics_df = pd.concat([metrics_df, this_df])    
#     return metrics_df

# def run_ts_models(X_train, y_train, X_test, y_test, window_size=2750):

#     # Hydra
#     metrics_df = run_hydra(X_train, y_train, X_test, y_test, window_size)
#     # Rocket
#     rocket_metrics = run_rocket(X_train, y_train, X_test, y_test, window_size)
#     metrics_df = pd.concat([metrics_df, rocket_metrics], ignore_index=True)

#     return metrics_df
    


def to_labels(pos_probs, threshold):
    # apply threshold to positive probabilities to create labels
    return (pos_probs >= threshold).astype('int')

def predict_eval(model, win_size, X_in=None, y_in=None, starttime=None, adapt_threshold=False, verbose=1):
    target_names = ['ADL', 'Fall']
    model_name, clf = model
    has_proba = hasattr(clf, 'predict_proba')
    if X_in is not None:
        X_train, X_test = X_in
        y_train, y_test = y_in
    X_train = X_train[:, :win_size]
    X_test = X_test[:, :win_size]
    print("> {:<17} (win_size={})".format(model_name, win_size), end='\t')
    if starttime is None:
        starttime = timeit.default_timer()
    if verbose:
        print("Training", end='')
    clf.fit(X_train, y_train)
    if verbose:
        print("/Testing", end=' ')
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
    runtime = timeit.default_timer() - starttime
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    if verbose:
        print(f"in {np.round(runtime,2)} secs.")
    if verbose > 1:
        if has_proba:
            print(f'AUC: {np.round(roc_auc_score(y_test, probs), 2)}')
        else:
            print("Skipping AUC since chosen classifier has no predict_proba() method")
            print(classification_report(y_test, y_pred, target_names=target_names))
    if verbose > 2:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.grid(False)
        plt.show()
    return dict({'model': [model_name],
                 'window_size': win_size,
                 'runtime':[runtime],
                 'precision':[precision],
                 'recall':[recall],
                 'f1-score':[f1]}
                )