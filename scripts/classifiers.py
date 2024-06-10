import os, time, timeit, json, mat73, argparse
import numpy as np, pandas as pd
from scipy.signal import resample
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt, seaborn as sns
sns.set_style("ticks")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scripts import farseeing

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
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay

from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.classification.deep_learning import FCNClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import DrCIFClassifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.convolution_based import RocketClassifier, HydraClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier

from scripts.TsCaptum.explainers import Shapley_Value_Sampling as SHAP

def explain_model(model, X, y, chunks):
    shap = SHAP(model)
    exp = shap.explain(X, labels=y, n_segments=chunks)
    return exp

def run_tabular_models(X_train, y_train, X_test, y_test, freq, s=27.5):
    window_size = int(s * freq)
    tabular_models = {
        'LogisticCV': LogisticRegressionCV(cv=5, n_jobs=-1, solver='newton-cg'),
        'RandomForest': RandomForestClassifier(n_estimators=150, random_state=0),
        'KNN': KNeighborsClassifier(),
        'RidgeCV': RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
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

    return metrics_df, tabular_models

def run_ts_models(X_train, y_train, X_test, y_test, freq, s=27.5):
    window_size = int(s * freq)
    fast = {
        # 'Hydra': HydraClassifier(random_state=0, n_jobs=-1),
        # 'Rocket': RocketClassifier(random_state=0, n_jobs=-1),
        # 'MultiRocketHydra': MultiRocketHydraClassifier(random_state=0, n_jobs=-1),
        # 'Catch22': Catch22Classifier(random_state=0, n_jobs=-1),
        'QUANT': QUANTClassifier(random_state=0),
    }
    slow = {
        'FCN': FCNClassifier(n_epochs=100, random_state=0),
    }
    very_slow = {
        'DrCIF': DrCIFClassifier(random_state=0, n_jobs=-1),
        'InceptionTime': InceptionTimeClassifier(n_epochs=100, random_state=0),
    }

    if s==27.5:
        ts_models = {**fast}
    elif s==7:
        ts_models = {**fast, **slow}
    else:
        ts_models = {**fast, **slow, **very_slow}

    metrics_df = pd.DataFrame(columns=['model', 'window_size', 'runtime', 'precision', 'recall', 'f1-score'])
    for model in ts_models.items():
        metrics = predict_eval(model, window_size, X_in=(X_train, X_test),
                               y_in=(y_train, y_test))
        these_metrics = pd.DataFrame(data=metrics)
        metrics_df = pd.concat([metrics_df, these_metrics], ignore_index=True)

    return metrics_df, ts_models

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

def to_labels(pos_probs, threshold):
    # apply threshold to positive probabilities to create labels
    return (pos_probs >= threshold).astype('int')

def predict_eval(model, win_size, X_in=None, y_in=None, starttime=None, adapt_threshold=False, verbose=1):
    target_names = ['ADL', 'Fall']
    model_name, clf = model
    clf = make_pipeline(
        StandardScaler(),
        SimpleImputer(missing_values=np.nan, strategy='mean'),
        clf
    )
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