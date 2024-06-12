import timeit
from tqdm import tqdm
import numpy as np, pandas as pd
from scipy.signal import resample
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt, seaborn as sns
sns.set_style("ticks")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scripts import farseeing, fallalld, sisfall

from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
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

def get_models(type=None, models_subset=None):
    all_models = {
        'tabular': {
            'LogisticCV': LogisticRegressionCV(cv=5, n_jobs=-1, solver='newton-cg'),
            'RandomForest': RandomForestClassifier(n_estimators=150, random_state=0),
            'KNN': KNeighborsClassifier(),
            'RidgeCV': RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=150, max_features=0.1,
                                            criterion="entropy", n_jobs=-1,
                                            random_state =0)
        },
        'ts': {
            'Hydra': HydraClassifier(random_state=0, n_jobs=-1),
            'Rocket': RocketClassifier(random_state=0, n_jobs=-1),
            'MultiRocketHydra': MultiRocketHydraClassifier(random_state=0, n_jobs=-1),
            'Catch22': Catch22Classifier(random_state=0, n_jobs=-1),
            'QUANT': QUANTClassifier(random_state=0),
            # 'FCN': FCNClassifier(n_epochs=100, random_state=0)
        }
    }

    if type is None: # run all models
        models = {**all_models['tabular'], **all_models['ts']}
    else: # the saner choice
        models = all_models[type]
    if models_subset is not None: # select model subset
        models = {m: models[m] for m in models_subset}
    
    return models


def run_models(X_train, y_train, X_test, y_test, freq, s=7, type=None, subset=None):
    models = get_models(type=type, models_subset=subset)
    metrics_df = pd.DataFrame(columns=['model', 'window_size', 'runtime', 'precision', 'recall', 'f1-score'])
    for model in models.items():
        metrics = predict_eval(model, s, freq, X_in=(X_train, X_test),
                               y_in=(y_train, y_test))
        these_metrics = pd.DataFrame(data=metrics)
        metrics_df = pd.concat([metrics_df, these_metrics], ignore_index=True)

    return metrics_df, models

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

def predict_eval(model, s, freq, X_in=None, y_in=None, starttime=None, adapt_threshold=False, verbose=1):
    win_size = int(s*freq)
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
    # print(f'(w={s})', end=': ')
    print(f'{model_name}', end=' ')
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
    runtime = timeit.default_timer() - starttime
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f'({np.round(runtime,2)}s)', end='. ')
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
                 'window_size': s,
                 'runtime':[np.round(runtime, 2)],
                 'precision':[np.round(precision, 2)],
                 'recall':[np.round(recall, 2)],
                 'f1-score':[np.round(f1, 2)]}
                )

def chunk_list(l, n):
    n_per_set = len(l)//n
    for i in range(1, n_per_set*n, n_per_set):
        chunk = l[i:i+n_per_set]
        if len(chunk) < n_per_set:
            break
        yield chunk

def get_freq(dataset):
    if dataset==farseeing:
        return 100
    elif dataset==fallalld:
        return 238
    else:
        return 200

def get_dataset_name(dataset):
    names = {
        farseeing: 'farseeing',
        fallalld: 'fallalld',
        sisfall: 'sisfall'
    }
    return names[dataset]

def cross_validate(dataset, model_type=None, models_subset=None,
                   s=7, cv=5, df=None, verbose=True):
    dataset_name = get_dataset_name(dataset)
    if df is None:
        df = dataset.load()
    subjects = list(df['SubjectID'].unique())
    # divide subjects into cv sets
    test_sets = list(chunk_list(subjects, cv))
    freq = get_freq(dataset)
    metrics_df = None
    for i, test_set in enumerate(test_sets):
        test_df = df[df['SubjectID']==test_set[0]]
        train_df = df.drop(df[df['SubjectID']==test_set[0]].index)
        for id in test_set[1:]:
            this_df = df[df['SubjectID']==id]
            test_df = pd.concat([test_df, this_df], ignore_index=True)
            train_df.drop(this_df.index, inplace=True)
            train_df.reset_index().drop(columns=['index'], inplace=True)
        X_train, y_train = dataset.get_X_y(train_df)
        X_test, y_test = dataset.get_X_y(test_df)
        if verbose:
            print(f'\n\n-- fold {i+1} ({len(test_set)} subjects) --')
            print(f"Train set: X: {X_train.shape}, y: {y_train.shape}\
            ([ADLs, Falls])", np.bincount(y_train))
            print(f"Test set: X: {X_test.shape}, y: {y_test.shape}\
            ([ADLs, Falls])", np.bincount(y_test))
        if metrics_df is None:
            metrics_df, _ = run_models(X_train, y_train, X_test, y_test, type=model_type,
                                       freq=freq, s=s, subset=models_subset)
            metrics_df['fold'] = i
        else:
            this_df, _ = run_models(X_train, y_train, X_test, y_test, type=model_type,
                                    freq=freq, s=s, subset=models_subset)
            this_df['fold'] = i
            metrics_df = pd.concat([metrics_df, this_df], ignore_index=True)
    mean_df = metrics_df.groupby(['model']).mean().round(2)
    std_df = metrics_df.groupby(['model']).std().round(2)
    cols = ['model', 'window_size', 'runtime', 'precision', 'recall', 'f1-score']
    aggr = {c: [] for c in cols}
    for i in mean_df.index:
        aggr['model'].append(i)
        for col in cols[1:]:
            aggr[col].append(f'{mean_df.loc[i][col]} $\pm$ {std_df.loc[i][col]}')
    aggr_df = pd.DataFrame(data=aggr)
    aggr_df.to_csv(f'results/{dataset_name}_{model_type}_{s}.csv')
    return metrics_df, aggr_df

def boxplot(df, dataset, model_type, metric='f1-score', save=False, **kwargs):
    plt.figure(figsize=(5, 3), dpi=400)
    sns.boxplot(data=df.sort_values(by='model'),
                x='model', y='f1-score',
                width=0.3, **kwargs)
    plt.grid(axis='y')
    plt.xlabel('')
    sns.despine()
    plt.title(f'{model_type.capitalize()} models CV {metric}s on {dataset.upper()}')
    plt.xticks(rotation=15)
    if save:
        plt.savefig(f'figs/{dataset}_{model_type}_boxplot.eps',
                    format='eps', bbox_inches='tight')
    plt.show()