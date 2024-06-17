from cProfile import label
import copy
import timeit
from turtle import color
from matplotlib import axis
from tqdm.notebook import tqdm
import numpy as np, pandas as pd
from scipy.signal import resample
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt, seaborn as sns
import matplotlib.colors as mcolors
sns.set_style("ticks")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scripts import farseeing, fallalld, sisfall, utils
from scripts.utils import get_freq

from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker

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
from sklearn.preprocessing import MinMaxScaler
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

def explain_model(model, X, y, chunks, normalise=True):
    X = X[:, np.newaxis, :]
    if X.shape[0]==1:
        print("Found 1 sample. Doubling it to avoid errors")
        X = np.vstack([X, X])
        y = np.vstack([y, y]).flatten()
    shap = SHAP(model)
    exp = shap.explain(X, labels=y, n_segments=chunks, normalise=normalise)
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
        }
    }

    if type is None: # run all models
        models = {**all_models['tabular'], **all_models['ts']}
    else: # the saner choice :-)
        models = all_models[type]
    if models_subset is not None: # select model subset
        models = {m: models[m] for m in models_subset}
    
    return models


def run_models(X_train, y_train, X_test, y_test, freq, s=7, type=None, subset=None, verbose=1, cm_grid=(1,5)):
    trained_models = {}
    models = get_models(type=type, models_subset=subset)
    metrics_df = pd.DataFrame(columns=['model', 'window_size', 'runtime', 'precision', 'recall', 'f1-score'])
    if verbose > 2:
        fig, axs = plt.subplots(*cm_grid, figsize=(10,3), sharey=True)
        fig.supxlabel('Predicted label')
    for m, model in tqdm(enumerate(models.items())):
        metrics, trained_model, cm = predict_eval(
            model, s, freq, X_in=(X_train, X_test),
            y_in=(y_train, y_test), verbose=verbose)
        if verbose > 2:
            ax = axs.flat[m]
            plot_cm(cm, ax=ax, model_name=model[0])
            ax.set_title(model[0])
            ax.set_xlabel('')
            if m>0:
                ax.set_ylabel('')
        these_metrics = pd.DataFrame(data=metrics)
        metrics_df = pd.concat([metrics_df, these_metrics], ignore_index=True)
        trained_models[model[0]]=trained_model
    plt.savefig('figs/confmats.eps', format='eps', bbox_inches='tight')
    plt.show()

    return metrics_df, trained_models

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
    # clf = make_pipeline(
    #     StandardScaler(),
    #     SimpleImputer(missing_values=np.nan, strategy='mean'),
    #     clf
    # )
    has_proba = hasattr(clf, 'predict_proba')
    if X_in is not None:
        X_train, X_test = X_in
        y_train, y_test = y_in
    X_train = X_train[:, :win_size]
    X_test = X_test[:, :win_size]
    clf.fit(X_train, y_train)
    if starttime is None:
        starttime = timeit.default_timer()
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
    runtime = (timeit.default_timer() - starttime)/X_test.shape[0]
    runtime = np.round(runtime * 1000) # milliseconds (ms)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=1)
    if verbose > 1:
        if has_proba:
            print(f'{model_name} AUC: {np.round(roc_auc_score(y_test, probs), 2)}')
        else:
            print("Skipping AUC since chosen classifier has no predict_proba() method")
            print(classification_report(y_test, y_pred, target_names=target_names))
    cm = confusion_matrix(y_test, y_pred) if verbose > 2 else None
    return dict({'model': [model_name],
                 'window_size': s,
                 'runtime':[np.round(runtime, 2)],
                 'precision':[np.round(precision, 2)],
                 'recall':[np.round(recall, 2)],
                 'f1-score':[np.round(f1, 2)]}
                ), clf, cm

def plot_cm(cm, model_name, ax, fontsize=20, colorbar=False):
    target_names = ['ADL', 'Fall']
    plt.rcParams.update({'font.size': fontsize})
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=colorbar)
    # plt.grid(False)
    plt.rcParams.update({'font.size': 10})


def chunk_list(l, n):
    n_per_set = len(l)//n
    for i in range(1, n_per_set*n, n_per_set):
        chunk = l[i:i+n_per_set]
        if len(chunk) < n_per_set:
            break
        yield chunk

def get_dataset_name(dataset):
    names = {
        farseeing: 'FARSEEING',
        fallalld: 'FallAllD',
        sisfall: 'SisFall'
    }
    return names[dataset]

def cross_validate(dataset, model_type=None, models_subset=None,
                   s=7, cv=5, df=None, verbose=True, random_state=0):
    dataset_name = get_dataset_name(dataset)
    if df is None:
        df = dataset.load()
    rng = np.random.default_rng(random_state)
    subjects = list(df['SubjectID'].unique())
    rng.shuffle(subjects)
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

def cross_dataset_eval(d1, d2):
    # resample to the lower frequency
    new_freq = min(get_freq(d1), get_freq(d2))
    X_d1, y_d1 = utils.train_test_subjects_split(d1, clip=d1!=farseeing, new_freq=new_freq, split=False)
    
    X_d2, y_d2 = utils.train_test_subjects_split(d2, clip=d2!=farseeing, new_freq=new_freq, split=False)

    print(f'<----- {get_dataset_name(d1)} > {get_dataset_name(d2)} ----->')
    d1d2_df, _ = run_models(X_d1, y_d1, X_d2, y_d2, freq=new_freq, type='ts')
    d1d2_df['scenario'] = f'{get_dataset_name(d1)}>{get_dataset_name(d2)}'
    print(f'<----- {get_dataset_name(d2)} > {get_dataset_name(d1)} ----->')
    d2d1_df, _ = run_models(X_d2, y_d2, X_d1, y_d1, freq=new_freq, type='ts')
    d2d1_df['scenario'] = f'{get_dataset_name(d2)}>{get_dataset_name(d1)}'

    # Mix all and train_test_split
    X = np.concatenate([X_d1, X_d2], axis=0)
    y = np.concatenate([y_d1, y_d2], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)

    print(f'<----- {get_dataset_name(d1)} + {get_dataset_name(d2)} ----->')
    mixed_df, _ = run_models(X_train, y_train, X_test, y_test, freq=new_freq, type='ts')
    mixed_df['scenario'] = f'{get_dataset_name(d1)}+{get_dataset_name(d2)}'

    return pd.concat([d1d2_df, d2d1_df, mixed_df], ignore_index=True)


def get_sample_attributions(clf, X_test, y_test, c=28, normalise=True, n=2):
    y_pred = clf.predict(X_test)
    true_falls = np.logical_and(y_test==1, y_pred==1)
    false_falls = np.logical_and(y_test==0, y_pred==1)
    true_adls = np.logical_and(y_test==0, y_pred==0)
    false_adls = np.logical_and(y_test==1, y_pred==0)
    tp_exp = explain_model(clf, X_test[true_falls][:n],
                           y_test[true_falls][:n], chunks=c,
                           normalise=normalise)
    fp_exp = explain_model(clf, X_test[false_falls][:n],
                           y_test[false_falls][:n], chunks=c,
                           normalise=normalise)
    tn_exp = explain_model(clf, X_test[true_adls][:n],
                            y_test[true_adls][:n], chunks=c,
                            normalise=normalise)
    fn_exp = explain_model(clf, X_test[false_adls][:n],
                            y_test[false_adls][:n], chunks=c,
                            normalise=normalise)
    tp = {'sample':X_test[true_falls][0], 'attr': tp_exp[0]}
    fp = {'sample':X_test[false_falls][0], 'attr': fp_exp[0]}
    tn = {'sample':X_test[true_adls][0], 'attr': tn_exp[0]}
    fn = {'sample':X_test[false_adls][0], 'attr': fn_exp[0]}

    return [tp, fp, tn, fn]

def scale_arr(arr):
    scaler = MinMaxScaler(feature_range=(-1,1))
    return scaler.fit_transform(arr.reshape(-1,1)).flatten()

def plot_sample_with_attributions(attr_dict):
    titles = ['True Falls', 'False Alarms', 'True ADLs', 'Misses']
    fig, axs = plt.subplots(5, 4, figsize=(10, 10), dpi=400,
                            sharey='row', sharex='col', layout='constrained')
    plt.rcParams.update({'font.size': 10})
    cmap = plt.get_cmap('coolwarm')
    attributions = copy.deepcopy(attr_dict)
    for i, (model_name, exps) in enumerate(attributions.items()):
        axs[i,0].set_ylabel(model_name)
        for e, exp in enumerate(exps):
            ax = axs[i,e]
            if i==0:
                ax.set_title(titles[e])
            y = scale_arr(exp['sample'])
            x = np.arange(len(y))
            c = exp['attr'].flatten()
            ax.plot(c, linestyle=':', label='attribution profile', alpha=0.3)
            # Normalize the color values
            norm = mcolors.Normalize(vmin=-1, vmax=1)
            for j in range(len(x)-1):
                ax.plot(x[j:j+2], y[j:j+2], color=cmap(norm(c[j])), linewidth=1.5, label='normalised sample' if j==0 else None)
            ticks_loc = ax.get_xticks().tolist()
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax.set_xticklabels([i//100 for i in ticks_loc])
            # ax.grid(which='both', axis='x')     
    axs[1,3].legend()
    fig.supylabel('Attribution score')
    # Adding color bar to show the color scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = plt.axes((1.01, 0.05, 0.015, 0.92))
    plt.colorbar(sm, cax=cax)
    fig.supxlabel('Time in seconds')
    plt.savefig('figs/model_explanation.eps', format='eps', bbox_inches='tight')
    plt.show()