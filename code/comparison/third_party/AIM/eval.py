import numpy as np
from baselines import AdaFairClassifier
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, roc_auc_score, f1_score
from aif360.sklearn.metrics import generalized_entropy_error

from fairens import FairAugEnsemble
from fairlearn.postprocessing import ThresholdOptimizer

METRICS = {
    "all": ["acc", "bacc", "dp", "eo", "si", "ge", "cs"],
    "utility": ["acc", "bacc", "ap", "roc", "f1"],
    "fairness": ["dp", "eo", "si", "ge", "cs"],
}


def flip_s_in_X(X, s):

    def locate_s_in_X(X, s):
        s_idx = []
        for i in range(X.shape[1]):
            if np.all(X[:, i] == s):
                s_idx.append(i)
        if len(s_idx) == 1:
            return s_idx[0]
        elif len(s_idx) < 1:
            raise ValueError("Could not locate sensitive feature in X.")
        elif len(s_idx) > 1:
            raise ValueError(f"Multiple {len(s_idx)} sensitive features found in X.")

    # assert np.all(X[:, 0] == s)
    X_ = X.copy()
    # i = locate_s_in_X(X_, s)
    i = 0
    X_[:, i] = 1 - X_[:, i]
    return X_


def spouse_inconsistency(clf, X, s, random_state=None):
    X_flip = flip_s_in_X(X, s)
    y_pred = get_y_pred(clf, X, s, random_state=random_state)
    y_pred_flip = get_y_pred(clf, X_flip, s, random_state=random_state)
    score = np.mean(y_pred != y_pred_flip)
    return score


def get_y_pred(model, X, s, random_state=None):
    try:
        if isinstance(model, AdaFairClassifier):
            y_pred = model.predict(X)
        elif isinstance(model, FairAugEnsemble):
            y_pred = model.predict(X, sensitive_features=s)
        elif isinstance(model, ThresholdOptimizer):
            y_pred = model.predict(X, sensitive_features=s, random_state=random_state)
        else:
            try:
                y_pred = model.predict(
                    X, sensitive_features=s, random_state=random_state
                )
            except:
                try:
                    y_pred = model.predict(X, sensitive_features=s)
                except:
                    y_pred = model.predict(X)
    except Exception as e:
        raise RuntimeError(f"Failed to predict using the provided model: {e}")
    return y_pred


def evaluate(model, X, y, s, pos_label=1, random_state=None, in_percentage=False):
    """
    Evaluate the fairness and performance metrics of a model on the provided dataset.

    Parameters:
        model: The trained predictive model to be evaluated.
        X (numpy array or pandas DataFrame): Input features for evaluation.
        y (numpy array or pandas Series): True labels for evaluation.
        s (numpy array or pandas Series): Sensitive attribute values for evaluation.
        in_percentage (bool, optional): Whether to return the metrics in percentage. Defaults to False.

    Returns:
        dict: A dictionary containing accuracy, demographic parity difference (dp), equalized odds difference (eo),
              accuracy for each group (acc_grp), and positive rate for each group (pos_rate_grp).
    """
    y_pred = get_y_pred(model, X, s, random_state=random_state)
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    ap = average_precision_score(y, y_pred)
    roc = roc_auc_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    dp = demographic_parity_difference(y, y_pred, sensitive_features=s)
    eo = equalized_odds_difference(y, y_pred, sensitive_features=s)
    ge = generalized_entropy_error(y, y_pred, alpha=2, pos_label=pos_label)
    try:
        si = spouse_inconsistency(model, X, s, random_state=random_state)
    except Exception as e:
        raise e
        si = None
    acc_grp, pos_rate_grp = {}, {}
    g_adv, max_pos_rate = None, 0
    for su in np.unique(s):
        su = int(su)
        mask = s == su
        acc_grp[su] = accuracy_score(y[mask], y_pred[mask]).round(3)
        pos_rate_grp[su] = np.mean(y_pred[mask]).round(3)
        if pos_rate_grp[su] > max_pos_rate:
            g_adv, max_pos_rate = su, pos_rate_grp[su]
    acc_cls = {}
    for yu in np.unique(y):
        yu = int(yu)
        mask = y == yu
        acc_cls[yu] = accuracy_score(y[mask], y_pred[mask]).round(3)
    if in_percentage:
        acc *= 100
        bacc *= 100
        dp *= 100
        eo *= 100
    return {
        "acc": acc,
        "bacc": bacc,
        "ap": ap,
        "roc": roc,
        "f1": f1,
        "dp": dp,
        "eo": eo,
        "ge": ge,
        "si": si,
        "acc_grp": acc_grp,
        "pos_rate_grp": pos_rate_grp,
        "g_adv": g_adv,
        "acc_cls": acc_cls,
    }


def evaluate_multi_split(model, data_dict, random_state=None):
    """
    Evaluate a model on multiple datasets.

    Parameters:
        model: The trained predictive model to be evaluated.
        data_dict (dict): A dictionary containing dataset names as keys and tuple of (X, y, s) as values.

    Returns:
        dict: A dictionary containing evaluation results for each dataset in data_dict.
    """
    results = {}
    for data_name, (X, y, s) in data_dict.items():
        results[data_name] = evaluate(model, X, y, s, random_state=random_state)
    return results


def verbose_print(result_dict):
    """
    Print the evaluation results in a formatted and verbose manner.

    Parameters:
        result_dict (dict): A dictionary containing evaluation results for different datasets.
    """
    info = ""
    max_len = max([len(k) for k in result_dict.keys()])
    for data_name, result in result_dict.items():
        info = f"{data_name:<{max_len}s}"
        for metric in METRICS["all"]:
            info += f" | {metric}={result[metric]:.3f}"
        info += (
            f" | acc_grp={result['acc_grp']} | pos_rate_grp={result['pos_rate_grp']}"
        )
        print(info)
