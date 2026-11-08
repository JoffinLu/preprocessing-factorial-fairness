from collections import Counter

import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import clone, is_classifier
from sklearn.ensemble._base import BaseEnsemble
from tqdm import tqdm
from utils import idx_to_mask, mask_to_idx, seed_generator


def random_seeds(master_seed, num_seeds):
    # Create random state object with master seed
    rng = np.random.RandomState(master_seed)
    # Generate random seeds and yield them one at a time
    for i in range(num_seeds):
        yield rng.randint(0, 100000)


class FairAugEnsemble(BaseEnsemble):
    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=10,
        how=None,
        verbose=False,
        bootstrap=False,
        random_state=None,
    ):
        assert is_classifier(estimator), "Base estimator must be a classifier"
        self.estimator = estimator
        self.base_estimator = None
        self.estimator_params = {}
        self.n_estimators = n_estimators
        self._validate_estimator()
        self.flag_proba = hasattr(self.estimator_, 'predict_proba')
        self.how = how
        self.verbose = verbose
        self.bootstrap = bootstrap
        # Master random seed
        self.random_state = random_state
        # Seeds for each estimator
        self.seed_generator = seed_generator(random_state)
        self.random_seeds = [
            next(self.seed_generator) for _ in range(self.n_estimators)
        ]

    def _init_group_stats(self, X, y, s):
        y_count, s_count = Counter(y), Counter(s)
        y_unique, s_unique = list(y_count.keys()), list(s_count.keys())
        assert (
            len(y_count) == len(s_count) == 2
        ), "We only handle binary classification with binary group."
        assert (
            set(y_count.keys()) == set(s_count.keys()) == set([0, 1])
        ), "label/attribute value should be in \{0, 1\}."
        stats = {}
        for su in s_unique:
            su_mask = s == su
            y_su = y[su_mask]
            stats[su] = {
                'size': s_count[su],
                'n_pos': (y_su == 1).sum(),
                'n_neg': (y_su == 0).sum(),
                'pos_ratio': (y_su == 1).sum() / s_count[su],
                'neg_ratio': (y_su == 0).sum() / s_count[su],
                'idx': np.where(su_mask)[0],
                'idx_pos': np.where((y == 1) & su_mask)[0],
                'idx_neg': np.where((y == 0) & su_mask)[0],
            }
        self.grp_stats = stats
        self.meta_info = {
            'overall_pos_ratio': (y == 1).sum() / len(y),
            'g_adv': max(stats, key=lambda x: stats[x]['pos_ratio']),
            'g_dis': min(stats, key=lambda x: stats[x]['pos_ratio']),
            'g_maj': max(stats, key=lambda x: stats[x]['size']),
            'g_min': min(stats, key=lambda x: stats[x]['size']),
        }

    def bootstrap_sample(self, X, y, s, random_state):
        """Sample the data with replacement within each group."""
        grp = self.grp_stats
        meta = self.meta_info
        # set the random seed
        np.random.seed(random_state)
        new_idx = np.zeros_like(y, dtype=int)
        pos_mask, neg_mask = y == 1, y == 0
        for g in grp.keys():
            g_mask = s == g
            idx_pos = np.where(pos_mask & g_mask)[0]
            new_idx[idx_pos] = np.random.choice(
                idx_pos, size=len(idx_pos), replace=True
            )
            idx_neg = np.where(neg_mask & g_mask)[0]
            new_idx[idx_neg] = np.random.choice(
                idx_neg, size=len(idx_neg), replace=True
            )
        X_boot, y_boot, s_boot = X[new_idx], y[new_idx], s[new_idx]
        return X_boot, y_boot, s_boot

    def fit(self, X, y, sensitive_features: np.ndarray):
        s = sensitive_features
        n_samples, n_features = X.shape
        self._init_group_stats(X, y, s)

        # Ensemble training
        self.estimators_ = []
        for i in range(self.n_estimators):
            X_, y_, s_ = X.copy(), y.copy(), s.copy()
            # Data augmentation
            if i == 0:  # no data aug for the first estimator
                X_, y_, s_ = X_, y_, s_
            else:
                X_, y_, s_ = self._augment_data(
                    X_, y_, s_, random_state=self.random_seeds[i]
                )
            # Bootstrap sampling
            if self.bootstrap:
                X_, y_, s_ = self.bootstrap_sample(
                    X_, y_, s_, random_state=self.random_seeds[i]
                )
            # Train the estimator
            estimator = (
                self._make_estimator()
            )  # this will add estimator into estimators_
            try:
                estimator.set_params(
                    random_state=self.random_seeds[i],
                )
            except:
                pass
            estimator.fit(X_, y_)
        return self

    def _augment_data(self, X, y, s, random_state):
        """Augment/sample the data according to the augmentation mode."""
        how = self.how
        grp = self.grp_stats
        meta = self.meta_info
        if isinstance(how, dict):
            mode = how['mode']
        else:
            raise TypeError("`augmentation` must be a dict.")
        # set the random seed
        np.random.seed(random_state)

        # no augmentation
        if mode is None or mode == 'dummy':
            X_aug, y_aug, s_aug = X, y, s
        # subsample optimistic augmentation
        elif mode == 'sub_optim':
            g_adv = meta['g_adv']
            max_pos_ratio = grp[g_adv]['pos_ratio']
            sample_idx = []
            for g in grp.keys():
                if g == g_adv:
                    sample_idx.append(grp[g]['idx'])
                else:  # subsample the neg samples in disadvantaged group
                    n_neg_exp = int(
                        grp[g]['n_pos'] * (1 - max_pos_ratio) / max_pos_ratio
                    )
                    idx_sub_neg = np.random.choice(
                        grp[g]['idx_neg'], size=n_neg_exp, replace=False
                    )
                    sample_idx.append(idx_sub_neg)
                    sample_idx.append(grp[g]['idx_pos'])
            sample_idx = np.concatenate(sample_idx)
            X_aug, y_aug, s_aug = X[sample_idx], y[sample_idx], s[sample_idx]
        # not implemented
        else:
            raise NotImplementedError

        # describe_data({
        #     'Augmented data': (X_aug, y_aug, s_aug),
        # })

        return X_aug, y_aug, s_aug

    def predict(self, X, sensitive_features=None):
        y_pred_proba = self.predict_proba(X)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred

    def predict_proba(self, X, sensitive_features=None):
        if self.flag_proba:
            y_pred_proba = [
                estimator.predict_proba(X) for estimator in self.estimators_
            ]
        else:
            y_pred_proba = [estimator.predict(X) for estimator in self.estimators_]
        y_pred_proba = np.mean(np.array(y_pred_proba), axis=0)
        return y_pred_proba


from baselines.reduction import ReductionClassifier
from baselines.reweighting import ReweightClassifier
from baselines.threshold import ThresholdClassifier


class FairEnsemble(BaseEnsemble):
    default_how = {
        'pred': 'calib',
        'consensus': 'disag',  # 'disag', 'other-disag',
        'uniformity': 0.1,
        'drop_ratio': 1.0,
        'pos_ratio': 'max',
        'bootstrap': False,
        'action': 'drop',
    }
    default_fair_how = {
        'method': None,
        'constraint': None,
    }

    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=10,
        fair_how=None,
        how=None,
        verbose=False,
        bootstrap=False,
        random_state=None,
    ):
        assert is_classifier(estimator), "Base estimator must be a classifier"
        self.fair_how = self.default_fair_how.copy()
        if fair_how is not None:
            self.fair_how.update(fair_how)
        # validate estimator
        fair_how = self.fair_how
        self.init_estimator = estimator
        if fair_how['method'] is None:
            self.estimator = estimator
        else:
            if fair_how['method'] == 'Reweight':
                self.estimator = ReweightClassifier(
                    estimator=estimator,
                )
            elif fair_how['method'] == 'Reduction':
                if fair_how['constraint'] == 'DP':
                    self.estimator = ReductionClassifier(
                        estimator=estimator,
                        constraints='DemographicParity',
                    )
                elif fair_how['constraint'] == 'EO':
                    self.estimator = ReductionClassifier(
                        estimator=estimator,
                        constraints='EqualizedOdds',
                    )
                else:
                    raise NotImplementedError(
                        f"Constraint {fair_how['constraint']} not implemented."
                    )
            elif fair_how['method'] == 'Threshold':
                if fair_how['constraint'] == 'DP':
                    self.estimator = ThresholdClassifier(
                        estimator=estimator,
                        constraints='demographic_parity',
                    )
                elif fair_how['constraint'] == 'EO':
                    self.estimator = ThresholdClassifier(
                        estimator=estimator,
                        constraints='equalized_odds',
                    )
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        self.base_estimator = None
        self.estimator_params = {}
        self._validate_estimator()
        self.flag_proba = hasattr(self.estimator_, 'predict_proba')
        # store parameters
        self.n_estimators = n_estimators
        self.how = self.default_how.copy()
        self.how.update(how)
        self.verbose = verbose
        self.bootstrap = bootstrap
        # Master random seed
        self.random_state = random_state
        # Seeds for each estimator
        self.seed_generator = seed_generator(random_state)
        self.random_seeds = [
            next(self.seed_generator) for _ in range(self.n_estimators)
        ]

    def _init_stats(self, X, y, s):
        y_count, g_count = Counter(y), Counter(s)
        groups = list(g_count.keys())
        assert (
            set(y_count.keys()) == set(g_count.keys()) == set([0, 1])
        ), "Label (y) and sensitive attribute (s) values should be in \{0, 1\}."
        data_pos_ratio = (y == 1).sum() / len(y)
        avg_group_size = len(y) / len(g_count)
        stats_grp = {}
        for g in groups:
            mask_g = s == g
            y_g = y[mask_g]
            g_pos_ratio = (y_g == 1).sum() / g_count[g]
            stats_grp[g] = {
                'size': g_count[g],
                'n_pos': (y_g == 1).sum(),
                'n_neg': (y_g == 0).sum(),
                'pos_ratio': g_pos_ratio,
                'neg_ratio': 1 - g_pos_ratio,
                'idx': mask_to_idx(mask_g),  # global index
                'idx_pos': mask_to_idx(
                    (y == 1) & mask_g
                ),  # global index of positive samples
                'idx_neg': mask_to_idx(
                    (y == 0) & mask_g
                ),  # global index of negative samples
                # group type: advantaged or disadvantaged
                'type_group': 'adv' if g_pos_ratio > data_pos_ratio else 'dis',
                # group size: majority or minority
                'type_size': 'maj' if g_count[g] > avg_group_size else 'min',
            }
        self.stats_grp = stats_grp
        self.stats_meta = {
            'pos_ratio': data_pos_ratio,
            # group label with max/min positive ratio
            'g_max_pos_ratio': max(stats_grp, key=lambda x: stats_grp[x]['pos_ratio']),
            'g_min_pos_ratio': min(stats_grp, key=lambda x: stats_grp[x]['pos_ratio']),
            # group label with max/min size
            'g_max_size': max(stats_grp, key=lambda x: stats_grp[x]['size']),
            'g_min_size': min(stats_grp, key=lambda x: stats_grp[x]['size']),
        }

    def print_distribution(self, y, s, prefix=""):
        res = {}
        for g in self.stats_grp.keys():
            msk_g = s == g
            for y_ in [0, 1]:
                y_mask = y == y_
                res[f"g={g}, y={y_}"] = (msk_g & y_mask).sum()
        print(prefix, res)

    def new_estimator(self, estimator=None, random_state=None):
        if estimator is None:
            estimator = clone(self.estimator_)
        else:
            estimator = clone(estimator)
        try:
            estimator.set_params(
                random_state=random_state,
            )
        except:
            pass
        return estimator

    def fit(self, X, y, sensitive_features: np.ndarray):
        s = sensitive_features
        self._init_stats(X, y, s)
        # empty estimators
        self.estimators_grp_ = {g: [] for g in self.stats_grp.keys()}
        self.estimators_ = []
        self.fit_iter_grp_experts(X, y, s, random_state=self.random_state)
        for i in range(self.n_estimators):
            seed = self.random_seeds[i]
            self.fit_iter_calibrated(X, y, s, random_state=seed)
        return self

    def fit_iter_grp_experts(
        self, X, y, sensitive_features: np.ndarray, random_state=None
    ):
        s = sensitive_features
        grp = self.stats_grp
        seeds = seed_generator(random_state)
        X_, y_, s_ = X.copy(), y.copy(), s.copy()

        # group-specific estimators
        for g in grp.keys():
            msk_g = s == g
            X_g, y_g, s_g = X_[msk_g], y_[msk_g], s_[msk_g]
            estimator = self.new_estimator(
                estimator=self.init_estimator, random_state=next(seeds)
            )
            try:
                estimator.fit(X_g, y_g, sensitive_features=s_g)
            except Exception as e:
                estimator.fit(X_g, y_g)
            self.estimators_grp_[g].append(estimator)

    def fit_iter_calibrated(
        self, X, y, sensitive_features: np.ndarray, random_state=None
    ):
        assert (
            min([len(clfs) for clfs in self.estimators_grp_.values()]) > 0
        ), "Group-specific experts not fitted yet."
        s = sensitive_features
        how = self.how
        grp = self.stats_grp
        meta = self.stats_meta
        seeds = seed_generator(random_state)
        X_, y_, s_ = X.copy(), y.copy(), s.copy()

        # compute target positive ratio
        if how['pos_ratio'] == 'overall':
            pos_ratio_tgt = meta['pos_ratio']
        elif how['pos_ratio'] == 'max':
            pos_ratio_tgt = grp[meta['g_max_pos_ratio']]['pos_ratio']
        elif how['pos_ratio'] == 'min':
            pos_ratio_tgt = grp[meta['g_min_pos_ratio']]['pos_ratio']
        else:
            raise ValueError(
                f"Unsupport pos_ratio: {how['pos_ratio']}, choose from ['overall', 'max', 'min']"
            )

        drop_ratio = self.how['drop_ratio']
        # print (self.estimators_grp_)
        y_pred_grp = {
            g: self.predict_estimators(
                X_, sensitive_features=s_, estimators=self.estimators_grp_[g]
            )
            for g in grp.keys()
        }
        dict_target_subgrp_labels = {g: None for g in grp.keys()}
        msk_drop = np.zeros_like(y_, dtype=bool)
        for g in grp.keys():
            rng = np.random.RandomState(
                next(seeds)
            )  # random number generator for this group
            msk_g = s_ == g
            info = grp[g]
            n_neg = info['n_neg']
            n_pos = info['n_pos']
            # if pos_ratio is already satisfied, skip
            if grp[g]['pos_ratio'] == pos_ratio_tgt:
                continue
            # compute disagreement mask
            y_pred_self = y_pred_grp[g]
            y_pred_other = np.mean(
                [y_pred_grp[g_] for g_ in grp.keys() if g_ != g], axis=0
            )

            # locating samples to drop & compute max number of samples to drop
            if info['type_group'] == 'adv':  # advantaged group
                assert (
                    info['pos_ratio'] > pos_ratio_tgt
                ), "Advantaged group should have higher pos_ratio than target."
                # drop ground truth positive samples
                msk_drop_subgrp = (y_ == 1) & msk_g
                dict_target_subgrp_labels[g] = 1
                if self.how['action'] == 'drop':
                    n_drop_max = n_pos - int((n_neg / (1 - pos_ratio_tgt)) - n_neg)
                elif self.how['action'] == 'flip':
                    n_drop_max = n_pos - int(info['size'] * pos_ratio_tgt)
                else:
                    raise ValueError(
                        f"Unsupport action: {self.how['action']}, choose from ['drop', 'flip']"
                    )

            elif info['type_group'] == 'dis':  # disadvantaged group
                assert (
                    info['pos_ratio'] < pos_ratio_tgt
                ), "Disadvantaged group should have lower pos_ratio than target."
                # drop ground truth negative samples
                msk_drop_subgrp = (y_ == 0) & msk_g
                dict_target_subgrp_labels[g] = 0
                if self.how['action'] == 'drop':
                    n_drop_max = n_neg - int((n_pos / pos_ratio_tgt) - n_pos)
                elif self.how['action'] == 'flip':
                    n_drop_max = n_neg - int(info['size'] * (1 - pos_ratio_tgt))
                else:
                    raise ValueError(
                        f"Unsupport action: {self.how['action']}, choose from ['drop', 'flip']"
                    )

            else:
                raise ValueError(f"Unrecognized group type: {grp[g]['type_group']}")

            if self.how['consensus'] == 'disag':
                msk_agree = y_pred_self != y_pred_other
            elif self.how['consensus'] == 'other-disag':
                msk_agree = (y_pred_self != y_pred_other) & (y_pred_self == y_)
            else:
                raise ValueError(
                    f"Unsupport consensus: {self.how['consensus']}, choose from ['disag', 'other-disag']"
                )

            # random sampling
            n_drop = int(n_drop_max * drop_ratio)  # number of samples to drop
            idx_agree_g = mask_to_idx(msk_drop_subgrp & msk_agree)
            idx_disag_g = mask_to_idx(msk_drop_subgrp & ~msk_agree)
            # assign sample probability
            idx_g = np.concatenate([idx_disag_g, idx_agree_g])
            p_drop_g = np.concatenate(
                [
                    np.ones_like(idx_disag_g, dtype=float),
                    np.full_like(idx_agree_g, self.how['uniformity'], dtype=float),
                ]
            )
            p_drop_g /= p_drop_g.sum()
            # print(f"G={g}, y={dict_target_subgrp_labels[g]}, subgrp_size={msk_drop_subgrp.sum()}, n_drop_max={n_drop_max}, n_drop={n_drop}, p_drop_g={p_drop_g}")
            # print(p_drop_g.max(), p_drop_g.min())
            # sample from disagreement samples
            idx_drop_g = rng.choice(idx_g, size=n_drop, replace=False, p=p_drop_g)
            # print (idx_drop_g)
            # # if enough samples with disagreement to drop
            # if n_drop <= len(idx_disag_g):
            #     idx_drop_g = rng.choice(idx_disag_g, size=n_drop, replace=False)
            # # else, drop all samples with disagreement and randomly sample from the rest
            # else:
            #     # idx_drop_g = idx_disag_g
            #     idx_drop_g = np.concatenate([
            #         idx_disag_g,
            #         rng.choice(idx_agree_g, size=n_drop - len(idx_disag_g), replace=False)
            #     ])
            msk_drop_g = idx_to_mask(idx_drop_g, len(y_))
            # update the drop mask
            msk_drop |= msk_drop_g

        # get calibrated set
        msk_keep = ~msk_drop
        if self.how['action'] == 'drop':
            X_, y_, s_ = X_[msk_keep], y_[msk_keep], s_[msk_keep]
        elif self.how['action'] == 'flip':
            y_[msk_drop] = 1 - y_[msk_drop]
        else:
            raise ValueError(
                f"Unsupport action: {self.how['action']}, choose from ['drop', 'flip']"
            )
        # bootstrap sampling
        if self.how['bootstrap']:
            X_, y_, s_ = self.bootstrap_sampling(
                X_,
                y_,
                s_,
                dict_target_subgrp_labels=dict_target_subgrp_labels,
                stratify=self.how['bootstrap'],
                random_state=random_state,
            )
        # self.print_distribution(y_, s_, "Calibrated set")
        # fit the calibrated estimator
        estimator = self.new_estimator(random_state=random_state)
        try:
            estimator.fit(X_, y_, sensitive_features=s_)
        except Exception as e:
            # print(f"Error: {e}")
            estimator.fit(X_, y_)
        self.estimators_.append(estimator)

    def predict(self, X, sensitive_features):
        y_pred_proba = self.predict_proba(X, sensitive_features)
        return np.argmax(y_pred_proba, axis=1)

    def get_estimators(self):
        pred_mode = self.how['pred']
        if pred_mode == 'adv+dis':
            estimators = self.estimators_grp_[0] + self.estimators_grp_[1]
        elif pred_mode == 'adv':
            estimators = self.estimators_grp_[self.meta_info['g_adv']]
        elif pred_mode == 'dis':
            estimators = self.estimators_grp_[self.meta_info['g_dis']]
        elif pred_mode == 'calib':
            estimators = self.estimators_
        else:
            raise ValueError(f"Unknown pred_mode: {pred_mode}")
        return estimators

    def predict_proba(self, X, sensitive_features):
        estimators = self.get_estimators()
        return self.predict_proba_estimators(X, sensitive_features, estimators)

    def predict_proba_estimators(self, X, sensitive_features, estimators: list):
        def to_one_hot(y, n_classes):
            y_one_hot = np.zeros((len(y), n_classes))
            y_one_hot[np.arange(len(y)), y] = 1
            return y_one_hot

        if self.flag_proba:
            try:
                y_pred_proba = [
                    estimator.predict_proba(X, sensitive_features=sensitive_features)
                    for estimator in estimators
                ]
            except Exception as e:
                y_pred_proba = [estimator.predict_proba(X) for estimator in estimators]
        else:
            try:
                y_pred_proba = [
                    estimator.predict(X, sensitive_features=sensitive_features)
                    for estimator in estimators
                ]
            except Exception as e:
                y_pred_proba = [estimator.predict(X) for estimator in estimators]
            y_pred_proba = [to_one_hot(y_pred, n_classes=2) for y_pred in y_pred_proba]
        y_pred_proba = np.mean(np.array(y_pred_proba), axis=0)
        return y_pred_proba

    def predict_estimators(self, X, sensitive_features, estimators: list):
        if self.flag_proba:
            y_pred_proba = self.predict_proba_estimators(
                X, sensitive_features, estimators
            )
            return np.argmax(y_pred_proba, axis=1)
        else:
            try:
                y_pred = [
                    estimator.predict(X, sensitive_features=sensitive_features)
                    for estimator in estimators
                ]
            except Exception as e:
                y_pred = [estimator.predict(X) for estimator in estimators]
            return np.mean(np.array(y_pred), axis=0)

    @staticmethod
    def bootstrap_sampling(
        X, y, s, dict_target_subgrp_labels=None, stratify='regular', random_state=None
    ):
        rng = np.random.RandomState(random_state)
        if stratify is 'regular':
            idx = rng.choice(len(y), size=len(y), replace=True)
        elif stratify == 'y':
            idx = np.concatenate(
                [
                    rng.choice(mask_to_idx(y == y_), size=(y == y_).sum(), replace=True)
                    for y_ in np.unique(y)
                ]
            )
        elif stratify == 's':
            idx = np.concatenate(
                [
                    rng.choice(mask_to_idx(s == s_), size=(s == s_).sum(), replace=True)
                    for s_ in np.unique(s)
                ]
            )
        elif stratify == 'y+s':
            idx = np.concatenate(
                [
                    rng.choice(
                        mask_to_idx((y == y_) & (s == s_)),
                        size=((y == y_) & (s == s_)).sum(),
                        replace=True,
                    )
                    for y_ in np.unique(y)
                    for s_ in np.unique(s)
                ]
            )
        elif stratify == 'untargeted':
            idxs = []
            for g in np.unique(s):
                for y_ in np.unique(y):
                    msk = (y == y_) & (s == g)
                    if y_ == dict_target_subgrp_labels[g]:
                        # if the target subgrp is already undersampled, do not sample from it
                        idxs.append(mask_to_idx(msk))
                    else:
                        # if the target subgrp is not undersampled, bootstrap sample from it
                        idxs.append(
                            rng.choice(mask_to_idx(msk), size=msk.sum(), replace=True)
                        )
            idx = np.concatenate(idxs)
        else:
            raise ValueError(f"Unknown bootstrap stratify: {stratify}")
        return X[idx], y[idx], s[idx]
