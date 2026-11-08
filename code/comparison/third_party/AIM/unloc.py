import sklearn
# import shap

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils_unloc import DataProcessor, UnfairnessAttributer, ComparableSampleAnalyzer
from utils import generate_random_seeds

def get_filter_mask(i, y, s, group_constraint='na', label_constraint='na'):
    filter_msk = np.ones(len(y), dtype=bool)
    if group_constraint == 'na':
        pass
    elif group_constraint == 'same':
        filter_msk[s != s[i]] = False
    elif group_constraint == 'diff':
        filter_msk[s == s[i]] = False
    else:
        raise NotImplementedError(f"Invalid group_constraint: {group_constraint}")

    if label_constraint == 'na':
        pass
    elif label_constraint == 'same':
        filter_msk[y != y[i]] = False
    elif label_constraint == 'diff':
        filter_msk[y == y[i]] = False
    else:
        raise NotImplementedError(f"Invalid label_constraint: {label_constraint}")
    
    return filter_msk

def get_mixup_idx(i, P, y, s, group_constraint='na', label_constraint='na', top_n=10, random_seed=None):
    """Get the most similar index with highest correspondence score (P)"""
    # print (f"P shape: {P.shape} y shape: {y.shape} s shape: {s.shape}, i: {i}")
    filter_msk = get_filter_mask(i, y, s, group_constraint, label_constraint)
    # print (f"group_constraint: {group_constraint} s: s[{i}] = {s[i]}, s[filter_msk] = {s[filter_msk]}")
    # print (f"label_constraint: {label_constraint} y: y[{i}] = {y[i]}, y[filter_msk] = {y[filter_msk]}")
    filter_idx = np.where(filter_msk)[0]
    filter_sort_idx = np.argsort(P[i][filter_idx])
    cand_idx = filter_idx[filter_sort_idx][-top_n:][::-1]
    # print (cand_idx, P[i][cand_idx])
    # permute the comparable indices
    if random_seed is not None:
        np.random.seed(random_seed)
    # randomly select one index from the top_n
    return np.random.choice(cand_idx)

def mixup_instance(i, mixup_i, df_x, df_x_input, name_feats, seed=None):
    feats_encoded = df_x.columns
    feats_input = df_x_input.columns

    if seed is not None:
        np.random.seed(seed)
    mix_weight = np.random.uniform(0, 1, 1)[0]

    x_seed_enc = df_x.iloc[i].values
    x_mix_enc = df_x.iloc[mixup_i].values
    
    x_new_enc = []
    x_new_input = np.zeros(len(feats_input))
    
    for idx_feat_enc, feat_enc in enumerate(feats_encoded):
        if feat_enc in name_feats['con']:
            # continuous feature, linearly mix the two samples
            value = value_input = mix_weight * x_seed_enc[idx_feat_enc] + (1-mix_weight) * x_mix_enc[idx_feat_enc]
            feat_input = feat_enc
        elif feat_enc in name_feats['cat']:
            # categorical feature, randomly choose one of the two samples with the given mix_weight
            value = value_input = np.random.choice([x_seed_enc[idx_feat_enc], x_mix_enc[idx_feat_enc]], p=[mix_weight, 1-mix_weight])
            feat_input = feat_enc
        elif feat_enc in name_feats['onehot_raw']:
            # onehot feature, randomly choose one of the two samples with the given mix_weight
            value = np.random.choice([x_seed_enc[idx_feat_enc], x_mix_enc[idx_feat_enc]], p=[mix_weight, 1-mix_weight])
            value_input = 1
            feat_input = f"{feat_enc}={value}"
        else:
            raise NotImplementedError(f"Feature type not found for: {feat_enc}")

        # fill value in the encoded space
        x_new_enc.append(value)
        # fill value in the input space
        idx_feat_input = feats_input.get_loc(feat_input)
        x_new_input[idx_feat_input] = value_input

    x_new_enc = np.array(x_new_enc)
    return x_new_enc, x_new_input


class UnLoc():
    
    def __init__(self) -> None:
        pass

    def _save_df(self, df, name, path='./data_cache/score'):
        df.to_csv(f"{path}/{name}.csv", index=False)

    def _load_df(self, name, path='./data_cache/score'):
        return pd.read_csv(f"{path}/{name}.csv")

    def compute_df_encoded(self, X, features, s_col):
        dp = DataProcessor()
        df_train_withs = pd.DataFrame(X, columns=features)
        idx_feats, num_feats, name_feats = dp.parse_feature_types(df_train_withs, verbose=False)
        df_train_withs_nooh, feats_con, feats_cat = dp.get_reverse_onehot_dataframe(df_train_withs, name_feats, encode=False)
        self.df_train_withs = df_train_withs
        self.df_train_withs_nooh = df_train_withs_nooh
        self.name_feats_withs = name_feats
        
        # we dont use sensitive attribute for computing comparable samples
        df_train = pd.DataFrame(X, columns=features).drop(s_col, axis=1)
        idx_feats, num_feats, name_feats = dp.parse_feature_types(df_train, verbose=False)
        df_train_encoded, feats_con, feats_cat = dp.get_reverse_onehot_dataframe(df_train, name_feats)

        self.df_train = df_train
        self.df_train_encoded = df_train_encoded
        self.feats_con = feats_con
        self.feats_cat = feats_cat
        self.name_feats = name_feats
        
    def fit_transform(self, X, y, s, data_split_info, features, s_col, prox_kwargs, s_prv=1, s_prt=0):
        # guarantee X's 1st column is sensitive attribute
        assert (X[:, 0] == s).all()
        
        file_name = f"{data_split_info}_{str(prox_kwargs)}_score"
        
        self.compute_df_encoded(X, features, s_col)

        try:
            df_score = self._load_df(file_name)
            print(f"Loaded score from {file_name}.csv")
            return df_score
        except:
            pass

        P = self.get_proximity(X, y, s, data_split_info, features, s_col, prox_kwargs, s_prv=s_prv, s_prt=s_prt)

        # compute confidence score
        attributer = UnfairnessAttributer(P)
        df_score = attributer.compute_confidence(X, y, s, s_prv=s_prv, s_prt=s_prt)

        # save df_score
        self._save_df(df_score, file_name)
        print(f"Saved score to {file_name}.csv")

        return df_score

    def get_proximity(self, X, y, s, data_split_info, features, s_col, prox_kwargs, s_prv=1, s_prt=0):
        
        df_train_encoded, feats_con, feats_cat = self.df_train_encoded, self.feats_con, self.feats_cat
        
        csa = ComparableSampleAnalyzer(
            df_train_encoded, feats_con, feats_cat, 
            save_path='data_cache', data_setting=data_split_info, 
            compress=False, verbose=True
        )
        P = csa.get_proximity(**prox_kwargs)
        return P
    

    def fair_aug(
        self, aug_ratio, X_train, y_train, s_train, P, 
        s_prv=1, s_prt=0, filter=None, weights=None, dummy=False, random_seed=None, verbose=False
    ):
        if aug_ratio == 0:
            return X_train, y_train, s_train, 0
        
        if filter is not None:
            if filter == 'prt-pos':
                mask_seeds = (y_train == 1) & (s_train == s_prt)
            elif filter == 'pos':
                mask_seeds = (y_train == 1)
            else:
                raise NotImplementedError
        else: mask_seeds = np.ones(len(y_train), dtype=bool)
        
        if weights is None or dummy:
            weights = np.ones(len(y_train), dtype=float)
        else:
            assert weights.shape == y_train.shape

        seed_weights = weights[mask_seeds]
        seed_weights /= seed_weights.sum()
        
        # compute n aug
        n_aug = int(aug_ratio * len(seed_weights))
        
        # sample seeds
        idx_seeds = np.where(mask_seeds)[0]
        np.random.seed(random_seed)
        idx_aug_seeds = np.random.choice(
            idx_seeds, n_aug, p=seed_weights, replace=True
        )

        random_seeds = generate_random_seeds(n_aug, random_seed)

        X_new, y_new, s_new = [], [], []

        for i, idx_seed in enumerate(idx_aug_seeds):
            rand_seed = random_seeds[i]
            idx_mixup = get_mixup_idx(
                idx_seed, P, y_train, s_train,
                group_constraint='same', label_constraint='diff', 
                top_n=10, random_seed=rand_seed
            )
            _, x_new_input = mixup_instance(
                idx_seed, idx_mixup, 
                self.df_train_withs_nooh, 
                self.df_train_withs, 
                self.name_feats_withs, 
                seed=rand_seed
            )
            X_new.append(x_new_input)
            y_new.append(y_train[idx_seed])
            s_new.append(s_train[idx_seed])
        
        X_edited = np.concatenate([X_train, np.array(X_new)], axis=0)
        y_edited = np.concatenate([y_train, np.array(y_new)], axis=0)
        s_edited = np.concatenate([s_train, np.array(s_new)], axis=0)
        
        if verbose:
            print(
                f"{len(y_edited) - len(y_train)} samples added from the dataset.\n"
                f"Dataset shape: {X_train.shape} -> {X_edited.shape}\n"
                f"S distribution {np.unique(s_train, return_counts=True)} -> {np.unique(s_edited, return_counts=True)}\n"
                f"Y distribution {np.unique(y_train, return_counts=True)} -> {np.unique(y_edited, return_counts=True)}"
            )

        return X_edited, y_edited, s_edited, n_aug
        

    def fair_removal(
        self, X_train, y_train, s_train, weights, edit_ratio, how='removal',
        s_prv=1, s_prt=0, filter=None, verbose=False, dummy=False, random_state=42,
    ):
        # Sort indices by descending weights
        sorted_indices = np.argsort(weights)[::-1]

        if filter is not None:
            if filter == 'prt-neg':
                filter_idx = (y_train[sorted_indices] == 0) & (s_train[sorted_indices] == s_prt)
            elif filter == 'prv-pos':
                filter_idx = (y_train[sorted_indices] == 1) & (s_train[sorted_indices] == s_prv)
            elif filter == 'prt-neg-prv-pos':
                filter_idx = ((y_train[sorted_indices] == 0) & (s_train[sorted_indices] == s_prt)) | ((y_train[sorted_indices] == 1) & (s_train[sorted_indices] == s_prv))
            elif filter == 'prv':
                filter_idx = (s_train[sorted_indices] == s_prv)
            elif filter == 'prt':
                filter_idx = (s_train[sorted_indices] == s_prt)
            elif filter == 'neg':
                filter_idx = (y_train[sorted_indices] == 0)
            elif filter == 'pos':
                filter_idx = (y_train[sorted_indices] == 1)
            else:
                raise NotImplementedError

            sorted_indices = sorted_indices[filter_idx]
        
        n_edit = int(edit_ratio * len(sorted_indices))
        # select the removal indices
        edit_indices = sorted_indices[:n_edit]

        if dummy:
            # set seed
            np.random.seed(random_state)
            edit_indices = np.random.choice(sorted_indices, n_edit, replace=False)
        else:
            edit_indices = sorted_indices[:n_edit]
        
        # Create edit mask
        if how == 'removal':
            keep_mask = np.ones(len(X_train), dtype=bool)
            keep_mask[edit_indices] = False
            # Apply undersampled mask
            X_edited = X_train[keep_mask]
            y_edited = y_train[keep_mask]
            s_edited = s_train[keep_mask]
        elif how == 'relabel':
            relabel_mask = np.zeros(len(X_train), dtype=bool)
            relabel_mask[edit_indices] = True
            y_edited = y_train.copy()
            y_edited[relabel_mask] = 1 - y_edited[relabel_mask]
            X_edited = X_train
            s_edited = s_train
        else: raise NotImplementedError
        
        if verbose:
            print(
                f"{len(y_train) - len(y_edited)} samples removed from the dataset.\n"
                f"Dataset shape: {X_train.shape} -> {X_edited.shape}\n"
                f"S distribution {np.unique(s_train, return_counts=True)} -> {np.unique(s_edited, return_counts=True)}\n"
                f"Y distribution {np.unique(y_train, return_counts=True)} -> {np.unique(y_edited, return_counts=True)}"
            )
        return X_edited, y_edited, s_edited, n_edit