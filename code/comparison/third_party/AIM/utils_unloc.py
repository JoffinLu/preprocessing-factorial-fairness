import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os


def plot_in_and_cross_group_comp_sample_distr(comp_mat, idx_prv, idx_prt, t_ig=1, t_cg=0):

    n_prv, n_prt = len(idx_prv), len(idx_prt)
    comp_mat_in_prv = comp_mat[idx_prv][:, idx_prv]
    comp_mat_in_prt = comp_mat[idx_prt][:, idx_prt]

    prv_sum_ig = np.sum(comp_mat_in_prv, axis=1)
    prt_sum_ig = np.sum(comp_mat_in_prt, axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(sorted(prv_sum_ig, reverse=True))
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('# comparable samples')
    axs[0].set_title(f'PRV in-group comparable ({(prv_sum_ig == 0).sum()/n_prv:.2%} empty)')

    axs[1].plot(sorted(prt_sum_ig, reverse=True))
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('# comparable samples')
    axs[1].set_title(f'PRT in-group comparable ({(prt_sum_ig == 0).sum()/n_prt:.2%} empty)')

    plt.suptitle('In-group comparable samples')
    plt.tight_layout()
    plt.show()

    comp_mat_cross_group = comp_mat[idx_prv][:, idx_prt]
    prv_sum = comp_mat_cross_group.sum(axis=1)
    prt_sum = comp_mat_cross_group.sum(axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(sorted(prv_sum, reverse=True))
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('# comparable samples')
    axs[0].set_title(f'PRV cross-group comparable ({(prv_sum == 0).sum()/n_prv:.2%} empty)')

    axs[1].plot(sorted(prt_sum, reverse=True))
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('# comparable samples')
    axs[1].set_title(f'PRT cross-group comparable ({(prt_sum == 0).sum()/n_prt:.2%} empty)')

    plt.suptitle('Cross-group comparable samples')
    plt.tight_layout()
    plt.show()

def normalize_graph_adj(A, how='symmetric'):
    if how == 'sym':
        D_sqrt_inv = 1 / np.sqrt(np.sum(A, axis=1))
        D_sqrt_inv[np.isinf(D_sqrt_inv)] = 0
        normalized_A = A * D_sqrt_inv[:, np.newaxis]
        normalized_A = normalized_A * D_sqrt_inv[np.newaxis, :]
        return normalized_A
    
    elif how == 'left':
        D_inv = 1 / np.sum(A, axis=1)
        D_inv[np.isinf(D_inv)] = 0
        normalized_A = (A.T * D_inv).T
        return normalized_A
    
    elif how == 'right':
        D_inv = 1 / np.sum(A, axis=0)
        D_inv[np.isinf(D_inv)] = 0
        normalized_A = A * D_inv
        return normalized_A    

    else:
        raise ValueError("Invalid normalization method. Supported methods: 'symmetric', 'left', 'right', 'symmetric_laplacian'")


class DataProcessor():
    
    @staticmethod
    def parse_feature_types(df, onehot_name_seperator='=', verbose=True):
        """
        Parse feature types from the dataset
        
        Parameters
        ----------
        data : FairDataset
            FairDataset object
        onehot_name_seperator : str, optional
            Seperator of one-hot encoded feature names, by default '='
        
        Returns
        -------
        idx_feats_con : list
            Indices of continuous features
        idx_feats_cat : list
            Indices of categorical features
        idx_feats_onehot : list
            Indices of one-hot encoded features
        """

        # get all feature columns
        feats_all = df.columns

        # get onehot feature columns and continuous feature columns
        feats_binary = feats_all[df[feats_all].isin([0, 1]).all()]
        feats_con = list(feats_all.drop(feats_binary))
        feats_cat, feats_onehot = [], []
        for feat in feats_binary:
            if onehot_name_seperator in feat:
                feats_onehot.append(feat)
            else:
                feats_cat.append(feat)
        feats_onehot_raw = list(set([feat.split(onehot_name_seperator)[0] for feat in feats_onehot]))
        
        idx_feats_con = [feats_all.get_loc(feat) for feat in feats_con]
        idx_feats_cat = [feats_all.get_loc(feat) for feat in feats_cat]
        idx_feats_onehot = [feats_all.get_loc(feat) for feat in feats_onehot]

        idx_feats = {
            'con': idx_feats_con,
            'cat': idx_feats_cat,
            'onehot': idx_feats_onehot,
        }
        num_feats = {
            'con': len(feats_con),
            'cat': len(feats_cat),
            'onehot': len(feats_onehot),
            'onehot_raw': len(feats_onehot_raw),
            'all_raw': len(feats_con) + len(feats_cat) + len(feats_onehot_raw),
            'all_continuous': len(feats_con),
            'all_catgorical': len(feats_cat) + len(feats_onehot_raw),
        }
        name_feats = {
            'con': feats_con,
            'cat': feats_cat,
            'onehot': feats_onehot,
            'onehot_raw': feats_onehot_raw,
            'all_raw': feats_con + feats_cat + feats_onehot_raw,
        }
        
        if verbose:
            print(
                f"////// Feature Numbers //////\n"
                f"Total:        {len(feats_all):<3d}\n"
                f"Continuous:   {len(feats_con):<3d}\t e.g., {feats_con[:5]} \n"
                f"Categorical:  {len(feats_cat):<3d}\t e.g., {feats_cat[:5]} \n"
                f"One-hot:      {len(feats_onehot):<3d}\t e.g., {feats_onehot[:5]}\n"
                f"Raw one-hot:  {len(feats_onehot_raw):<3d}\t e.g., {feats_onehot_raw[:5]}\n"
                f"Raw total:    {num_feats['all_raw']:<3d}"
            )

        return idx_feats, num_feats, name_feats

    def reverse_one_hot_encoding(self, df, onehot_name_seperator='='):
        # Extract the prefix of one-hot encoded columns
        prefix = df.columns.str.split(onehot_name_seperator, expand=True).get_level_values(0).unique()
        # Initialize an empty DataFrame to store the reversed one-hot encoding
        reversed_df = pd.DataFrame(index=df.index)
        # Iterate through each prefix and reverse the one-hot encoding
        for p in prefix:
            # Filter columns with the current prefix
            columns = df.columns[df.columns.str.startswith(f"{p}=")]
            # Get the column with maximum value (1) for each row
            reversed_df[p] = df[columns].idxmax(axis=1).str.replace(f"{p}=", "")

        return reversed_df

    def get_reverse_onehot_dataframe(self, df, name_feats, encode=True):

        if len(name_feats['onehot']) == 0:
            return df, name_feats['con'], name_feats['cat']
        
        df_no_onehot = self.reverse_one_hot_encoding(df[name_feats['onehot']])
        df_no_onehot = pd.concat([df[name_feats['con']], df[name_feats['cat']], df_no_onehot], axis=1)
        feats_con = name_feats['con']
        feats_cat = name_feats['cat'] + name_feats['onehot_raw']

        if encode:
            # Map non-numeric features to integer-encoded values
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            for feat in feats_cat:
                df_no_onehot[feat] = label_encoder.fit_transform(df_no_onehot[feat])

        return df_no_onehot, feats_con, feats_cat
    
    def get_raw_dataframe(self, df, data, train_idx, name_feats):
        df_raw, feats_con, feats_cat = self.get_reverse_onehot_dataframe(df, name_feats, encode=False)
        data_raw_train = data.df_raw.iloc[train_idx].reset_index(drop=True)
        df_raw[feats_con] = data_raw_train[feats_con]
        df_raw['Y'] = data_raw_train[data.y_col]
        df_raw['S'] = data_raw_train[data.s_col]
        return df_raw


class ComparableSampleAnalyzer:
        
        def __init__(
                self, df, feats_con, feats_cat, save_path, data_setting, 
                compress=True, verbose=False
            ):
            self.df = df
            self.feats_con = feats_con
            self.feats_cat = feats_cat
            self.con_data = df[feats_con].values
            self.cat_data = df[feats_cat].values
            self.save_path = save_path
            self.data_setting = data_setting
            # save path should be a valid directory
            assert os.path.isdir(save_path), f"save_path {save_path} is not a valid directory"
            # it should contains two subdirectories: diff and gower
            assert os.path.isdir(f"{save_path}/diff"), f"{save_path}/diff is not a valid directory"
            assert os.path.isdir(f"{save_path}/gower"), f"{save_path}/gower is not a valid directory"
            self.compress = compress
            self.verbose = verbose
            if verbose:
                self._print_info()

        def _print_info(self):
            print(
                f"////// Comparable Analyzer //////\n"
                f"Data shape: {self.df.shape}\n"
                f"[{len(self.feats_con)}] Continuous features: {self.feats_con}\n"
                f"[{len(self.feats_cat)}] Categorical features: {self.feats_cat}\n"
            )

        def _save_diff_matrices(self, path, prefix):
            file_path = f"{path}/diff/{prefix}_diff_matrices.h5"
            with h5py.File(file_path, "w") as file:
                if self.compress:
                    print (f"Compressing & saving diff matrices to {file_path}")
                    file.create_dataset("con_diff_matrix", data=self.con_diff_matrix, compression='gzip', compression_opts=1, chunks=True)
                    file.create_dataset("cat_diff_matrix", data=self.cat_diff_matrix, compression='gzip', compression_opts=1, chunks=True)
                else:
                    print (f"Saving diff matrices to {file_path}")
                    file.create_dataset("con_diff_matrix", data=self.con_diff_matrix)
                    file.create_dataset("cat_diff_matrix", data=self.cat_diff_matrix)
            print(f"Diff matrices saved to {file_path}")

        def _load_diff_matrices(self, path, prefix):
            file_path = f"{path}/diff/{prefix}_diff_matrices.h5"
            with h5py.File(file_path, "r") as file:
                self.con_diff_matrix = file["con_diff_matrix"][()]
                self.cat_diff_matrix = file["cat_diff_matrix"][()]
            print(f"Diff matrices loaded from {file_path}")

        def _compute_diff_matrices(self, save=True):
            # Load from file if the diff matrices are already saved
            path = self.save_path
            prefix = self.data_setting
            try:
                self._load_diff_matrices(path, prefix)
                return
            except:
                pass
            
            start_time = time.time()
            con_data, cat_data = self.con_data, self.cat_data
            # Compute the absolute differences for continuous features
            self.con_diff_matrix = np.max(np.abs(con_data[:, np.newaxis, :] - con_data), axis=-1)
            if self.verbose:
                print(f"Time for computing continuous differences: {time.time() - start_time:.2f}s")
                start_time = time.time()
            # Compute the differences for categorical features
            self.cat_diff_matrix = np.sum(cat_data[:, np.newaxis, :] != cat_data, axis=-1)
            if self.verbose:
                print(f"Time for computing categorical differences: {time.time() - start_time:.2f}s")
            # Save the diff matrices for future use
            if save:
                self._save_diff_matrices(path, prefix)

        def get_comparable_matrix(self, t_con, t_cat, relax=True, max_relax=3, relax_factor=0.1, include_self=False):
            
            assert isinstance(t_con, float) and 0 <= t_con <= 1, "t_con should be in [0, 1]"
            assert isinstance(t_cat, int) and t_cat >= 0, "t_cat should be a non-negative integer"
            assert max_relax >= 1, "max_relax should be at least 1"

            if self.verbose:
                print (f"Threshold for continuous features:  {t_con}")
                print (f"Threshold for categorical features: {t_cat}")
            
            # check if self has the attribute con_diff_matrix and cat_diff_matrix
            if not hasattr(self, "con_diff_matrix") or not hasattr(self, "cat_diff_matrix"):
                self._compute_diff_matrices()

            start_time = time.time()
            # Check continuous features condition
            con_condition_matrix = self.con_diff_matrix <= t_con
            if self.verbose:
                print(f"Time for computing continuous condition matrix: {time.time() - start_time:.2f}s")
                start_time = time.time()
            # Check categorical features condition
            cat_condition_matrix = self.cat_diff_matrix <= t_cat
            if self.verbose:
                print(f"Time for computing categorical condition matrix: {time.time() - start_time:.2f}s")

            # Combine conditions for comparability
            comp_matrix = con_condition_matrix & cat_condition_matrix
            comp_matrix = comp_matrix.astype(float)

            if not include_self:
                np.fill_diagonal(comp_matrix, False)

            # link isolated nodes without comparable samples
            if relax:
                idx_iso = np.where(comp_matrix.sum(axis=1) == 0)[0]
                print (f"Linking isolated nodes ({len(idx_iso)}) ... ", end="")
                # print (len(idx_iso), idx_iso)
                for i in idx_iso:
                    con_diff, cat_diff = self.con_diff_matrix[i], self.cat_diff_matrix[i]
                    for r in range(2, max_relax+1):
                        weight = relax_factor ** (r-1)
                        sub_comp_mask = (con_diff <= t_con * r) & (cat_diff <= t_cat * r)
                        sub_comp_mask[i] = False
                        sub_comp_idx = np.where(sub_comp_mask)[0]
                        if sub_comp_mask.sum() > 0:
                            # link with the one with samllest con_diff
                            sub_comp_con_diff = con_diff[sub_comp_idx]
                            sub_comp_idx = sub_comp_idx[sub_comp_con_diff.argmin()]
                            comp_matrix[i, sub_comp_idx] = weight
                            comp_matrix[sub_comp_idx, i] = weight
                            break
                    if comp_matrix[i].sum() == 0:
                        comp_matrix[i, i] = 1
                print (f"Done! {(comp_matrix.sum(axis=0) == 0).sum()} remains.")

            return comp_matrix

        def analyze_comparable_matrix(self, comp_mat):
            n_edges = (comp_mat > 0).sum(axis=1)
            print(
                f"# ISOLATED samples with no comparable samples: {np.sum(n_edges == 0)}\n"
            )

            # Sort the row sum in descending order
            plt.plot(sorted(n_edges, reverse=True))
            plt.xlabel('Index')
            plt.ylabel('# Edges')
            plt.title('Sorted number of edges for each sample')
            plt.show()

        def get_gower_matrix(self):
            
            path = self.save_path
            prefix = self.data_setting

            try:
                self.gower_matrix = self._load_matrix(path, "gower", prefix, "gower")
                return self.gower_matrix
            except:
                pass

            import gower
            self.gower_matrix = gower.gower_matrix(
                self.df,
                cat_features=[True if feat in self.feats_cat else False for feat in self.df.columns]
            )
            # Save the gower matrix for future use
            self._save_matix(self.gower_matrix, path, "gower", prefix, "gower")
            return self.gower_matrix

        def _save_matix(self, mat, path, type, prefix, setting):
            file_path = f"{path}/{type}/{prefix}_{setting}_matrix.h5"
            with h5py.File(file_path, "w") as file:
                if self.compress:
                    print (f"Compressing & saving {type}-{setting} matrix to {file_path}")
                    file.create_dataset("data", data=mat, compression='gzip', compression_opts=1, chunks=True)
                else:
                    print (f"Saving {type}-{setting} matrix to {file_path}")
                    file.create_dataset("data", data=mat)
            print(f"Matrix saved to {file_path}")

        def _load_matrix(self, path, type, prefix, setting):
            file_path = f"{path}/{type}/{prefix}_{setting}_matrix.h5"
            with h5py.File(file_path, "r") as file:
                mat = file["data"][()]
            print(f"Matrix loaded from {file_path}")
            return mat

        def compute_rwr_proximity(self, adj, restart_prob, norm='sym', remove_diag=True):
            """
            Compute the RWR proximity matrix from the adjacency matrix
            """
            assert 0 <= restart_prob <= 1, "restart_prob should be in [0, 1]"

            # Normalize the adjacency matrix
            W = normalize_graph_adj(adj, how=norm)
            # Compute the RWR proximity matrix
            Q = np.linalg.inv(np.diag(np.ones(W.shape[0])) - restart_prob * W)
            P = (1 - restart_prob) * Q
            if remove_diag:
                np.fill_diagonal(P, 0)
            return P

        def get_proximity(self, t_con, t_cat, restart_prob, mat_norm='sym', relax=True, max_relax=3, relax_factor=0.1, include_self=False):
            """
            Get the proximity matrix from the comparable matrix
            """

            path = self.save_path
            prefix = self.data_setting
            setting = f"TC={t_con}_TD={t_cat}_RP={restart_prob}_norm={mat_norm}"

            # if has already cached, load from file
            try:
                self.P = self._load_matrix(path, "proximity", prefix, setting)
                return self.P
            except:
                pass
            
            print (f"Computing proximity matrix for {setting} ...")
            comp_mat = self.get_comparable_matrix(
                t_con, t_cat, relax=relax, max_relax=max_relax, 
                relax_factor=relax_factor, include_self=include_self
            )
            self.P = self.compute_rwr_proximity(comp_mat, restart_prob, norm=mat_norm)
            self._save_matix(self.P, path, "proximity", prefix, setting)
            
            return self.P

def plot_solution_explanability_ratio(w, top_n_values, s_prv=1, s_prt=0, ax=None, title=None):
    
    for s in [s_prv, s_prt]:
        if s == s_prv:
            s_name = 'Priviledged'
            print(f"Computing solution explanabilty ratio for Priviledged Group {s_prv} ...")
            w_cumsum = np.cumsum(np.sort(w / w.sum(axis=1, keepdims=True), axis=1)[:, ::-1], axis=1)
        elif s == s_prt:
            s_name = 'Unpriviledged'
            print(f"Computing solution explanabilty ratio for Unpriviledged Group {s_prt} ...")
            wt = w.T
            w_cumsum = np.cumsum(np.sort(wt / wt.sum(axis=1, keepdims=True), axis=1)[:, ::-1], axis=1)
        else:
            raise NotImplementedError
        fig, axes = plt.subplots(1, len(top_n_values), figsize=(20, 5))
        for i, n in enumerate(top_n_values):
            axes[i].plot(np.sort(w_cumsum[:, n]))
            axes[i].set(
                title=f'Top {n} corresponding cases',
                ylabel='Sum of correspondence scores',
                ylim=[-.05, 1.05],
            )
            axes[i].grid()
        fig.suptitle(f"Solution explanabilty ratio for {s_name} group")
        plt.tight_layout()
        plt.show()

class UnfairnessAttributer():
    
    def __init__(self, P) -> None:
        self.P = P
        
    def compute_confidence(self, X, y, s, s_prv=1, s_prt=0, use_conf=True, visualize=False):
        idx_prv, idx_prt = np.where(s == s_prv)[0], np.where(s == s_prt)[0]
        P = self.P
        
        n_sample, n_feat = X.shape

        # compute confidence score
        y_mat = np.tile(y, (n_sample, 1)).astype(bool)
        y_disag_mat = (y_mat != y_mat.T)

        P_prv, P_prt = P[idx_prv][:, idx_prv], P[idx_prt][:, idx_prt]
        y_disag_prv = y_disag_mat[idx_prv][:, idx_prv]
        y_disag_prt = y_disag_mat[idx_prt][:, idx_prt]

        """confidence as same label ratio from the same group"""
        conf_score_prv = (P_prv*~y_disag_prv).sum(axis=1) / (P_prv).sum(axis=1)
        conf_score_prv[np.isnan(conf_score_prv)] = conf_score_prv[~np.isnan(conf_score_prv)].mean()
        conf_score_prt = (P_prt*~y_disag_prt).sum(axis=1) / (P_prt).sum(axis=1)
        conf_score_prt[np.isnan(conf_score_prt)] = conf_score_prt[~np.isnan(conf_score_prt)].mean()

        conf_score = np.zeros(n_sample)
        conf_score[idx_prv] = conf_score_prv
        conf_score[idx_prt] = conf_score_prt
        
        """compute unfairness score"""
        y_disag_prv_prt = y_disag_mat[idx_prv][:, idx_prt]

        # take confidence score into account
        w = P[idx_prv][:, idx_prt]
        if use_conf:
            w_prt_norm = w * conf_score_prv[:, np.newaxis]
            w_prv_norm = w * conf_score_prt
        else:
            w_prt_norm = w.copy()
            w_prv_norm = w.copy()

        w_prt_norm /= w_prt_norm.sum(axis=s_prt, keepdims=True)
        w_prt_norm[np.isnan(w_prt_norm)] = 0
        w_prv_norm /= w_prv_norm.sum(axis=s_prv, keepdims=True)
        w_prv_norm[np.isnan(w_prv_norm)] = 0

        # based on the correspondence matrix w, compute the disagreement score for each case
        prt_disag_score = (w_prt_norm*y_disag_prv_prt).sum(axis=0)
        prv_disag_score = (w_prv_norm*y_disag_prv_prt).sum(axis=1)
        # compute the disagreement contribution score for each case
        prt_disag_contrib = (w_prv_norm*y_disag_prv_prt).sum(axis=0)
        prv_disag_contrib = (w_prt_norm*y_disag_prv_prt).sum(axis=1)

        disag_score = np.zeros(n_sample)
        disag_contr = np.zeros(n_sample)
        # use disagreement score
        disag_score[idx_prv] = prv_disag_score
        disag_score[idx_prt] = prt_disag_score
        # use disagreement contribution score
        disag_contr[idx_prv] = prv_disag_contrib
        disag_contr[idx_prt] = prt_disag_contrib
        
        if visualize:
            y_prv, y_prt = y[idx_prv], y[idx_prt]
            k_bins = 40
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            sns.histplot(data=pd.DataFrame({
                'disag_score': prv_disag_score,
                'label': y_prv,
            }), x='disag_score', bins=k_bins, hue='label', element="step", ax=ax1)
            ax1.set_title('PRV group')
            sns.histplot(data=pd.DataFrame({
                'disag_score': prt_disag_score,
                'label': y_prt,
            }), x='disag_score', bins=k_bins, hue='label', element="step", ax=ax2)
            ax2.set_title('PRT group')
            plt.suptitle(f"Disagreement Score")
            plt.tight_layout()
            plt.show()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            sns.histplot(data=pd.DataFrame({
                'disag_score': prv_disag_contrib,
                'label': y_prv,
            }), x='disag_score', bins=k_bins, hue='label', element="step", ax=ax1)
            ax1.set_title('PRV group')
            # ax1.set_yscale('log')
            sns.histplot(data=pd.DataFrame({
                'disag_score': prt_disag_contrib,
                'label': y_prt,
            }), x='disag_score', bins=k_bins, hue='label', element="step", ax=ax2)
            ax2.set_title('PRT group')
            # ax2.set_yscale('log')
            plt.suptitle(f"Contributed Disagreement Score")
            plt.tight_layout()
            plt.show()

        df_score = pd.DataFrame({
            'conf': conf_score,
            'unf': disag_score,
            'unf_contr': disag_contr,
        })
        
        return df_score