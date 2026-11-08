import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate


class FairDataset:
    """
    A class for handling fair datasets with sensitive and response attributes.

    Attributes:
        name (str): Name of the dataset.
        fullname (str): Name of the dataset concatenated with the sensitive attribute name.
        csv_path (str): Path to the CSV file containing the dataset.
        s_col (str): Name of the sensitive attribute column in the dataset.
        y_col (str): Name of the response attribute column in the dataset.
        df (pd.DataFrame): Pandas DataFrame containing the dataset.
        split (tuples): the split data in the form of (X, y, s) tuples
                        for training, validation, and test sets.
        split_info (str): Information about the data split, including split sizes,
                          random state, and whether the input data includes the sensitive attribute.
        train_size (float): Proportion of the dataset to include in the training split.
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator for reproducibility.
        x_with_s (bool): Whether the input data should include the sensitive attribute.
    """

    def __init__(
        self,
        dataname: str,
        csv_path: str,
        s_col: str,
        y_col: str,
        normalize: bool = True,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        concat_train_val: bool = False,
        random_state: int = None,
        x_with_s: bool = True,
    ) -> None:
        """
        Initialize the FairDataset object.

        Parameters:
            dataname (str): Name of the dataset.
            csv_path (str): Path to the CSV file containing the dataset.
            s_col (str): Name of the sensitive attribute column in the dataset.
            y_col (str): Name of the response attribute column in the dataset.
            normalize (bool, optional): Whether to normalize the non-sensitive attribute columns.
                                        Defaults to True.
            train_size (float, optional): Proportion of the dataset to include in the training split.
                                          Defaults to 0.6.
            val_size (float, optional): Proportion of the dataset to include in the validation split.
                                        Defaults to 0.2.
            test_size (float, optional): Proportion of the dataset to include in the test split.
                                         Defaults to 0.2.
            concat_train_val (bool, optional): Whether to concatenate the training and validation sets.
            random_state (int, optional): Seed used by the random number generator for reproducibility.
                                          Defaults to None.
            x_with_s (bool, optional): Whether the input data should include the sensitive attribute.
                                       Defaults to True.
        """
        self.name = dataname
        self.fullname = f"{dataname}_{s_col}"
        self.csv_path = csv_path
        self.s_col = s_col
        self.y_col = y_col
        self.concat_train_val = concat_train_val
        try:
            df = pd.read_csv(csv_path)
            self.df_raw = df.copy()
        except Exception as e:
            raise f"Error encountered while reading csv file: {e}"
        assert s_col in df.columns, f"Sensitive attribute {s_col} not in df.columns"
        assert y_col in df.columns, f"Response attribute {y_col} not in df.columns"
        assert set(df[s_col].unique()) == {0, 1}, f"Sensitive col {s_col} is not binary"
        assert set(df[y_col].unique()) == {0, 1}, f"Response col {y_col} is not binary"

        if normalize:
            norm_cols = [col for col in df.columns if col not in [s_col, y_col]]
            df[norm_cols] = MinMaxScaler().fit_transform(df[norm_cols])
        
        # move the sensitive to the first column
        df = df[[s_col] + [col for col in df.columns if col != s_col]]

        # guarantee group 0 is the protected (disadvantaged) group
        s_uniques = np.unique(df[s_col]).astype(int)
        pos_ratio = {}
        for su in [0, 1]:
            pos_ratio[su] = df[y_col][df[s_col] == su].mean()
        if pos_ratio[0] > pos_ratio[1]:
            df[s_col] = df[s_col].map({0: 1, 1: 0})
        
        self.df = df

        # split & save data
        self.update_split(
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            x_with_s=x_with_s,
        )

    @property
    def data(self):
        """
        Get the data split into training, validation, and test sets.

        Returns:
            tuple: 3 tuples tuple containing the training, validation, and test sets.

        Example:
            (X_train, y_train, s_train), (X_val, y_val, s_val), (X_test, y_test, s_test) = dataset.data
        """
        split = self.split
        return split["train"], split["val"], split["test"]

    @property
    def info(self) -> str:
        """
        Get information about the FairDataset object.

        Returns:
            str: Information about the dataset, including name, shape, path,
                 sensitive and response attribute names, and split information.
        """
        return (
            f"Dataset    : {self.name} {self.df.shape} load from {self.csv_path}\n"
            f"Sens/Res   : {self.s_col}/{self.y_col}\n"
            f"Split      : {self.split_info}"
        )

    def describe(self) -> None:
        """
        Print descriptive information about the FairDataset object.
        """
        print(self.info)
        describe_data(self.split)

    def brief(self) -> None:
        """
        Print brief information about the FairDataset object.
        """
        print(self.info)
        brief_data(self.split)

    @staticmethod
    def get_subgroup_kfold_stratify_split_indexs(
        df, s_col, y_col,
        n_splits=5, n_train=3, n_val=1, n_test=1, 
        random_state=None
    ):
        from sklearn.model_selection import StratifiedKFold

        assert n_train + n_val + n_test == n_splits, f"n_train + n_val + n_test must equal to n_splits"
        # guarantee s_col and y_col are binary and only contains 0 and 1
        assert set(df[s_col].unique()) == {0, 1}, f"{s_col} must be binary"
        assert set(df[y_col].unique()) == {0, 1}, f"{y_col} must be binary"
        # guarantee s_col is the first column
        assert df.columns[0] == s_col, f"{s_col} must be the first column"

        stratify = df[y_col].values * len(df[s_col].unique()) + df[s_col].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = skf.split(df, stratify)
        index_splits = []
        for _, test_index in splits:
            index_splits.append(test_index)
            # print(df.iloc[test_index][[s_col, y_col]].value_counts())
        return index_splits

    def get_subgroup_split(
        self, i_split, random_state, x_with_s=True,
        n_splits=5, n_train=3, n_val=1, n_test=1, 
    ):
        df, s_col, y_col = self.df, self.s_col, self.y_col

        assert n_train + n_val + n_test == n_splits, f"n_train + n_val + n_test must equal to n_splits"
        # guarantee s_col and y_col are binary and only contains 0 and 1
        assert set(df[s_col].unique()) == {0, 1}, f"{s_col} must be binary"
        assert set(df[y_col].unique()) == {0, 1}, f"{y_col} must be binary"
        # guarantee s_col is the first column
        assert df.columns[0] == s_col, f"{s_col} must be the first column of the DataFrame"
        # guarantee i_split is less than n_splits
        assert i_split < n_splits, f"i_split (got {i_split}) must be less than n_splits (got {n_splits})"

        index_splits = self.get_subgroup_kfold_stratify_split_indexs(
            df, s_col, y_col,
            n_splits=n_splits, n_train=n_train, n_val=n_val, n_test=n_test, 
            random_state=random_state
        )
        # test be the index_splits[i_split], val be the index_splits[(i_split-1) % n_split], train be the rest
        i_test = i_split
        i_val = (i_split-1) % n_splits
        i_train = [i for i in range(n_splits) if i not in [i_test, i_val]]

        idx_test = index_splits[i_test]
        idx_val = index_splits[i_val]
        idx_train = []
        for i in i_train:
            idx_train.append(index_splits[i])
        idx_train = np.concatenate(idx_train)

        # get feature names based on x_with_s
        if x_with_s:
            X_drop_cols = [y_col]
        else:
            X_drop_cols = [y_col, s_col]
        feat_names = df.drop(columns=X_drop_cols).columns.values

        X_train = df.iloc[idx_train][feat_names].values
        y_train = df.iloc[idx_train][y_col].values
        s_train = df.iloc[idx_train][s_col].values
        X_val = df.iloc[idx_val][feat_names].values
        y_val = df.iloc[idx_val][y_col].values
        s_val = df.iloc[idx_val][s_col].values
        X_test = df.iloc[idx_test][feat_names].values
        y_test = df.iloc[idx_test][y_col].values
        s_test = df.iloc[idx_test][s_col].values

        return (
            (X_train, y_train, s_train),
            (X_val, y_val, s_val),
            (X_test, y_test, s_test),
            (idx_train, idx_val, idx_test)
        )

    def update_split(
        self,
        train_size: float = None,
        val_size: float = None,
        test_size: float = None,
        random_state: int = None,
        x_with_s: bool = None,
    ) -> tuple:
        """
        Update the data split based on provided or default split sizes.

        Parameters:
            train_size (float, optional): Proportion of the dataset to include in the training split.
            val_size (float, optional): Proportion of the dataset to include in the validation split.
            test_size (float, optional): Proportion of the dataset to include in the test split.
            random_state (int, optional): Seed used by the random number generator for reproducibility.
            x_with_s (bool, optional): Whether the input data should include the sensitive attribute.

        Returns:
            tuple: A tuple containing the training, validation, and test sets after the update.
        """
        # split data
        (
            (X_train, y_train, s_train),
            (X_val, y_val, s_val),
            (X_test, y_test, s_test),
            (idx_train, idx_val, idx_test)
        ) = self.train_val_test_split(
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            x_with_s=x_with_s,
        )
        # update attributess
        self.train_size = train_size if train_size is not None else self.train_size
        self.val_size = val_size if val_size is not None else self.val_size
        self.test_size = test_size if test_size is not None else self.test_size
        self.random_state = (
            random_state if random_state is not None else self.random_state
        )
        self.x_with_s = x_with_s if x_with_s is not None else self.x_with_s

        if self.concat_train_val:
            X_train = np.concatenate([X_train, X_val])
            y_train = np.concatenate([y_train, y_val])
            s_train = np.concatenate([s_train, s_val])
            # update attributes
            self.split_info = (
                f"train/test = {self.train_size+self.val_size}/{self.test_size}, "
                f"random_state = {self.random_state}, x_with_s = {self.x_with_s}"
            )
            self.split = {
                "train": (X_train, y_train, s_train),
                "test": (X_test, y_test, s_test),
            }
            self.split_idx = {
                "train": np.concatenate([idx_train, idx_val]),
                "test": idx_test,
            }
            return_data = (
                (X_train, y_train, s_train),
                (X_test, y_test, s_test),
            )
        else:
            self.split_info = (
                f"train/val/test = {self.train_size}/{self.val_size}/{self.test_size}, "
                f"random_state = {self.random_state}, x_with_s = {self.x_with_s}"
            )
            self.split = {
                "train": (X_train, y_train, s_train),
                "val": (X_val, y_val, s_val),
                "test": (X_test, y_test, s_test),
            }
            self.split_idx = {
                "train": idx_train,
                "val": idx_val,
                "test": idx_test,
            }
            return_data = (
                (X_train, y_train, s_train),
                (X_val, y_val, s_val),
                (X_test, y_test, s_test),
            )

        return return_data

    def train_val_test_split(
        self,
        train_size: float = None,
        val_size: float = None,
        test_size: float = None,
        random_state: int = None,
        x_with_s: bool = None,
    ) -> tuple:
        """
        Split the dataset into training, validation, and test sets.

        Parameters:
            train_size (float, optional): Proportion of the dataset to include in the training split.
            val_size (float, optional): Proportion of the dataset to include in the validation split.
            test_size (float, optional): Proportion of the dataset to include in the test split.
            random_state (int, optional): Seed used by the random number generator for reproducibility.
            x_with_s (bool, optional): Whether the input data should include the sensitive attribute.

        Returns:
            tuple: A tuple containing the training, validation, and test sets.
        """
        train_size = self.train_size if train_size is None else train_size
        val_size = self.val_size if val_size is None else val_size
        test_size = self.test_size if test_size is None else test_size
        random_state = self.random_state if random_state is None else random_state
        x_with_s = self.x_with_s if x_with_s is None else x_with_s
        assert train_size + val_size + test_size == 1, f"Split sizes do not add up to 1"
        assert (
            "stratify" not in self.df.columns
        ), f'Column "stratify" already exists in {self.name} dataset'

        s_col, y_col = self.s_col, self.y_col
        df = self.df.copy()
        df["stratify"] = df[y_col].values * len(df[s_col].unique()) + df[s_col].values
        df_train_val, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df["stratify"]
        )
        df_train, df_val = train_test_split(
            df_train_val,
            test_size=val_size / (train_size + val_size),
            random_state=random_state,
            stratify=df_train_val["stratify"],
        )
        idx_train, idx_val, idx_test = (
            df_train.index.values,
            df_val.index.values,
            df_test.index.values,
        )
        if x_with_s:
            X_drop_cols = [y_col, "stratify"]
        else:
            X_drop_cols = [y_col, s_col, "stratify"]
        
        self.feature_names = df_train.drop(columns=X_drop_cols).columns.values

        X_train = df_train.drop(columns=X_drop_cols).values
        X_val = df_val.drop(columns=X_drop_cols).values
        X_test = df_test.drop(columns=X_drop_cols).values
        y_train, y_val, y_test = (
            df_train[y_col].values,
            df_val[y_col].values,
            df_test[y_col].values,
        )
        s_train, s_val, s_test = (
            df_train[s_col].values,
            df_val[s_col].values,
            df_test[s_col].values,
        )

        return (
            (X_train, y_train, s_train),
            (X_val, y_val, s_val),
            (X_test, y_test, s_test),
            (idx_train, idx_val, idx_test)
        )


def train_val_test_split_dataframe(
    df, train_size, val_size, test_size, random_state, stratify="stratify"
):
    """
    Split a DataFrame into training, validation, and test sets.

    Parameters:
        df (pandas DataFrame): The DataFrame to be split.
        train_size (float): Proportion of the dataset to include in the training split.
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator for reproducibility.
        stratify (str, optional): Column name used for stratified sampling. Defaults to 'stratify'.

    Returns:
        tuple: A tuple containing the training, validation, and test DataFrames.
    """
    assert train_size + val_size + test_size == 1
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[stratify]
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (train_size + val_size),
        random_state=random_state,
        stratify=train_val[stratify],
    )
    return train, val, test


def describe_data(data: dict):
    """
    Print descriptive information about the data in a dictionary of datasets.

    Parameters:
        data (dict): A dictionary containing dataset names as keys and tuple of (X, y, s) as values.
    """

    def print_aligned_table(table_data, column_names, row_names):
        # Add the row names as the first element in each row of table_data
        table_data_with_row_names = [
            [row_name] + row_data for row_name, row_data in zip(row_names, table_data)
        ]
        # Add an empty cell at the top-left corner to account for the row names column header
        column_names_with_empty_cell = [""] + column_names
        # Print the table with tabulate
        print(
            tabulate(
                table_data_with_row_names,
                headers=column_names_with_empty_cell,
                tablefmt="grid",
            )
        )

    for data_name, (X, y, s) in data.items():
        print(f"{data_name} data [#samples {X.shape[0]} #features {X.shape[1]}]:")
        s_uniques = np.unique(s).astype(int)
        row_names = [f"s={s}" for s in s_uniques]
        y_uniques = np.unique(y).astype(int)
        col_names = [f"y={y}" for y in y_uniques] + ["pos_rate"]
        des_df = []
        for su in s_uniques:
            des_row = []
            for yu in y_uniques:
                mask = (s == su) & (y == yu)
                des_row.append(mask.sum())
            des_row.append(y[s == su].mean().round(4))
            des_df.append(des_row)
        print_aligned_table(des_df, col_names, row_names)
    print()


def brief_data(data: dict):
    """
    Print brief information about the data in a dictionary of datasets.

    Parameters:
        data (dict): A dictionary containing dataset names as keys and tuple of (X, y, s) as values.
    """
    for data_name, (X, y, s) in data.items():
        print(f"{data_name:<10s} | ", end="")
        s_uniques = np.unique(s).astype(int)
        size = {}
        pos_ratio = {}
        for su in s_uniques:
            su_mask = s == su
            size[su] = su_mask.sum()
            y_su = y[su_mask]
            pos_ratio[su] = ((y_su == 1).sum() / su_mask.sum()).round(4)
        print(f"size {size} | grp_pos_ratio: {pos_ratio:}")
    print()


def split_save_data(
    df,
    columns,
    dir_path,
    s_col,
    y_col,
    train_size,
    val_size,
    test_size,
    random_state,
    normalize=True,
):
    """
    Split the DataFrame into training, validation, and test sets, and save them as CSV files.

    Parameters:
        df (pandas DataFrame): The DataFrame to be split and saved.
        columns (list): List of column names to be included in the saved CSV files.
        dir_path (str): Directory path where the CSV files will be saved.
        s_col (str): Name of the sensitive attribute column in the DataFrame.
        y_col (str): Name of the response attribute column in the DataFrame.
        train_size (float): Proportion of the dataset to include in the training split.
        val_size (float): Proportion of the dataset to include in the validation split.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator for reproducibility.
        normalize (bool, optional): Whether to normalize the non-sensitive attribute columns. Defaults to True.
    """
    if normalize:
        norm_cols = [col for col in df.columns if col not in [s_col, y_col]]
        df[norm_cols] = MinMaxScaler().fit_transform(df[norm_cols])
    df_ = df.copy()
    df_["stratify"] = df[y_col].values * len(df[s_col].unique()) + df[s_col].values
    train_df, val_df, test_df = train_val_test_split_dataframe(
        df_, train_size, val_size, test_size, random_state
    )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    train_df[columns].to_csv(dir_path + "/train.csv", index=False)
    val_df[columns].to_csv(dir_path + "/val.csv", index=False)
    test_df[columns].to_csv(dir_path + "/test.csv", index=False)
    print(
        f"Save data to {dir_path} successfully!\n"
        f"Columns:    {columns}\n"
        f"Train size: {len(train_df)}\n"
        f"Val size:   {len(val_df)}\n"
        f"Test size:  {len(test_df)}\n"
    )
