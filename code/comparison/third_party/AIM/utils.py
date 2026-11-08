import random
import numpy as np
try:
    from baselines import AdaFairClassifier  # noqa: F401
except ImportError:
    AdaFairClassifier = None  # 可选依赖，AIM 核心不使用
try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
except ImportError:
    demographic_parity_difference = None
    equalized_odds_difference = None
from sklearn.metrics import accuracy_score


def mask_to_idx(mask: np.ndarray):
    return np.where(mask)[0]


def idx_to_mask(indices: np.ndarray, size: int):
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return mask


def dict_info(d):
    info = ''
    for k, v in d.items():
        info += f'{k}: {v}\n'
    return info


def generate_random_seeds(n_seeds, master_seed):
    random.seed(master_seed)
    seeds = [random.randint(0, np.iinfo(np.uint32).max) for _ in range(n_seeds)]
    return seeds


def seed_generator(master_seed):
    """
    Generator function to produce random seeds based on a master seed.

    Parameters:
        master_seed (int): The master seed used to initialize the random number generator.

    Yields:
        int: A random seed generated using the master_seed.
    """
    rng = np.random.default_rng(master_seed)
    while True:
        yield rng.integers(np.iinfo(np.uint32).max)

def describe_data(X, y, s):
    """
    Print descriptive information about the data in a dictionary of datasets.

    Parameters:
        data (dict): A dictionary containing dataset names as keys and tuple of (X, y, s) as values.
    """

    def print_aligned_table(table_data, column_names, row_names):
        from tabulate import tabulate
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