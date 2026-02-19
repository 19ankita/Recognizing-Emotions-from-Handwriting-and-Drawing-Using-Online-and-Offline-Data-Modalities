import numpy as np

def train_test_split_ids(ids, test_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    ids = np.array(ids)
    rng.shuffle(ids)

    n_test = int(len(ids) * test_ratio)
    test_ids = ids[:n_test]
    train_ids = ids[n_test:]

    return train_ids, test_ids
