import time
from typing import Callable, Generic, TypeVar

import numpy as np
import pandas as pd
import multiprocessing as mp
import ctypes
from functools import partial

_T = TypeVar("_T")

# Trick to avoid serialization when UNIX fork is used
class ForkRef(Generic[_T]):
    def __init__(self, obj: _T) -> None:
        self._ref = id(obj)

    def __call__(self) -> _T:
        return ctypes.cast(self._ref, ctypes.py_object).value


def apply_to_chunk(df_ref: ForkRef[pd.DataFrame], func: Callable, chunk: int, i: int) -> pd.Series:
    return df_ref().iloc[i * chunk : (i + 1) * chunk].apply(func, axis=1)


def parallel_apply(df: pd.DataFrame, func: Callable, chunk_size: int = 1024, n_jobs: int = -1) -> pd.Series:
    n_chunks = int(len(df) // chunk_size)

    # Using UNIX fork is crucial!
    # This does not work on Windows!
    ctx = mp.get_context("fork")

    process_pool = ctx.Pool(n_jobs)
    result_chunked = process_pool.map(
        partial(apply_to_chunk, ForkRef(df), func, chunk_size),
        range(0, n_chunks),
    )

    return pd.concat(result_chunked)


def compute_similarity_fn(row):
    if row["user_id_lhs"] == row["user_id_rhs"]:
        return -1

    x = set(row["item_id_lhs"])
    y = set(row["item_id_rhs"])

    return len(x & y) / (len(x) * len(y))


def user_similarity(interactions: pd.DataFrame) -> pd.DataFrame:
    interactions_grouped_by_user = interactions.groupby("user_id", as_index=False).agg(list)

    cross_rows = interactions_grouped_by_user.join(
        interactions_grouped_by_user,
        how="cross",
        lsuffix="_lhs",
        rsuffix="_rhs",
    )

    cross_rows["similarity"] = parallel_apply(cross_rows, compute_similarity_fn, n_jobs=4)

    cross_rows.sort_values(
        ["user_id_lhs", "similarity"],
        ascending=(True, False),
        inplace=True,
        ignore_index=True,
    )

    top_similar_users = (
        cross_rows[["user_id_lhs", "user_id_rhs", "similarity"]]
        .groupby("user_id_lhs")
        .head(10)
        .rename(columns={
            "user_id_lhs": "user_id",
            "user_id_rhs": "similar_user_id",
        })
    )

    return top_similar_users


if __name__ == "__main__":
    interactions = pd.read_csv("../data/ml-1m/interactions_2k.csv", dtype=np.int32)

    t0 = time.perf_counter()
    result = user_similarity(interactions)
    t1 = time.perf_counter()

    print(result)
    print()
    print(f"Execution time: {(t1 - t0):.4f} sec.")
