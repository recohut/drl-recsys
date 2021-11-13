from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd
import numpy as np
import ast
from typing import List, Union, Dict, Tuple
from multiprocessing.pool import Pool
import os
import scipy


def parallel_literal_eval(
    series: Union[pd.Series, np.ndarray], pool: Pool = None, use_tqdm: bool = True
) -> list:
    if pool:
        return _parallel_literal_eval(series, pool, use_tqdm)
    else:
        with Pool(os.cpu_count()) as p:
            return _parallel_literal_eval(series, p, use_tqdm)


def _parallel_literal_eval(
    series: Union[pd.Series, np.ndarray], pool: Pool, use_tqdm: bool = True
) -> list:

    if use_tqdm:
        return list(tqdm(pool.map(literal_eval_if_str, series), total=len(series)))
    else:
        return pool.map(literal_eval_if_str, series)


def literal_eval_if_str(element):
    if isinstance(element, str):
        return ast.literal_eval(element)
    return element


def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    data = data[~np.isnan(data)]
    a = 1.0 * data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h
