from typing import Callable
from hdfe import Groupby
import numpy as np
import pandas as pd
import time

"""
Generates the results in blog post
http://esantorella.com/2016/06/16/groupby/
"""

n_iters = 1000
n_decimals = 4


def print_results(pandas_1, pandas_100, groupby_1, groupby_100):
    print('time to compute group means once with Pandas: {0}'
          .format(round(pandas_1, n_decimals)))
    print('time to compute group means {0} times with Pandas: {1}'
          .format(n_iters, round(pandas_100, n_decimals)))
    print('time to compute group means once with Grouped: {0}'
          .format(round(groupby_1, n_decimals)))
    print('time to compute group means {0} times with Grouped: {1}'
          .format(n_iters, round(groupby_100, n_decimals)))
    print('Improvement', groupby_100 / pandas_100)
    return


def get_transform_comparisions(f: Callable, data: pd.DataFrame):
    start = time.perf_counter()
    pandas_answer = data.groupby('first category')['y'].transform(f)
    pandas_1 = time.perf_counter() - start

    start = time.perf_counter()
    grouped = data.groupby('first category')['y']
    for i in range(n_iters):
        grouped.transform(f)

    pandas_100 = time.perf_counter() - start

    # Compute group means using Grouped class
    start = time.perf_counter()
    y = data['y'].values
    first_category = data['first category'].values
    group_means = Groupby(first_category).apply(f, y)
    groupby_one = time.perf_counter() - start
    np.testing.assert_almost_equal(pandas_answer.values, group_means)

    start = time.perf_counter()
    grouped = Groupby(first_category)
    for _ in range(n_iters):
        grouped.apply(f, y)

    groupby_100 = time.perf_counter() - start
    return pandas_1, pandas_100, groupby_one, groupby_100


def get_apply_comparisions(f: Callable, data: pd.DataFrame):
    start = time.perf_counter()
    pandas_answer = data.groupby('first category')['y'].apply(f)
    pandas_1 = time.perf_counter() - start

    start = time.perf_counter()
    grouped = data.groupby('first category')['y']
    if f == np.mean:
        for i in range(n_iters):
            grouped.mean()
    else:
        for i in range(n_iters):
            grouped.apply(f)

    pandas_100 = time.perf_counter() - start

    # Compute group means using Grouped class
    start = time.perf_counter()
    first_category = data['first category'].values
    y = data['y'].values
    group_means = Groupby(first_category).apply(f, y, broadcast=False)
    groupby_one = time.perf_counter() - start

    np.testing.assert_almost_equal(pandas_answer.values, group_means)

    start = time.perf_counter()
    grouped = Groupby(first_category)
    for _ in range(n_iters):
        grouped.apply(f, y, broadcast=False)

    groupby_100 = time.perf_counter() - start
    return pandas_1, pandas_100, groupby_one, groupby_100


def f(x): return np.mean(x)


def make_result_df(df: pd.DataFrame):

    result_df = pd.DataFrame(columns=['Pandas', 'Groupby'],
                             index=pd.MultiIndex.from_product((['Apply', 'Transform'],
                                                               ['Cython', 'Python'])),
                             data=np.zeros((4, 2)))

    print('\nTransform, np.mean: With the np.mean function, Pandas uses Cython and does great')
    results = get_transform_comparisions(np.mean, df)
    print_results(*results)
    result_df.loc['Transform', :].loc['Cython', :] = [results[1], results[3]]

    print('\nTransform, user-defined: Without Cython, Pandas is terrible')
    results = get_transform_comparisions(f, df)
    print_results(*results)
    result_df.loc['Transform', :].loc['Python', :] = [results[1], results[3]]

    print('\nApply, np.mean: With the np.mean function, Pandas uses Cython and does great')
    results = get_apply_comparisions(np.mean, df)
    print_results(*results)
    result_df.loc['Apply', :].loc['Cython', :] = [results[1], results[3]]

    print('\nTransform, user-defined: Without Cython, Pandas is terrible')
    results = get_apply_comparisions(f, df)
    print_results(*results)
    result_df.loc['Apply', :].loc['Python', :] = [results[1], results[3]]

    result_df /= result_df.values[0, 0]
    return result_df.apply(lambda x: np.round(x, 1))


def main():
    # Compute group means using Pandas groupby
    np.random.seed(int('hi', 36))
    n_obs = 10**4
    n_categories = 10**2

    df = pd.DataFrame({'first category': np.random.choice(n_categories, n_obs),
                       'y': np.random.normal(0, 1, n_obs)})
    assert not Groupby(df['first category']).already_sorted
    result_table = make_result_df(df)
    print(result_table)

    # Try again when already sorted
    df.sort_values('first category', inplace=True)
    assert Groupby(df['first category']).already_sorted
    result_table = make_result_df(df)
    print(result_table)
    return


if __name__ == '__main__':
    main()
