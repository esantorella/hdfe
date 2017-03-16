from hdfe import Groupby
import numpy as np
import pandas as pd
import time


# Demo with small data
df = pd.DataFrame({'first category': [0, 1, 2, 0, 1, 1, 0], 
                   'y': np.arange(0, .7, .1)})


# Create data
group_means = df.groupby('first category')['y'].apply(np.mean)

def get_group_mean(x):
    return pd.Series(np.full(len(x), np.mean(x)),
                     x.index)
                     
df['mean'] = df.groupby('first category')['y'].apply(get_group_mean)

# Compute group means using Pandas groupby
np.random.seed(int('hi', 36))
n_iters = 100
n_decimals = 4
n_obs = 10**4
n_categories = 10**3

first_category = np.random.choice(n_categories, n_obs)
y = np.random.normal(0, 1, n_obs)

df = pd.DataFrame({'first category': first_category,
                   'y': y})
                     
start = time.clock()
pandas_answer = df.groupby('first category')['y'].apply(get_group_mean)
print('time to compute group means once with Pandas: {0}'\
      .format(round(time.clock() - start, n_decimals)))

start = time.clock()
grouped = df.groupby('first category')['y']
for i in range(n_iters):
    grouped.apply(get_group_mean)
print('time to compute group means {0} times with Pandas: {1}'\
      .format(n_iters, round(time.clock() - start, n_decimals)))

# Compute group means using Grouped class
start = time.clock()
group_means = Groupby(first_category).apply(np.mean, y)
print('time to compute group means once with Grouped: {0}'\
      .format(round(time.clock() - start, n_decimals)))

start = time.clock()
grouped = Groupby(first_category)
for i in range(n_iters):
    grouped.apply(np.mean, y)
    
print('time to compute group means {0} times with Grouped: {1}'\
      .format(n_iters, round(time.clock() - start, n_decimals)))

print(np.hstack(pandas_answer.values) - group_means)
