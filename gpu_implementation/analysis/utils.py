import pickle
import re
import pandas as pd
import numpy as np
import json
import seaborn as sns
import metrics
from pathlib import Path

def is_singletask(cfg):
    return cfg['games'][0] == cfg['games'][1]

def get_config(exp):
    log_filename = str(exp['dir'] / "log.txt")
    with open(log_filename, "r") as f:
        data = f.read()
    config_data = data.find(" Logging to: ")
    config_data = data[0:config_data]
    config_data = config_data[23:]
    config_data = config_data[0:len(config_data)-23]
    config = json.loads(config_data)
    config['iterations'] = max(get_iterations(exp['dir']))

    return config

def get_iterations(logdir):
    iterations = []
    for i in logdir.iterdir():
        f = i.name
        if not f.endswith('-game1_elite.pkl'):
            continue
        itr = re.sub('^0*', '', f.split('-')[0])
        if itr == '':
            itr = 0
        else:
            itr = int(itr)
            iterations.append(itr)
    iterations.sort()
    return iterations

def plot_rewards(exp):
    rewards = exp['rewards']
    cfg = exp['cfg']
    sns.set(rc={'figure.figsize':(20, 10)})
    df = rewards.copy()
    df.set_index('iteration')
    myplot = df.set_index('iteration').plot()
    myplot.set(title="Game score comparison")
    myplot.set(ylabel='Game score')
    return myplot.get_figure()

def plot_pareto(rewards, cfg):
    if is_singletask(cfg):
        raise RuntimeWarning("Cannot run pareto on a single task experiment")
    sns.set(rc={'figure.figsize': (20, 10)})
    df = rewards.copy()
    cols = [cfg['games'][0] + '_rewards', cfg['games'][1] + '_rewards']
    df = df.loc[:, cols]
    myplot = sns.scatterplot(x=df[cols[0]], y=df[cols[1]], data=df)
    myplot.set(title="Pareto")
    return myplot.get_figure()

def compute_hv_value(rewards, cfg):
    df = rewards.copy()
    cols = [cfg['games'][0] + '_rewards', cfg['games'][1] + '_rewards']
    df = df.loc[:, cols]
    x, y = df[cols[0]], df[cols[1]]
    costs = np.transpose(np.array([x, y]))
    points_PF_x, points_PF_y = metrics.f_true_PF_points(costs)
    HV_value = metrics.f_computeHypervolume(np.array([points_PF_x, points_PF_y]))
    return HV_value

def get_rewards(exp):
    logdir = exp['dir']
    last_iteration = exp['cfg']['iterations']
    rewards_df = pd.DataFrame(columns=['game0_rewards', 'game1_rewards', 'game0_elite', 'game1_elite', 'iteration'])
    for i in range(0, last_iteration):
        df = {
            'game0_rewards': [np.mean(get_iter_log(logdir, i, 'game0_rewards'))],
            'game1_rewards': [np.mean(get_iter_log(logdir, i, 'game1_rewards'))],
            'game0_elite': [np.mean(get_iter_log(logdir, i, 'game0_elite'))],
            'game1_elite': [np.mean(get_iter_log(logdir, i, 'game1_elite'))]
        }
        df['iteration'] = [i]
        rdf = pd.DataFrame.from_dict(df)
        rewards_df = pd.concat([rewards_df, rdf], sort=True)

    cfg = exp['cfg']
    if is_singletask(cfg):
        game_callist = ['game0_rewards', 'game0_elite', 'iteration']
        rewards_df = rewards_df.loc[:, game_callist]
        rewards_df.columns = [cfg['games'][0] + '_rewards', cfg['games'][0] + '_elite', 'iteration']
    else:
        rewards_df.columns = [
            cfg['games'][0] + '_rewards', cfg['games'][0] + '_elite',
            cfg['games'][1] + '_rewards', cfg['games'][1] + '_elite',
            'iteration'
        ]

    return rewards_df

def get_iter_log(logdir, iteration, pickle_file):
    filename = logdir / "{:04d}-{}.pkl".format(iteration, pickle_file)
    with open(str(filename), 'rb') as f:
        return pickle.load(f)

def get_iter_logs(iteration):
    df = {}
    loadfiles = ['state', 'offsprings',
                 'game0_elite', 'game0_rewards', 'game0_episode_lengths',
                 'game1_elite', 'game1_rewards', 'game1_episode_lengths'
                ]
    for loadfile in loadfiles:
        df[loadfile] = get_iter_log(iteration, loadfile)
    return df

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y
