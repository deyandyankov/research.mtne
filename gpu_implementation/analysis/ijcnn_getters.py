import re
import json
import pandas as pd
import numpy as np
import pickle

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

def get_last_iteration(iterations):
    return iterations[-1]


def get_iter_log(logdir, iteration, pickle_file):
    filename = logdir / "{:04d}-{}.pkl".format(iteration, pickle_file)
    with open(str(filename), 'rb') as f:
        return pickle.load(f)

def get_iter_logs(logdir, iteration):
    df = {}
    loadfiles = ['state', 'offsprings',
                 'game0_elite', 'game0_rewards', 'game0_episode_lengths',
                 'game1_elite', 'game1_rewards', 'game1_episode_lengths'
                ]
    for loadfile in loadfiles:
        df[loadfile] = get_iter_log(logdir, iteration, loadfile)
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

def get_config_body(logdir):
    with open(str(logdir / "log.txt"), "r") as f:
        data = f.read()
    config_data = data.find(" Logging to: ")
    return data[0:config_data]

def get_config(logdir):
    config = json.loads(get_config_body(logdir)[23:-23])
    config['logdir'] = logdir
    config['iterations'] = get_iterations(logdir)
    config['last_iteration'] = get_last_iteration(config['iterations'])
    return config

def ucfirst(s):
    return s[0].upper() + s[1:]

def get_game_name(config, game_idx):
    return ucfirst(config['games'][game_idx])

def df_change_game_names(config, df):
    for col in df.columns:
        if 'game0' in col:
            df.rename(columns={col: col.replace('game0', get_game_name(config, 0))}, inplace=True)
        if 'game1' in col:
            df.rename(columns={col: col.replace('game1', get_game_name(config, 1))}, inplace=True)
    return df

def get_rewards(config):
    logdir, iterations = config['logdir'], config['iterations']
    rewards = pd.DataFrame(columns=['game0_rewards', 'game1_rewards', 'game0_parent_mean', 'game1_parent_mean', 'iteration'])
    for i in iterations:
        df = {
            'game0_rewards': [np.mean(get_iter_log(logdir, i, 'game0_rewards'))],
            'game1_rewards': [np.mean(get_iter_log(logdir, i, 'game1_rewards'))],
            'game0_parent_mean': [np.mean(get_iter_log(logdir, i, 'game0_elite'))],
            'game1_parent_mean': [np.mean(get_iter_log(logdir, i, 'game1_elite'))]
        }
        df['iteration'] = [i]
        rdf = pd.DataFrame.from_dict(df)
        rewards = pd.concat([rewards, rdf], sort=True)
    rewards = df_change_game_names(config, rewards)
    return rewards

def get_rewards_eplen(config):
    logdir, last_iteration = config['logdir'], config['last_iteration']
    rewards_eplen = pd.DataFrame(columns=['eplen', 'reward', 'iteration'])
    for i in range(last_iteration):
        df = {
            'eplen': np.array(get_iter_log(logdir, i, 'game0_episode_lengths')).flatten(),
            'reward': np.array(get_iter_log(logdir, i, 'game0_rewards')).flatten()
        }
        edf0 = pd.DataFrame.from_dict(df)
        edf0['game'] = [0] * edf0['eplen'].shape[0]
        edf0['iteration'] = [i] * edf0['eplen'].shape[0]

        df = {
            'eplen': np.array(get_iter_log(logdir, i, 'game1_episode_lengths')).flatten(),
            'reward': np.array(get_iter_log(logdir, i, 'game1_rewards')).flatten()
        }
        edf1 = pd.DataFrame.from_dict(df)
        edf1['game'] = [1] * edf1['eplen'].shape[0]
        edf1['iteration'] = [float(i)] * edf1['eplen'].shape[0]

        rewards_eplen = pd.concat([rewards_eplen, edf0, edf1], sort=True)
    return rewards_eplen
