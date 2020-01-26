import scipy
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
    config['iterations'] = get_iterations(exp['dir'])
    config['last_iteration'] = config['iterations'][-1]

    return config

def get_iterations(logdir):
    iterations = []
    for i in logdir.iterdir():
        f = i.name
        if not f.endswith('-game0_elite.pkl'):
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

def plot_pareto(exp):
    rewards = exp['rewards']
    cfg = exp['cfg']
    if is_singletask(cfg):
        return None
    sns.set(rc={'figure.figsize': (20, 10)})
    df = rewards.copy()
    cols = [exp['type'] + '-' + cfg['games'][0] + '_rewards', exp['type'] + '-' + cfg['games'][1] + '_rewards']
    df = df.loc[:, cols]
    myplot = sns.scatterplot(x=df[cols[0]], y=df[cols[1]], data=df)
    myplot.set(title="Pareto")
    return myplot.get_figure()

def compute_pareto(rewards_game0, rewards_game1):
    costs = np.transpose(np.array([rewards_game0, rewards_game1]))
    points_PF_x, points_PF_y = metrics.f_true_PF_points(costs)
    return points_PF_x, points_PF_y

def compute_hv_value(rewards_game0, rewards_game1):
    points_PF_x, points_PF_y = compute_pareto(rewards_game0, rewards_game1)
    HV_value = metrics.f_computeHypervolume(np.array([points_PF_x, points_PF_y]))
    return HV_value

def get_all_rewards_from_experiments(experiments):
    ST_z = experiments['ST-zaxxon']['rewards']
    ST_r = experiments['ST-riverraid']['rewards']
    MT = experiments['MT']['rewards']
    return ST_z.join(ST_r).join(MT)

def get_rewards(exp):
    logdir = exp['dir']
    last_iteration = exp['cfg']['iterations']
    iteration_limit = 201

    if str(exp['dir']).endswith('MT-zaxxon-riverraid-50000'):
        iteration_limit = 801

    if str(exp['dir']).endswith("evaluate_riverraid_using_zaxxon_model"):
        last_iteration = 200

    rewards_df = pd.DataFrame(columns=['game0_rewards', 'game1_rewards', 'game0_elite', 'game1_elite', 'iteration'])
    for i in range(0, last_iteration):
        df = {
            'game0_rewards': [np.mean(get_iter_log(logdir, i, 'game0_rewards'))],
            'game0_elite': [np.mean(get_iter_log(logdir, i, 'game0_elite'))],
        }
        if is_singletask(exp['cfg']):
            df['game1_rewards'] = df['game0_rewards']
            df['game1_elite'] = df['game0_elite']
        else:
            df['game1_rewards'] = [np.mean(get_iter_log(logdir, i, 'game1_rewards'))]
            df['game1_elite'] = [np.mean(get_iter_log(logdir, i, 'game1_elite'))]

        df['iteration'] = [i]
        rdf = pd.DataFrame.from_dict(df)
        rewards_df = pd.concat([rewards_df, rdf], sort=True)

    cfg = exp['cfg']
    if is_singletask(cfg):
        game_callist = ['game0_rewards', 'game0_elite', 'iteration']
        rewards_df = rewards_df.loc[:, game_callist]
        rewards_df.columns = [exp['type'] + '-' + cfg['games'][0] + '_rewards', exp['type'] + '-' + cfg['games'][0] + '_elite', 'iteration']
    else:
        rewards_df.columns = [
            exp['type'] + '-' + cfg['games'][0] + '_rewards', exp['type'] + '-' + cfg['games'][0] + '_elite',
            exp['type'] + '-' + cfg['games'][1] + '_rewards', exp['type'] + '-' + cfg['games'][1] + '_elite',
            'iteration'
        ]

    rewawrds_df = rewards_df.loc[0:last_iteration, :]
 
    ### EXPLAIN WHY WE SET RESULTS OF 200 TO RESULTS OF 199
    if str(exp['dir']).endswith("evaluate_riverraid_using_zaxxon_model") or str(exp['dir']).endswith("evaluate_zaxxon_using_riverraid_model"):
        addrow = rewards_df.tail(1)
        addrow.iteration = max(addrow.iteration) + 1
        rewards_df = rewards_df.append(addrow)
        addrow = rewards_df.tail(1)
        addrow.iteration = max(addrow.iteration) + 1
        rewards_df = rewards_df.append(addrow)
    rewards_df = rewards_df[rewards_df.iteration < last_iteration]

    return rewards_df.set_index('iteration')

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

def get_bins(min_val, max_val, bin_size):
    return [i for i in range(int(min_val), int(max_val) + bin_size, bin_size)]

def get_dkl_data(cfg, game_idx, iteration, bin_size, epsilon, elite_or_rewards='rewards', min_reward=0, max_reward=10000):
    if elite_or_rewards not in ['elite', 'rewards']:
        raise RuntimeError("elit_or_rewards must be elite or rewards")

    rewards = get_iter_log(cfg['dir'], iteration, 'game' + str(game_idx) + '_' + elite_or_rewards)
    rewards = np.array(list(map(lambda x: np.mean(x), rewards)))
    bins = get_bins(min_reward, max_reward, bin_size)
    rewards_per_bin = np.histogram(rewards, bins)[0]
    original_proportion_rewards_per_bin = rewards_per_bin / rewards.shape[0]
    epsilon_proportion_rewards_per_bin = original_proportion_rewards_per_bin + epsilon
    proportion_rewards_per_bin = epsilon_proportion_rewards_per_bin / sum(epsilon_proportion_rewards_per_bin)
    return proportion_rewards_per_bin

def compute_dkl(cfg, game_idx, iteration, bin_size, epsilon, iteration_lag=1, elite_or_rewards='rewards'):
    offspring = get_dkl_data(cfg, game_idx, int(iteration), bin_size, epsilon, elite_or_rewards)
    parent_iteration = int(iteration) - iteration_lag
    if parent_iteration < 0:
        parent_iteration = 0
    parent = get_dkl_data(cfg, game_idx, parent_iteration, bin_size, epsilon, elite_or_rewards)
    return scipy.stats.entropy(offspring, parent)


def get_game_rewards(cfg, iterations=200):
    res = []
    for iteration in range(0, iterations):
        game0_rewards = np.array(list(map(lambda x: np.mean(x), get_iter_log(cfg['dir'], iteration, 'game0_rewards'))))
        if 'dir_competitor' in cfg.keys():
            game1_rewards = np.array(list(map(lambda x: np.mean(x), get_iter_log(cfg['dir_competitor'], iteration, 'game0_rewards'))))
        else:
            game1_rewards = np.array(list(map(lambda x: np.mean(x), get_iter_log(cfg['dir'], iteration, 'game1_rewards'))))
        res.append(pd.DataFrame({'iteration': iteration, 'game0_rewards': game0_rewards, 'game1_rewards': game1_rewards}))
    return pd.concat(res)

def get_hypervolume_data(exp, iterations=200):
    hvs = []
    mean_game0_rewards = []
    mean_game1_rewards = []
    for iteration in range(0, iterations):
        game0_rewards = get_iter_log(exp['dir'], iteration, 'game0_rewards');
        other_game_index = 'game0_rewards' if exp['cfg']['games'][0] == exp['cfg']['games'][1] else 'game1_rewards'
        game1_rewards = get_iter_log(exp['dir'], iteration, other_game_index);
        game0_rewards = np.array(list(map(lambda x: np.mean(x), game0_rewards)))
        game1_rewards = np.array(list(map(lambda x: np.mean(x), game1_rewards)))
        hv_iteration = compute_hv_value(game0_rewards, game1_rewards)
        hvs.append(hv_iteration)
        mean_game0_rewards.append(game0_rewards)
        mean_game1_rewards.append(game1_rewards)
    df = pd.DataFrame.from_dict({'hv': hvs, 'mean_game0_rewards': mean_game0_rewards, 'mean_game1_rewards': mean_game1_rewards})
    df['Epoch'] = list(range(0, iterations))
    return df.set_index('Epoch')

def get_paretos(cfg, iterations=200):
    game_rewards = get_game_rewards(cfg, iterations)
    for index, row in game_rewards.iterrows():
        print(row['c1'], row['c2'])
    paretos = []
    mean_game0_rewards = []
    mean_game1_rewards = []
    for iteration in range(0, iterations):
        game0_rewards = get_iter_log(cfg['dir'], iteration, 'game0_rewards');
        other_game_index = 'game0_rewards' if cfg['cfg']['games'][0] == cfg['cfg']['games'][1] else 'game1_rewards'
        game1_rewards = get_iter_log(cfg['dir'], iteration, other_game_index);
        game0_rewards = np.array(list(map(lambda x: np.mean(x), game0_rewards)))
        game1_rewards = np.array(list(map(lambda x: np.mean(x), game1_rewards)))
        hv_iteration = compute_hv_value(game0_rewards, game1_rewards)
        hvs.append(hv_iteration)
        mean_game0_rewards.append(game0_rewards)
        mean_game1_rewards.append(game1_rewards)
    df = pd.DataFrame.from_dict({'hv': hvs, 'mean_game0_rewards': mean_game0_rewards, 'mean_game1_rewards': mean_game1_rewards})
    df['iteration'] = list(range(0, iterations))
    return df.set_index('iteration')
