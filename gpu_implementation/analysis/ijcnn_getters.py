import re
import json
import pandas as pd
import numpy as np
import pickle
import utils

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


def get_config_body(logdir):
    with open(str(logdir / "log.txt"), "r") as f:
        data = f.read()
    config_data = data.find(" Logging to: ")
    return data[0:config_data]

def get_config(logdir):
    config = json.loads(get_config_body(logdir)[23:-23])
    config['logdir'] = logdir
    config['dir'] = logdir
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

def get_rewards(exp):
    logdir, iterations = exp['dir'], exp['cfg']['iterations']
    rewards = pd.DataFrame(columns=['game0_rewards', 'game1_rewards', 'game0_parent_mean', 'game1_parent_mean', 'iteration'])
    for i in iterations:
        df = {
            'game0_rewards': [np.mean(get_iter_log(logdir, i, 'game0_rewards'))],
            'game0_parent_mean': [np.mean(get_iter_log(logdir, i, 'game0_elite'))],
        }
        if exp['type'] == 'MT':
            df['game1_rewards'] = [np.mean(get_iter_log(logdir, i, 'game1_rewards'))]
            df['game1_parent_mean'] = [np.mean(get_iter_log(logdir, i, 'game1_elite'))]

        df['iteration'] = [i]
        rdf = pd.DataFrame.from_dict(df)
        rewards = pd.concat([rewards, rdf], sort=True)
    rewards = df_change_game_names(exp['cfg'], rewards)
    return rewards

def get_rewards_eplen(exp):
    logdir, last_iteration = exp['dir'], exp['cfg']['last_iteration']
    rewards_eplen = pd.DataFrame(columns=['eplen', 'reward', 'iteration'])
    for i in range(last_iteration):
        df = {
            'eplen': np.array(get_iter_log(logdir, i, 'game0_episode_lengths')).flatten(),
            'reward': np.array(get_iter_log(logdir, i, 'game0_rewards')).flatten()
        }
        edf0 = pd.DataFrame.from_dict(df)
        edf0['game'] = [0] * edf0['eplen'].shape[0]
        edf0['iteration'] = [i] * edf0['eplen'].shape[0]

        if exp['type'] == 'MT':
            df = {
                'eplen': np.array(get_iter_log(logdir, i, 'game1_episode_lengths')).flatten(),
                'reward': np.array(get_iter_log(logdir, i, 'game1_rewards')).flatten()
            }
            edf1 = pd.DataFrame.from_dict(df)
            edf1['game'] = [1] * edf1['eplen'].shape[0]
            edf1['iteration'] = [float(i)] * edf1['eplen'].shape[0]

        if exp['type'] == 'MT':
            rewards_eplen = pd.concat([rewards_eplen, edf0, edf1], sort=True)
        
        if exp['type'] == 'ST':
            rewards_eplen = pd.concat([rewards_eplen, edf0], sort=True)
            
    return rewards_eplen

def get_hypervolume(experiments, iterations):
    df_dict = {}
    for exp_name, exp in experiments.items():
        df_dict[exp_name] = utils.get_hypervolume_data(exp, iterations)['hv']
    hv_df = pd.DataFrame.from_dict(df_dict)
    return hv_df


# thresholds from https://arxiv.org/pdf/1703.03864.pdf
def get_outerperformer_data(mt_exp, iterations, game0_threshold=6380, game1_threshold=5009):
    threshold_results = []

    for i in range(0, iterations):
        g0r = utils.get_iter_log(mt_exp['dir'], i, 'game0_rewards')
        g1r = utils.get_iter_log(mt_exp['dir'], i, 'game1_rewards')
        g0r_flat = [item for sublist in g0r for item in sublist]
        g1r_flat = [item for sublist in g1r for item in sublist]

        for reward_index in range(0, len(g0r_flat)):
            reward_0 = g0r_flat[reward_index]
            reward_1 = g1r_flat[reward_index]
            if reward_0 > game0_threshold and reward_1 > game1_threshold:
                threshold_results.append({'iteration': i, 'offspring_index': reward_index, 'zaxxon': reward_0, 'riverraid': reward_1})
    exceeded_threshold = pd.DataFrame(threshold_results)
    df_zeros = pd.DataFrame([{'breached': 0} for i in range(0, iterations)])
    df_breached = exceeded_threshold.groupby('iteration').count()
    df_breached = df_breached[['offspring_index']]
    df = pd.merge(df_zeros, df_breached, how='left', left_index=True, right_index=True)
    df_breached = pd.DataFrame({'threshold_breached': (df['offspring_index'] / (mt_exp['cfg']['population_size']/2) * 100)})
    df = df_breached.fillna(0)
    df['epoch'] = df.index
    return df


def get_raw_offspring(exp, game_idx, iterations=200):
    logdir = exp['dir']
    orig_column_name = 'game' + str(game_idx) + '_rewards'
    column_name = 'game' + str(game_idx) + '_offspring'
    rewards = pd.DataFrame(columns=[column_name, 'epoch'])
    for i in range(0, iterations):
        df = {
            column_name: get_iter_log(logdir, i, orig_column_name)
        }
        df['epoch'] = i
        rdf = pd.DataFrame.from_dict(df)
        rewards = pd.concat([rewards, rdf], sort=True)
    rewards = df_change_game_names(exp['cfg'], rewards)
    return rewards

def get_raw_parent(exp, game_idx, iterations=200):
    logdir = exp['dir']
    orig_column_name = 'game' + str(game_idx) + '_elite'
    column_name = 'game' + str(game_idx) + '_parent'
    rewards = pd.DataFrame(columns=[column_name, 'epoch'])
    for i in range(0, iterations):
        df = {
            column_name: get_iter_log(logdir, i, orig_column_name)
        }
        df['epoch'] = i
        rdf = pd.DataFrame.from_dict(df)
        rewards = pd.concat([rewards, rdf], sort=True)
    rewards = df_change_game_names(exp['cfg'], rewards)
    return rewards
