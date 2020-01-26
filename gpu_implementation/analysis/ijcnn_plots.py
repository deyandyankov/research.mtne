import seaborn as sns
import pandas as pd
import numpy as np
import ijcnn_getters

def rewards(exp):
    ### plot: Game0 and Game1 score and elite score
    config, rewards = exp['cfg'], exp['rewards']
    sns.set(rc={'figure.figsize':(20, 10)})
    rewards.set_index('iteration')
    p = rewards.set_index('iteration').plot()
    p.set(ylabel='Score')
    p.set(xlabel='Epoch')

def frames_vs_score(exp, game_idx):
    config, rewards_eplen = exp['cfg'], exp['rewards_eplen']
    game_name = ijcnn_getters.get_game_name(config, game_idx)
    sns.set(rc={'figure.figsize':(20, 10)})
    df = rewards_eplen.copy()
    df['Epoch'] = df['iteration']
    p = sns.scatterplot(x='eplen', y='reward', hue='Epoch', data=df.query('game == ' + str(game_idx)))
    p.set(title=game_name + ' frames vs. score')
    p.set(ylabel='Number of frames')
    p.set(xlabel='Score')

def hypervolume(hv_df, iterations=200):
    sns.set(rc={'figure.figsize': (5, 4)})
    hv_df['Epoch'] = range(0, iterations)
    hv_df.set_index('Epoch')
    del hv_df['Epoch']
    sns.set_style('white')
    p = hv_df.plot()
    # p = hv_df.plot(color='black')
    p.set_title('Hypervolume of offspring')
    p.set_ylabel('Hypervolume')
    p.set_xlim([0, 200])
