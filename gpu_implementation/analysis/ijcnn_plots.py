import seaborn as sns
import pandas as pd
import numpy as np
import ijcnn_getters

def rewards(config, rewards):
    ### plot: Game0 and Game1 score and elite score
    sns.set(rc={'figure.figsize':(20, 10)})
    rewards.set_index('iteration')
    p = rewards.set_index('iteration').plot()
    p.set(ylabel='Score')
    p.set(xlabel='Epoch')

def frames_vs_score(config, reward_eplen, game_idx):
    game_name = ijcnn_getters.get_game_name(config, game_idx)
    sns.set(rc={'figure.figsize':(20, 10)})
    df = reward_eplen.copy()
    df['Epoch'] = df['iteration']
    p = sns.scatterplot(x='eplen', y='reward', hue='Epoch', data=df.query('game == ' + str(game_idx)))
    p.set(title=game_name + ' frames vs. score')
    p.set(ylabel='Number of frames')
    p.set(xlabel='Score')
