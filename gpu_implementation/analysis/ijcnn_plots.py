import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ijcnn_getters

def save_plot(p, filename):
    p.get_figure().savefig('ijcnn_plots/{}.pdf'.format(filename))
    p.get_figure().savefig('ijcnn_plots/{}.eps'.format(filename))
    p.get_figure().savefig('ijcnn_plots/{}.png'.format(filename))

def rewards(exp):
    ### plot: Game0 and Game1 score and elite score
    config, rewards = exp['cfg'], exp['rewards']
    offspring = rewards.copy()
    sns.set(rc={'figure.figsize':(8, 6)})#||SDR:changed size||#(20, 10)})
    offspring.columns = ['Zaxxon_parent', 'Zaxxon_offspring', 'Riverraid_parent',
       'Riverraid_offspring', 'iteration']
    offspring.set_index('iteration')
    p = offspring.set_index('iteration').plot()
    p.set(ylabel='Score')
    p.set(xlabel='Epoch')
    p.set_xlim([0, 200])
    return p

def frames_vs_score(exp, game_idx, iterations=200):
    config, rewards_eplen = exp['cfg'], exp['rewards_eplen']
    game_name = ijcnn_getters.get_game_name(config, game_idx)
    sns.set(rc={'figure.figsize':(8, 5)})
    df = rewards_eplen.copy().query('iteration <= ' + str(iterations))
    df['Epoch'] = df['iteration']
    p = sns.scatterplot(x='eplen', y='reward', hue='Epoch', data=df.query('game == ' + str(game_idx)))
    #p = sns.heatmap(x='eplen', y='reward', vmin=0, vmax=1, data=df.query('game == ' + str(game_idx)))
    p.set(title=game_name + ' frames vs. score')
    p.set(ylabel='Score')
    p.set(xlabel='Number of frames')
    return p

def hypervolume(hv_df, iterations=200):
    sns.set(rc={'figure.figsize': (8, 5)})
    hv_df['Epoch'] = range(0, iterations)
    hv_df.set_index('Epoch')
    del hv_df['Epoch']
    sns.set_style('white')
    p = hv_df.plot()
    # p = hv_df.plot(color='black')
    p.set_title('Hypervolume of offspring')
    p.set_ylabel('Hypervolume')
    p.set_xlim([0, 200])
    return p

def outperformer(df, iterations, title):
    sns.set(rc={'figure.figsize': (8, 6)})
    sns.set_style('white')
    p = sns.lineplot(x='epoch', y='threshold_breached', data=df)
    p.set_title(title + "\n")
    p.set_xlim([0, iterations])
    p.set_ylabel('Percentage of Offspring')
    return p
    

def raw_score_threshold(df, yname, threshold, iterations=200):
    sns.set(rc={'figure.figsize':(20, 10)})
#     df = df.set_index('epoch')
    p = sns.scatterplot(x='epoch', y=yname, data=df)
    p.set(ylabel='Score')
    p.set(xlabel='Epoch')
    p.hlines(threshold, 0, iterations) # draw threshold across 0:iterations
    return p

def mmm(melted, threshold, title, iterations=200):
    sns.set(rc={'figure.figsize':(8, 5)})
    p = sns.lineplot(x='Epoch', y='Score', data=melted.query("variable in ['min', 'max']"))
    p.hlines(threshold, 0, iterations, linestyles='dashed') # draw threshold across 0:iterations
    p.set_xlim([0, iterations])
    p.set_title(title)
    return p