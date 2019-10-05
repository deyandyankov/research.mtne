__copyright__ = """
Copyright (c) 2018 Uber Technologies, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import json
import time
import sys
import base64
import pickle
import tempfile
import os
import signal
from shutil import copyfile
import tensorflow as tf
import numpy as np
from neuroevolution.tf_util import get_available_gpus, WorkerSession
from neuroevolution.helper import SharedNoiseTable, make_schedule
from neuroevolution.concurrent_worker import ConcurrentWorkers
from neuroevolution.optimizers import SGD, Adam
import neuroevolution.models
import tabular_logger as tlogger
from threading import Lock
import gym_tensorflow
from tensorflow.python.tools import inspect_checkpoint as chkp


class TrainingState(object):
    def __init__(self, exp):
        self.num_frames = 0
        self.timesteps_so_far = 0
        self.time_elapsed = 0
        self.validation_timesteps_so_far = 0
        self.it = 0
        self.mutation_power = make_schedule(exp['mutation_power'])
        self.exp = exp

        self.theta = None
        self.optimizer = None

        if isinstance(exp['episode_cutoff_mode'], int):
            self.tslimit = exp['episode_cutoff_mode']
            self.incr_tslimit_threshold = None
            self.tslimit_incr_ratio = None
            self.adaptive_tslimit = False
        elif exp['episode_cutoff_mode'].startswith('adaptive:'):
            _, args = exp['episode_cutoff_mode'].split(':')
            arg0, arg1, arg2, arg3 = args.split(',')
            self.tslimit, self.incr_tslimit_threshold, self.tslimit_incr_ratio, self.tslimit_max = int(arg0), float(arg1), float(arg2), float(arg3)
            self.adaptive_tslimit = True
            tlogger.info(
                'Starting timestep limit set to {}. When {}% of rollouts hit the limit, it will be increased by {}'.format(
                    self.tslimit, self.incr_tslimit_threshold * 100, self.tslimit_incr_ratio))
        elif exp['episode_cutoff_mode'] == 'env_default':
            self.tslimit, self.incr_tslimit_threshold, self.tslimit_incr_ratio = None, None, None
            self.adaptive_tslimit = False
        else:
            raise NotImplementedError(exp['episode_cutoff_mode'])

    def initialize(self, rs, noise, model):
        theta, _ = model.randomize(rs, noise)
        print("Theta shape: {}".format(theta.shape))
        self.set_theta(theta)

    def set_theta(self, theta):
        self.theta = theta
        self.optimizer = {'sgd': SGD, 'adam': Adam}[self.exp['optimizer']['type']](self.theta, **self.exp['optimizer']['args'])

    def sample(self, schedule):
        return schedule.value(iteration=self.it, timesteps_so_far=self.timesteps_so_far)

class Offspring(object):
    def __init__(self, seeds, rewards, ep_len, validation_rewards=[], validation_ep_len=[]):
        self.seeds = seeds
        self.rewards = rewards
        self.ep_len = ep_len
        self.validation_rewards = validation_rewards
        self.validation_ep_len = validation_ep_len

    @property
    def fitness(self):
        return np.mean(self.rewards)

    @property
    def training_steps(self):
        return np.sum(self.ep_len)


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)

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


def batched_weighted_sum(weights, vecs, batch_size):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed

def make_offspring(exp, noise, rs, worker, state, seeds_vector = None):
    for i in range(exp['population_size'] // 2 // len(exp['games'])):
        if seeds_vector is None:
            idx = noise.sample_index(rs, worker.model.num_params)
        else:
            idx = seeds_vector[i]
        mutation_power = state.sample(state.mutation_power)
        pos_theta = worker.model.compute_mutation(noise, state.theta, idx, mutation_power)
        yield (pos_theta, idx)

        neg_theta = worker.model.compute_mutation(noise, state.theta, idx, -mutation_power)
        diff = (np.max(np.abs((pos_theta + neg_theta) / 2 - state.theta)))
        assert diff < 1e-5, 'Diff too large: {}'.format(diff)
        yield (neg_theta, idx)


def save_pickle(iteration, log_dir, pickle_filename, dat):
    save_file = os.path.join(log_dir, "{:04d}-{}.pkl".format(iteration, pickle_filename))
    with open(save_file, 'wb+') as file:
        pickle.dump(dat, file, pickle.HIGHEST_PROTOCOL)
    tlogger.info("Saved {}".format(save_file))

def load_pickle(iteration, log_dir, pickle_filename):
    save_file = os.path.join(log_dir, "{:04d}-{}.pkl".format(iteration, pickle_filename))
    tlogger.info("Loading {}".format(save_file))
    with open(save_file, 'rb') as file:
        return pickle.load(file)

def obtain_proc_returns(learn_option, game_returns):
    if learn_option not in ["concat_rewards", "alternate_games", "equal_prob_random_choice"]:
        raise NotImplementedError(learn_option)

    if learn_option == 'concat_rewards':
        proc_returns = [compute_centered_ranks(gameret) for gameret in game_returns]
        return np.concatenate(tuple(proc_returns))

    if learn_option == 'alternate_games':
        pass

    if learn_option == 'equal_prob_random_choice':
        total_number_of_games = len(game_returns)
        probs = np.array([1] * total_number_of_games) / float(total_number_of_games)
        compute_centered_ranks_for_game = np.random.choice(range(total_number_of_games), p=probs)
        game_index = compute_centered_ranks_for_game
        print("Choosing to optimize for game_index: {}".format(game_index))
        proc_returns = [compute_centered_ranks(game_returns[game_index]) for g in game_returns]
        return np.concatenate(tuple(proc_returns))


def main(**exp):
    log_dir = tlogger.log_dir()

    tlogger.info(json.dumps(exp, indent=4, sort_keys=True))
    tlogger.info('Logging to: {}'.format(log_dir))
    Model = neuroevolution.models.__dict__[exp['model']]
    all_tstart = time.time()

    noise = SharedNoiseTable()
    rs = np.random.RandomState()

    def make_env0(b):
        return gym_tensorflow.make(game=exp["games"][0], batch_size=b)
    def make_env1(b):
        return gym_tensorflow.make(game=exp["games"][1], batch_size=b)

    workers = [
        ConcurrentWorkers(make_env0, Model, batch_size=64),
        ConcurrentWorkers(make_env1, Model, batch_size=64)
    ]

    saver = tf.train.Saver()

    tlogger.info('Start timing')
    tstart = time.time()
    tf_sess = tf.Session()
    tf_sess.run(tf.global_variables_initializer())
    state = TrainingState(exp)
    state.initialize(rs, noise, workers[0].model)

    workers[0].initialize(tf_sess)
    workers[1].initialize(tf_sess)

    for iteration in range(exp['iterations']):
        tlogger.info("BEGINNING ITERATION: {}".format(iteration))


        ##############
        ### GAME 0 ###
        ##############
        worker = workers[0]
        frames_computed_so_far = tf_sess.run(worker.steps_counter)
        game0_results = []
        game0_rewards = []
        game0_episode_lengths = []

        iterator = iter(worker.monitor_eval(make_offspring(exp, noise, rs, worker, state), max_frames=state.tslimit * 4))

        for pos_seeds, pos_reward, pos_length in iterator:
            neg_seeds, neg_reward, neg_length = next(iterator)
            assert pos_seeds == neg_seeds
            result = Offspring(pos_seeds, [pos_reward, neg_reward], [pos_length, neg_length])
            rewards = result.rewards
            game0_results.append(result)
            game0_rewards.append(rewards)
            game0_episode_lengths.append(result.ep_len)
        state.num_frames += tf_sess.run(worker.steps_counter) - frames_computed_so_far
        game0_returns_n2 = np.array([a.rewards for a in game0_results])
        game0_noise_inds_n = [a.seeds for a in game0_results]
        # tlogger.info("game0 rewards: {}".format(np.mean(game0_rewards)))
        # tlogger.info("game0 eplens: {}".format(game0_episode_lengths))
        save_pickle(iteration, log_dir, "game0_rewards", game0_rewards)
        save_pickle(iteration, log_dir, "game0_episode_lengths", game0_episode_lengths)

        ##############
        ### GAME 1 ###
        ##############
        worker = workers[1]
        frames_computed_so_far = tf_sess.run(worker.steps_counter)
        game1_results = []
        game1_rewards = []
        game1_episode_lengths = []
        seeds_vector = np.array(game0_noise_inds_n)
        iterator = iter(worker.monitor_eval(make_offspring(exp, noise, rs, worker, state, seeds_vector), max_frames=state.tslimit * 4))

        for pos_seeds, pos_reward, pos_length in iterator:
            neg_seeds, neg_reward, neg_length = next(iterator)
            assert pos_seeds == neg_seeds
            result = Offspring(pos_seeds, [pos_reward, neg_reward], [pos_length, neg_length])
            rewards = result.rewards
            game1_results.append(result)
            game1_rewards.append(rewards)
            game1_episode_lengths.append(result.ep_len)
        state.num_frames += tf_sess.run(worker.steps_counter) - frames_computed_so_far
        game1_returns_n2 = np.array([a.rewards for a in game1_results])
        game1_noise_inds_n = [a.seeds for a in game1_results]
        # tlogger.info("game1 rewards: {}".format(np.mean(game1_rewards)))
        # tlogger.info("game1 eplens: {}".format(game0_episode_lengths))
        save_pickle(iteration, log_dir, "game1_rewards", game1_rewards)
        save_pickle(iteration, log_dir, "game1_episode_lengths", game1_episode_lengths)

        tlogger.info("Saving offsprings seeds")
        save_pickle(iteration, log_dir, "offsprings_seeds", game1_noise_inds_n)

        ####################
        ### UPDATE THETA ###
        ####################
        game_returns = [game0_returns_n2, game1_returns_n2]
        proc_returns = obtain_proc_returns(exp['learn_option'], game_returns)

        assert game0_noise_inds_n == game1_noise_inds_n
        noise_inds_n = game0_noise_inds_n + game1_noise_inds_n # concatenate the two lists


# TOP 100 offspring
#        dx = proc_returns[:, 0]
#        dy = proc_returns[:, 1]
#        dist_squared = (np.ones(dx.shape) - np.abs(dx))**2 + (np.ones(dy.shape) - np.abs(dy))**2
#        top_n_rewards = dist_squared.argsort()[-100:][::-1]
#        batched_weighted_indices = (noise.get(idx, worker.model.num_params) for idx in noise_inds_n)
#        proc_returns = proc_returns[top_n_rewards, :]
#        batched_weighted_args = {
#            'deltas': proc_returns[:, 0] - proc_returns[:, 1],
#            'indices': [myval for myidx, myval in enumerate(batched_weighted_indices) if myidx in top_n_rewards]
#        }
#        noise_inds_n = batched_weighted_args['indices']
#        g, count = batched_weighted_sum(batched_weighted_args['deltas'], batched_weighted_args['indices'], batch_size=len(batched_weighted_args['deltas']))

# ALL offspring
        g, count = batched_weighted_sum(
                proc_returns[:, 0] - proc_returns[:, 1],
                (noise.get(idx, worker.model.num_params) for idx in noise_inds_n),
                batch_size=500
        )

        # NOTE: gradients are scaled by \theta
        returns_n2 = np.array([a.rewards for a in game0_results] + [a.rewards for a in game1_results])

# Only if using top 100
#        returns_n2 = returns_n2[top_n_rewards]

        g /= returns_n2.size

        assert g.shape == (worker.model.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
        update_ratio, state.theta = state.optimizer.update(-g + exp['l2coeff'] * state.theta)

        save_pickle(iteration, log_dir, "state", state)

        ######################
        ### EVALUATE ELITE ###
        ######################
        _, test_evals, test_timesteps = workers[0].monitor_eval_repeated([(state.theta, 0)], max_frames=None, num_episodes=exp['num_test_episodes']//2)[0]
        tlogger.info("game0 elite: {}".format(np.mean(test_evals)))
        save_pickle(iteration, log_dir, 'game0_elite', test_evals)
        save_pickle(iteration, log_dir, 'game0_elite_timestemps', test_timesteps)

        _, test_evals, test_timesteps = workers[1].monitor_eval_repeated([(state.theta, 0)], max_frames=None, num_episodes=exp['num_test_episodes']//2)[0]
        tlogger.info("game1 elite: {}".format(np.mean(test_evals)))
        save_pickle(iteration, log_dir, "game1_elite", test_evals)
        save_pickle(iteration, log_dir, 'game1_elite_timestemps', test_timesteps)

        state.num_frames += tf_sess.run(worker.steps_counter) - frames_computed_so_far

        saver.save(tf_sess, "{}/model-{}".format(log_dir, state.it))

        state.it += 1

    os.kill(os.getpid(), signal.SIGTERM)



    #     while True:
    #         tstart_iteration = time.time()
    #         frames_computed_so_far = sess.run(worker.steps_counter)
    #
    #         tlogger.info('Evaluating perturbations')
    #         iterator = iter(worker.monitor_eval(make_offspring(state), max_frames=state.tslimit * 4))
    #         results = []
    #
    #         for pos_seeds, pos_reward, pos_length in iterator:
    #             neg_seeds, neg_reward, neg_length = next(iterator)
    #             assert pos_seeds == neg_seeds
    #             results.append(Offspring(pos_seeds, [pos_reward, neg_reward], [pos_length, neg_length]))
    #         state.num_frames += sess.run(worker.steps_counter) - frames_computed_so_far
    #
    #         state.it += 1
    #         tlogger.record_tabular('Iteration', state.it)
    #         tlogger.record_tabular('MutationPower', state.sample(state.mutation_power))
    #         tlogger.record_tabular('TimestepLimitPerEpisode', state.tslimit)
    #
    #         # Trim unwanted results
    #         results = results[:exp['population_size']//2]
    #         assert len(results) == exp['population_size']//2
    #         rewards = np.array([b for a in results for b in a.rewards])
    #
    #         results_timesteps = np.array([a.training_steps for a in results])
    #         timesteps_this_iter = sum([a.training_steps for a in results])
    #         state.timesteps_so_far += timesteps_this_iter
    #
    #         tlogger.record_tabular('PopulationEpRewMax', np.max(rewards))
    #         tlogger.record_tabular('PopulationEpRewMean', np.mean(rewards))
    #         tlogger.record_tabular('PopulationEpRewMedian', np.median(rewards))
    #         tlogger.record_tabular('PopulationEpCount', len(rewards))
    #         tlogger.record_tabular('PopulationTimesteps', timesteps_this_iter)
    #
    #         # Update Theta
    #         returns_n2 = np.array([a.rewards for a in results])
    #         noise_inds_n = [a.seeds for a in results]
    #
    #         if exp['return_proc_mode'] == 'centered_rank':
    #             proc_returns_n2 = compute_centered_ranks(returns_n2)
    #         else:
    #             raise NotImplementedError(exp['return_proc_mode'])
    #         # Compute and take step
    #         g, count = batched_weighted_sum(
    #             proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
    #             (noise.get(idx, worker.model.num_params) for idx in noise_inds_n),
    #             batch_size=500
    #         )
    #         # NOTE: gradients are scaled by \theta
    #         g /= returns_n2.size
    #
    #         assert g.shape == (worker.model.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
    #         update_ratio, state.theta = state.optimizer.update(-g + exp['l2coeff'] * state.theta)
    #
    #         time_elapsed_this_iter = time.time() - tstart_iteration
    #         state.time_elapsed += time_elapsed_this_iter
    #         tlogger.info('Evaluate elite')
    #         _, test_evals, test_timesteps = worker.monitor_eval_repeated([(state.theta, 0)], max_frames=None, num_episodes=exp['num_test_episodes'])[0]
    #         test_timesteps = sum(test_timesteps)
    #         # Log Results
    #         tlogger.record_tabular('TestRewMean', np.mean(test_evals))
    #         tlogger.record_tabular('TestRewMedian', np.median(test_evals))
    #         tlogger.record_tabular('TestEpCount', len(test_evals))
    #         tlogger.record_tabular('TestEpLenSum', test_timesteps)
    #         tlogger.record_tabular('InitialRewMax', np.max(initial_performance))
    #         tlogger.record_tabular('InitialRewMean', np.mean(initial_performance))
    #         tlogger.record_tabular('InitialRewMedian', np.median(initial_performance))
    #
    #         tlogger.record_tabular('TimestepsThisIter', timesteps_this_iter)
    #         tlogger.record_tabular('TimestepsPerSecondThisIter', timesteps_this_iter/(time.time()-tstart_iteration))
    #         tlogger.record_tabular('TimestepsComputed', state.num_frames)
    #         tlogger.record_tabular('TimestepsSoFar', state.timesteps_so_far)
    #         tlogger.record_tabular('TimeElapsedThisIter', time_elapsed_this_iter)
    #         tlogger.record_tabular('TimeElapsedThisIterTotal', time.time()-tstart_iteration)
    #         tlogger.record_tabular('TimeElapsed', state.time_elapsed)
    #         tlogger.record_tabular('TimeElapsedTotal', time.time()-all_tstart)
    #
    #         tlogger.dump_tabular()
    #         fps = state.timesteps_so_far/(time.time() - tstart)
    #         tlogger.info('Timesteps Per Second: {:.0f}. Elapsed: {:.2f}h ETA {:.2f}h'.format(fps, (time.time() - all_tstart) / 3600, (exp['timesteps'] - state.timesteps_so_far) / fps / 3600))
    #
    #         results.clear()
    #
    # print("Done with with(WorkerSession)")

if __name__ == "__main__":
    with open(sys.argv[-1], 'r') as f:
        exp = json.loads(f.read())
    main(**exp)
