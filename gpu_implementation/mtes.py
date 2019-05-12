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
from neuroevolution.offspring import Offspring
from neuroevolution.trainingstate import TrainingState
from neuroevolution.tf_util import get_available_gpus, WorkerSession
from neuroevolution.helper import SharedNoiseTable
from neuroevolution.concurrent_worker import MTConcurrentWorkers
import neuroevolution.models
import tabular_logger as tlogger
from threading import Lock
from helper_methods import *

def main(**exp):
    log_dir = tlogger.log_dir()

    tlogger.info(json.dumps(exp, indent=4, sort_keys=True))
    tlogger.info('Logging to: {}'.format(log_dir))
    Model = neuroevolution.models.__dict__[exp['model']]
    all_tstart = time.time()

    def make_env_game0(b):
        return gym_tensorflow.make(game=exp['games'][0], batch_size=b)
    def make_env_game1(b):
        return gym_tensorflow.make(game=exp['games'][1], batch_size=b)

    def make_offspring(state):
        for i in range(exp['population_size'] // 4):
            idx = noise.sample_index(rs, worker.model.num_params)
            mutation_power = state.sample(state.mutation_power)
            pos_theta = worker.model.compute_mutation(noise, state.theta, idx, mutation_power)

            yield (pos_theta, idx)
            yield (pos_theta, idx)

            neg_theta = worker.model.compute_mutation(noise, state.theta, idx, -mutation_power)
            diff = (np.max(np.abs((pos_theta + neg_theta)/2 - state.theta)))
            assert diff < 1e-5, 'Diff too large: {}'.format(diff)

            yield (neg_theta, idx)
            yield (neg_theta, idx)

    worker = MTConcurrentWorkers([make_env_game0, make_env_game1], Model, batch_size=32)


    print("=== [mtes] worker.sess = {}".format(worker.sess))
    with WorkerSession(worker) as sess:
        print("=== [mtes] worker.sess = {}".format(worker.sess))
        noise = SharedNoiseTable()
        rs = np.random.RandomState()
        tlogger.info('Start timing')
        tstart = time.time()

        try:
            load_file = os.path.join(os.getcwd(), 'snapshot.pkl')
            with open(load_file, 'rb+') as file:
                state = pickle.load(file)
            tlogger.info("Loaded iteration {} from {}".format(state.it, load_file))
        except FileNotFoundError:
            tlogger.info('Failed to load snapshot')
            state = TrainingState(exp)

            if 'load_from' in exp:
                dirname = os.path.join(os.path.dirname(__file__), '..', 'neuroevolution', 'ga_legacy.py')
                load_from = exp['load_from'].format(**exp)
                os.system('python {} {} seeds.pkl'.format(dirname, load_from))
                with open('seeds.pkl', 'rb+') as file:
                    seeds = pickle.load(file)
                    state.set_theta(worker.model.compute_weights_from_seeds(noise, seeds))
                tlogger.info('Loaded initial theta from {}'.format(load_from))
            else:
                state.initialize(rs, noise, worker.model)

        tlogger.info('Start training')
        game_index, _, initial_performance, _ = worker.monitor_eval_repeated([(state.theta, 0)], max_frames=None, num_episodes=exp['num_test_episodes'])[0]

        print("=== past worker.monitor_eval_repeated")
        while True:
            print("=== next cycle in while loop")
            tstart_iteration = time.time()

            frames_computed_so_far = sess.run(worker.steps_counter)
            print("=== frames_captured_so_far = {}".format(frames_computed_so_far))
            tlogger.info('Evaluating perturbations')
            offspring_state = [o for o in make_offspring(state)]
            iterator = iter(worker.monitor_eval(offspring_state, max_frames=state.tslimit * 4))
            results = []
            for game_index, pos_seeds, pos_reward, pos_length in iterator:
                game_index, neg_seeds, neg_reward, neg_length = next(iterator)
                assert pos_seeds == neg_seeds
                results.append(Offspring(game_index, pos_seeds, [pos_reward, neg_reward], [pos_length, neg_length]))
            state.num_frames += sess.run(worker.steps_counter) - frames_computed_so_far

            state.it += 1
            tlogger.record_tabular('Iteration', state.it)
            tlogger.record_tabular('MutationPower', state.sample(state.mutation_power))
            tlogger.record_tabular('TimestepLimitPerEpisode', state.tslimit)

            # Trim unwanted results
            results = results[:exp['population_size']//2]
            assert len(results) == exp['population_size']//2
            rewards = np.array([b for a in results for b in a.rewards])

            results_timesteps = np.array([a.training_steps for a in results])
            timesteps_this_iter = sum([a.training_steps for a in results])
            state.timesteps_so_far += timesteps_this_iter

            tlogger.record_tabular('PopulationEpRewMax', np.max(rewards))
            tlogger.record_tabular('PopulationEpRewMean', np.mean(rewards))
            tlogger.record_tabular('PopulationEpRewMedian', np.median(rewards))
            tlogger.record_tabular('PopulationEpCount', len(rewards))
            tlogger.record_tabular('PopulationTimesteps', timesteps_this_iter)


            # Update Theta
            returns_n2 = np.array([a.rewards for a in results])
            noise_inds_n = [a.seeds for a in results]

            if exp['return_proc_mode'] == 'centered_rank':
                proc_returns_n2 = compute_centered_ranks(returns_n2)
            else:
                raise NotImplementedError(exp['return_proc_mode'])
            # Compute and take step
            g, count = batched_weighted_sum(
                proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
                (noise.get(idx, worker.model.num_params) for idx in noise_inds_n),
                batch_size=500
            )
            # NOTE: gradients are scaled by \theta
            g /= returns_n2.size

            assert g.shape == (worker.model.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
            update_ratio, state.theta = state.optimizer.update(-g + exp['l2coeff'] * state.theta)

            time_elapsed_this_iter = time.time() - tstart_iteration
            state.time_elapsed += time_elapsed_this_iter
            tlogger.info('Evaluate elite')

            game_index, _, test_evals, test_timesteps = worker.monitor_eval_repeated([(state.theta, 0)], max_frames=None, num_episodes=exp['num_test_episodes'])[0]
            test_timesteps = sum(test_timesteps)
            # Log Results
            tlogger.record_tabular('TestRewMin', np.min(test_evals))
            tlogger.record_tabular('TestRewMean', np.mean(test_evals))
            tlogger.record_tabular('TestRewMedian', np.median(test_evals))
            tlogger.record_tabular('TestRewMax', np.max(test_evals))
            tlogger.record_tabular('TestEpCount', len(test_evals))

            game_stats = [[], []]
            for k, v in enumerate(game_index):
                game_stats[v].append(test_evals[k])

            tlogger.record_tabular('Game0TestRewMin', np.min(game_stats[0]))
            tlogger.record_tabular('Game0TestRewMean', np.mean(game_stats[0]))
            tlogger.record_tabular('Game0TestRewMedian', np.median(game_stats[0]))
            tlogger.record_tabular('Game0TestRewMax', np.max(game_stats[0]))
            tlogger.record_tabular('Game0TestEpCount', len(game_stats[0]))

            tlogger.record_tabular('Game1TestRewMin', np.min(game_stats[1]))
            tlogger.record_tabular('Game1TestRewMean', np.mean(game_stats[1]))
            tlogger.record_tabular('Game1TestRewMedian', np.median(game_stats[1]))
            tlogger.record_tabular('Game1TestRewMax', np.max(game_stats[1]))
            tlogger.record_tabular('Game1TestEpCount', len(game_stats[1]))


            tlogger.record_tabular('TestEpLenSum', test_timesteps)
            tlogger.record_tabular('InitialRewMax', np.max(initial_performance))
            tlogger.record_tabular('InitialRewMean', np.mean(initial_performance))
            tlogger.record_tabular('InitialRewMedian', np.median(initial_performance))

            tlogger.record_tabular('TimestepsThisIter', timesteps_this_iter)
            tlogger.record_tabular('TimestepsPerSecondThisIter', timesteps_this_iter/(time.time()-tstart_iteration))
            tlogger.record_tabular('TimestepsComputed', state.num_frames)
            tlogger.record_tabular('TimestepsSoFar', state.timesteps_so_far)
            tlogger.record_tabular('TimeElapsedThisIter', time_elapsed_this_iter)
            tlogger.record_tabular('TimeElapsedThisIterTotal', time.time()-tstart_iteration)
            tlogger.record_tabular('TimeElapsed', state.time_elapsed)
            tlogger.record_tabular('TimeElapsedTotal', time.time()-all_tstart)

            tlogger.dump_tabular()

            fps = state.timesteps_so_far/(time.time() - tstart)
            tlogger.info('Timesteps Per Second: {:.0f}. Elapsed: {:.2f}h ETA {:.2f}h'.format(fps, (time.time() - all_tstart) / 3600, (exp['timesteps'] - state.timesteps_so_far) / fps / 3600))

            if state.adaptive_tslimit:
                if np.mean([a.training_steps >= state.tslimit for a in results]) > state.incr_tslimit_threshold:
                    state.tslimit = min(state.tslimit * state.tslimit_incr_ratio, state.tslimit_max)
                    tlogger.info('Increased threshold to {}'.format(state.tslimit))

            os.makedirs(log_dir, exist_ok=True)

            def save_pickle(dat, pickle_filename):
                save_file = os.path.join(log_dir, "{:04d}-{}.pkl".format(state.it, pickle_filename))
                with open(save_file, 'wb+') as file:
                    pickle.dump(dat, file, pickle.HIGHEST_PROTOCOL)
                tlogger.info("Saved {}".format(save_file))

            save_pickle(state, 'snapshot')
            save_pickle(game_stats, 'parent')
            save_pickle(returns_n2, 'offspring')
            save_pickle(offspring_state, 'offspring_state')

            # Stop if your reach the max amount of iterations
            if state.it >= exp['iterations']:
                tlogger.info('Training terminated after {} iterations. Exiting.'.format(state.it))
                os.kill(os.getpid(), signal.SIGTERM)

            results.clear()

if __name__ == "__main__":
    with open(sys.argv[-1], 'r') as f:
        exp = json.loads(f.read())
    main(**exp)
