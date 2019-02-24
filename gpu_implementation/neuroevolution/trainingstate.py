from .optimizers import SGD, Adam
from .helper import make_schedule

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
        print("DEYAN initialize() from TrainingState")
        theta, _ = model.randomize(rs, noise)
        self.set_theta(theta)

    def set_theta(self, theta):
        self.theta = theta
        self.optimizer = {'sgd': SGD, 'adam': Adam}[self.exp['optimizer']['type']](self.theta, **self.exp['optimizer']['args'])

    def sample(self, schedule):
        return schedule.value(iteration=self.it, timesteps_so_far=self.timesteps_so_far)
