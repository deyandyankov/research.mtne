import numpy as np

class Offspring(object):
    def __init__(self, game_index, seeds, rewards, ep_len, validation_rewards=[], validation_ep_len=[]):
        self.game_index = game_index
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
