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

def make_env_game0(b):
    return gym_tensorflow.make(game=exp['games'][0], batch_size=b)
def make_env_game1(b):
    return gym_tensorflow.make(game=exp['games'][1], batch_size=b)

def make_offspring(state):
    for i in range(exp['population_size'] // 2):
        idx = noise.sample_index(rs, worker.model.num_params)
        mutation_power = state.sample(state.mutation_power)
        pos_theta = worker.model.compute_mutation(noise, state.theta, idx, mutation_power)

        yield (pos_theta, idx)
        neg_theta = worker.model.compute_mutation(noise, state.theta, idx, -mutation_power)
        diff = (np.max(np.abs((pos_theta + neg_theta)/2 - state.theta)))
        assert diff < 1e-5, 'Diff too large: {}'.format(diff)

        yield (neg_theta, idx)
