import random

moves_stack = [x for x in range(100)]
oppo = []
prev_reward = 0
prev_action = 0
total_bnd = [0 for x in range(100)]
won_bnd = [0 for x in range(100)]


def get_bandit():
    best_bandit = 0
    best_score = 0
    for bnd in range(100):

        if total_bnd[bnd] <= 3:
            return bnd

        this_score = (won_bnd[bnd] / total_bnd[bnd])

        if this_score > best_score:
            best_score = this_score
            best_bandit = bnd

    return best_bandit


def agent(obs, conf):
    global moves_stack, oppo, prev_reward, prev_action, total_bnd, won_bnd

    if obs.step == 1:
        prev_action = moves_stack.pop(0)

        total_bnd[prev_action] += 1
        return prev_action

    my_idx = obs['agentIndex']

    if obs.step > 5:
        oppo.append(obs['lastActions'][1 - my_idx])

    reward_this_time = obs.reward - prev_reward
    prev_reward = obs.reward

    if reward_this_time > 0:
        moves_stack.insert(0, prev_action)
        won_bnd[prev_action] += 1

    if len(oppo) >= 3:
        if oppo[-1] == oppo[-2] and oppo[-1] == oppo[-3]:
            moves_stack.insert(0, oppo[-1])

    if len(moves_stack) == 0:
        moves_stack.insert(0, get_bandit())

    prev_action = moves_stack.pop(0)

    total_bnd[prev_action] += 1
    return prev_action