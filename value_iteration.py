# imports
import scipy as sp
import numpy as np
import time

# variables
mu = 10  # mean demand
max_stock = 50
order_cost = 40
item_cost = 2
holding_cost = 0.1
perishing_cost = 8
shortage_penalty = 10

gamma = 0.9  # discount factor
theta = 0.01  # convergence threshold

# probability distribution
max_demand = int(mu + (3*np.sqrt(mu)))
possible_demands = list(range(0, (max_demand + 1)))
probabilities = sp.stats.poisson.pmf(possible_demands, mu)
probabilities = probabilities / np.sum(probabilities)


def transition_function(x, u, w, shelf_life):  # return all possible next states
    stock = x + u - w
    if stock < 0:
        item_shortage = -stock
        return [(0, item_shortage, 0, 1.0)]
    else:
        item_shortage = 0
        transition_outcomes = []
        for perished in range(stock + 1):
            prob = sp.stats.binom.pmf(perished, stock, 1/shelf_life)
            next_state = stock - perished
            transition_outcomes.append((next_state, item_shortage, perished, prob))
        return transition_outcomes


def cost_function(leftover_stock, ordered, shortage, perished):  # return cost of transition
    total_cost = 0
    if (ordered > 0):
        total_cost += order_cost
    total_cost += ordered * item_cost
    total_cost += shortage * shortage_penalty
    total_cost += leftover_stock * holding_cost
    total_cost += perished * perishing_cost

    return total_cost


def do_value_iteration(shelf_life):

    states_x = list(range(0, max_stock + 1))
    values_v = np.zeros(max_stock + 1)
    optimal_actions = np.zeros(max_stock + 1)
    iterations = 0
    start = time.perf_counter()

    while (True):
        delta = 0

        for state in states_x:
            last_value = values_v[state]
            max_order = max_stock - state
            action_value_pairs = []

            for action in range(max_order + 1):
                value = 0

                for demand in possible_demands:
                    demand_prob = probabilities[demand]
                    transition_possibilities = transition_function(state, action, demand, shelf_life)

                    for next_state, shortage, perished, trans_prob in transition_possibilities:
                        cost = cost_function(next_state, action, shortage, perished)
                        value += demand_prob * trans_prob * (cost + gamma * values_v[next_state])

                action_value_pairs.append((action, value))

            best_action, best_value = action_value_pairs[0]

            for action, value in action_value_pairs:
                if value < best_value:
                    best_action, best_value = action, value

            optimal_actions[state] = best_action
            values_v[state] = best_value

            delta = max(delta, abs(last_value - values_v[state]))

        iterations += 1
        if delta < theta:
            break

    runtime = time.perf_counter() - start

    return (shelf_life, iterations, runtime, delta, values_v, optimal_actions)
