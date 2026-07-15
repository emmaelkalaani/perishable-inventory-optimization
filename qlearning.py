# imports
import scipy as sp
import numpy as np
import time

# variables
mu = 10
max_stock = 50
order_cost = 40
item_cost = 2
holding_cost = 0.1
perishing_cost = 8
shortage_penalty = 10
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
steps_per_episode = 1000

# probability distribution
max_demand = int(mu + (3*np.sqrt(mu)))
possible_demands = list(range(0, (max_demand + 1)))
probabilities = sp.stats.poisson.pmf(possible_demands, mu)
probabilities = probabilities / np.sum(probabilities)


def get_demand():
    return np.random.choice(possible_demands, p=probabilities)


# handle transitions between states


def transition_function(x, u, w, shelf_life):
    stock = x + u - w
    if stock < 0:
        item_shortage = -stock
        stock = 0
        perished = 0
    else:
        item_shortage = 0
        perished = np.random.binomial(stock, 1 / shelf_life)
        stock -= perished
    return stock, item_shortage, perished  # return the new stock amount and shortage / pershed amount


# handle calculating cost
def cost_function(leftover_stock, ordered, shortage, perished):
    total_cost = 0
    if (ordered > 0):
        total_cost += order_cost
    total_cost += ordered * item_cost
    total_cost += shortage * shortage_penalty
    total_cost += leftover_stock * holding_cost
    total_cost += perished * perishing_cost

    return total_cost  # return the total cost


def do_ql_per_episode(shelf_life, num_episodes, seed=1):

    np.random.seed(seed)
    q_values = np.zeros((max_stock + 1, max_stock + 1))
    visits_x_u = np.zeros((max_stock + 1, max_stock + 1))
    epsilon = 0.5  # exploration rate
    delta = 1
    delta_list = []
    episodes = 0
    start = time.perf_counter()

    while episodes < num_episodes:
        old_q_values = q_values.copy()
        current_state = np.random.randint(0, max_stock + 1)

        for step in range(steps_per_episode):
            max_order = max_stock - current_state
            if np.random.random() < epsilon:
                action = np.random.randint(0, max_order + 1)
            else:
                possible_actions = q_values[current_state, :max_order + 1]
                action = np.argmin(possible_actions)
            demand = get_demand()
            next_state, item_shortage, perished = transition_function(current_state, action, demand, shelf_life)
            cost = cost_function(next_state, action, item_shortage, perished)
            q_values[current_state, action] += alpha * (cost + gamma * np.min(q_values[next_state, :max_stock - next_state + 1]) - q_values[current_state, action])
            visits_x_u[current_state, action] += 1
            current_state = next_state

        delta = np.max(np.abs(q_values - old_q_values))
        delta_list.append(delta)  # store delta for reference

        episodes += 1
        epsilon = max(0.1, epsilon * 0.99999)  # decay

    runtime = time.perf_counter() - start

    optimal_actions = np.zeros(max_stock + 1)
    for state in range(max_stock + 1):
        max_order = max_stock - state
        optimal_actions[state] = np.argmin(q_values[state, :max_order + 1])

    return (shelf_life, episodes, q_values, optimal_actions, visits_x_u, runtime, delta_list)


def do_ql_per_thresh(shelf_life, convergence_thresh, seed=1):

    np.random.seed(seed)
    q_values = np.zeros((max_stock + 1, max_stock + 1))
    visits_x_u = np.zeros((max_stock + 1, max_stock + 1))
    alpha = 0.1  # learning rate
    epsilon = 0.5  # exploration rate
    delta = 1
    delta_list = []
    episodes = 0
    start = time.perf_counter()

    while delta > convergence_thresh:
        old_q_values = q_values.copy()
        current_state = np.random.randint(0, max_stock + 1)

        for step in range(steps_per_episode):
            max_order = max_stock - current_state
            if np.random.random() < epsilon:
                action = np.random.randint(0, max_order + 1)
            else:
                possible_actions = q_values[current_state, :max_order + 1]
                action = np.argmin(possible_actions)
            demand = get_demand()
            next_state, item_shortage, perished = transition_function(current_state, action, demand, shelf_life)
            cost = cost_function(next_state, action, item_shortage, perished)
            q_values[current_state, action] += alpha * (cost + gamma * np.min(q_values[next_state, :max_stock - next_state + 1]) - q_values[current_state, action])
            visits_x_u[current_state, action] += 1
            current_state = next_state

        delta = np.max(np.abs(q_values - old_q_values))
        delta_list.append(delta)  # store delta for reference

        episodes += 1
        epsilon = max(0.1, epsilon * 0.99999)  # decay

    runtime = time.perf_counter() - start

    optimal_actions = np.zeros(max_stock + 1)
    for state in range(max_stock + 1):
        max_order = max_stock - state
        optimal_actions[state] = np.argmin(q_values[state, :max_order + 1])

    return (shelf_life, episodes, q_values, optimal_actions, visits_x_u, runtime, delta_list)
