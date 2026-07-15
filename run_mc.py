import json
from monte_carlo import do_mc_per_episode

shelf_life_values = [2, 4, 8]
seeds = [1, 2, 3]
num_episodes = 100000

results = []
for shelf_life in shelf_life_values:
    for seed in seeds:
        shelf_life, episodes, q_values, optimal_actions, visits_x_u, runtime, delta_list = do_mc_per_episode(shelf_life, num_episodes, seed=seed)
        results.append({"shelf_life": shelf_life,
                        "seed": seed,
                        "episodes": episodes,
                        "runtime_sec": runtime,
                        "q_values": q_values.tolist(),
                        "policy": optimal_actions.tolist(),
                        "visits": visits_x_u.tolist(),
                        "delta_list": delta_list})

with open("mc_100k_results_mu10.json", "w") as file:
    json.dump(results, file, indent=2)
