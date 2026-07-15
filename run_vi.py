import json
from value_iteration import do_value_iteration

shelf_life_values = [2, 4, 8]
results = []

for shelf_life in shelf_life_values:
    shelf_life, iterations, runtime, delta, values_v, optimal_actions = do_value_iteration(shelf_life)
    results.append({"shelf_life": shelf_life,
                    "iterations": iterations,
                    "runtime_sec": runtime,
                    "final_delta": delta,
                    "values": values_v.tolist(),
                    "policy": optimal_actions.tolist()})

with open("vi_results_mu10.json", "w") as file:
    json.dump(results, file, indent=2)
