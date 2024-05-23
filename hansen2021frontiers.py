import numpy as np
np.random.seed(93)
from multiprocessing import Pool
from env import Hansen_env, Hansen_agent
from env import Hansen_initialize as initialize
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import json

def simulate(delta):
    info = {'alpha': 0.48, 'beta': 0.9, 'gamma': 0.6, 'delta': delta}
    env = Hansen_env(info)
    agent1, agent2 = Hansen_agent(), Hansen_agent()
    initialize(env, agent1, agent2)
    for _ in range(2000000-2):
        actions = (agent1.action(), agent2.action())
        rewards = env.step(actions)
        agent1.update(rewards[0])
        agent2.update(rewards[1])
    price1 = np.median(np.array(env.prices_log)[:, 0])
    price2 = np.median(np.array(env.prices_log)[:, 1])
    return price1, price2

if __name__ == '__main__':
    delta_list = [0.1, 0.2, 0.4, 1, 2.5, 5, 10]
    results = {delta:[] for delta in delta_list}
    for delta in delta_list:
        t0 = time.time()
        print(f"Now simulating params delta: {delta}.")
        params = [(delta,) for _ in range(500)]
        with Pool(processes=15) as pool:
            result = pool.starmap(simulate, params)
        t1 = time.time() - t0
        print(f"The calculation takes {t1} seconds.")
        results[delta] = result
    with open('data/hansen2021frontiers_results.json', 'w') as json_file:
        json.dump(results, json_file)
    print('Done!')