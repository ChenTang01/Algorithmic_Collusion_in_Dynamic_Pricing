import numpy as np
np.random.seed(93)
import matplotlib.pyplot as plt
from multiprocessing import Pool
from env import Calvano_env, Calvano_agent
from env import Calvano_simulate as simulate
import time

if __name__ == '__main__':
    Alpha = np.linspace(0.25, 0.05, 20)
    Beta = np.linspace(2e-7, 2e-5, 20)
    matrix_delta = np.empty((20, 20))
    for ia, alpha in enumerate(Alpha):
        for ib, beta in enumerate(Beta):
            t0 = time.time()
            print(f"now simulating params alpha: {alpha} and beta: {beta}")
            params = [(alpha, beta) for _ in range(10)]
            with Pool(processes=10) as pool:
                results = pool.starmap(simulate, params)
            delta = np.array(results).mean()
            matrix_delta[ia, ib] = delta 
            t1 = time.time() - t0
            print(f"The delta is {delta}, the calculation takes {t1} seconds.")
    np.save('data/calvano_matrix_delta.npy', matrix_delta)
    print('Done!')
    
    
        