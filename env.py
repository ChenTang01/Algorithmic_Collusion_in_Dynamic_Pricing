import numpy as np
np.random.seed(93)
import matplotlib.pyplot as plt
from collections import deque

class Calvano_env():

    def __init__(self, agent1_info, agent2_info, other_info):
        """
        All input parameters should be dictionaries.
        agent_info should contain keys: 'a', 'alpha', 'beta', 'c', 'delta'
        other_info should contain keys: 'a0', 'mu', 'm', 'xi', 'converge_time'
        """
        self.agent1_info = agent1_info
        self.agent2_info = agent2_info
        self.other_info = other_info
        self.rewards_log = deque(maxlen=100000) # [] in the notebook
        self.actions_log = deque(maxlen=100000) # [] in the notebook
        self.prices_log = deque(maxlen=100000) # [] in the notebook
        self.action_spaces = None
        self.nash = None
        self.collusion = None
        self.Delta = None
        self.delta = None
        self.converge_round = 0
        self.check()
        self.initialize()
    
    def check(self):
        agent_key = ['a', 'alpha', 'beta', 'c', 'delta']
        other_key = ['a0', 'mu', 'm', 'xi', 'converge_time']
        if not all(key in self.agent1_info for key in agent_key):
            raise ValueError("Incorrect Parameters")
        if not all(key in self.agent2_info for key in agent_key):
            raise ValueError("Incorrect Parameters")
        if not all(key in self.other_info for key in other_key):
            raise ValueError("Incorrect Parameters")
        
    def step(self, actions, converges):
        converge = False
        self.actions_log.append(actions)
        prices = self.action_spaces[0][actions[0]], self.action_spaces[1][actions[1]]
        self.prices_log.append(prices)
        rewards = self.execute(prices)
        self.rewards_log.append(rewards)
        states = (self.other_info['m']*actions[0] + actions[1], self.other_info['m']*actions[1] + actions[0])
        if converges[1] and converges[0]:
            self.converge_round += 1 
        else:
            self.converge_round = 0
        if self.converge_round >= self.other_info['converge_time']:
            converge = True
        return states, rewards, converge
    
    def execute(self, actions):
        a0, mu = self.other_info['a0'], self.other_info['mu']
        a1, a2 = self.agent1_info['a'], self.agent2_info['a']
        base = np.exp(a0 / mu)
        demands = np.exp((np.array([a1, a2]) - np.array(actions)) / mu)
        demands = demands / (demands.sum() + base)
        costs = np.array([self.agent1_info['c'], self.agent2_info['c']])
        rewards = demands * (actions - costs)
        return rewards
    
    def initialize(self, lower = 1, upper = 2.5, space = 0.01):
        """
        Compute the nash, collusion prices and the actions spaces.
        """
        prices = np.arange(lower, upper, space)
        matrix = np.empty((len(prices), len(prices), 2))
        for i, p1 in enumerate(prices):
            for j, p2 in enumerate(prices):
                matrix[i, j] = self.execute((p1, p2))
        max_agent1 = np.argmax(matrix[:, :, 0], axis=0)
        dominant1 = list(zip(max_agent1, range(len(prices))))
        max_agent2 = np.argmax(matrix[:, :, 1], axis=1)
        dominant2 = list(zip(range(len(prices)), max_agent2))
        nash = [x for x in dominant1 if x in dominant2]
        if len(nash) == 0:
            raise ValueError('There is no Nash-equilibrium.')
        """
        elif len(nash) > 1:
            print("There are multiple Nash-equilibriums, picked the first one.")
        """
        nash = nash[0]
        self.nash = np.array((prices[nash[0]], prices[nash[1]]))
        #print(f"The nash equilibrium prices are {self.nash[0]}, {self.nash[1]}")
        matrix = np.sum(matrix, axis=2)
        collusion = np.unravel_index(np.argmax(matrix), matrix.shape)
        self.collusion = np.array((prices[collusion[0]], prices[collusion[1]]))
        #print(f"The collusion prices are {self.collusion[0]}, {self.collusion[1]}")
        lower = self.nash - self.other_info['xi'] * (self.collusion - self.nash)
        upper = self.collusion + self.other_info['xi'] * (self.collusion - self.nash)
        self.action_spaces = [np.linspace(lower[0], upper[0], self.other_info['m']), 
                            np.linspace(lower[1], upper[1], self.other_info['m'])]
    
    def cal_Delta(self):
        nash = np.sum(self.execute(self.nash))
        collusion = np.sum(self.execute(self.collusion))
        profits = np.mean(np.sum(self.rewards_log, axis=1))
        Delta = (profits - nash) / (collusion - nash)
        self.Delta = Delta
        """
        #original version in the notebook
        nash = np.sum(self.execute(self.nash))
        collusion = np.sum(self.execute(self.collusion))
        profits = np.sum(self.rewards_log, axis=1)
        cum_profits = np.cumsum(profits) / np.arange(1, len(profits) + 1)
        delta = (profits - nash) / (collusion - nash)
        Delta = (cum_profits - nash) / (collusion - nash)
        self.delta = delta
        self.Delta = Delta
        """

    def plot(self):
        self.cal_Delta()
        Delta, delta = self.Delta, self.delta
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))

        ax1.plot(np.arange(1, len(Delta) + 1), Delta, label='Dynamics', color='aqua', linewidth=0.8)
        ax1.axhline(y=0, color='violet', linestyle='--', linewidth=1, label='Nash')
        ax1.axhline(y=1, color='crimson', linestyle='--', linewidth=1, label='Collusion')
        ax1.set_title('Cummulative Delta')
        ax1.set_xlabel('t')
        ax1.set_ylabel('Delta')

        ax2.plot(np.arange(1, len(delta) + 1), delta, label='Dynamics', color='aqua', linewidth=0.08)
        ax2.axhline(y=0, color='violet', linestyle='--', linewidth=1, label='Nash')
        ax2.axhline(y=1, color='crimson', linestyle='--', linewidth=1, label='Collusion')
        ax2.set_title('Instantaneous Delta')
        ax2.set_xlabel('t')
        ax2.set_ylabel('Delta')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

class Calvano_agent:
    def __init__(self, agent_info, other_info, env):
        self.t = 0
        self.last_action = None
        self.info = agent_info
        self.state_dim = other_info['m'] ** 2
        self.action_dim = other_info['m']
        self.state = np.random.choice(np.arange(self.state_dim)) # the initial state is randomly the same as in the paper
        self.Q = np.random.random((self.state_dim, self.action_dim))
        self.optimal = np.argmax(self.Q, axis=1)
        self.initialize_Q(env)

    def initialize_Q(self, env):
        for action, price in enumerate(env.action_spaces[0]):
            sum = np.sum([env.execute((price, comp))[0] for comp in env.action_spaces[0]])
            Q = sum / ((1-self.info['delta']) * self.action_dim)
            self.Q[:, action] = Q

    def action(self):
        epsilon = np.exp(-self.info['beta'] * self.t)
        optimal = np.argmax(self.Q, axis=1)
        if np.random.random() < epsilon:
            action = np.random.choice(np.arange(self.action_dim))
        else:
            action = optimal[self.state]
        self.last_action = action
        converge = (self.optimal == optimal).all()
        self.optimal = optimal
        return action, converge

    def update(self, new_state, reward):
        self.t += 1
        self.Q[self.state, self.last_action] = (1-self.info['alpha']) * self.Q[self.state, self.last_action] + self.info['alpha'] * (reward + self.info['delta']*np.max(self.Q[new_state, :])) 
        self.state = new_state
        
        

def Calvano_simulate(alpha, beta):
    agent1_info = {'a': 2, 'c': 1, 'alpha': alpha, 'beta': beta, 'delta': 0.95}
    agent2_info = {'a': 2, 'c': 1, 'alpha': alpha, 'beta': beta, 'delta': 0.95}
    other_info = {'a0': 0, 'mu': 0.25, 'm': 15, 'xi':0.1, 'converge_time': 100000}
    env = Calvano_env(agent1_info, agent2_info, other_info)
    agent1 = Calvano_agent(agent1_info, other_info, env)
    agent2 = Calvano_agent(agent2_info, other_info, env)
    iter = 0
    while iter <= 10000000:
        (action1, converge1), (action2, converge2) = agent1.action(), agent2.action()
        states, rewards, converge = env.step((action1, action2), (converge1, converge2))
        agent1.update(states[0], rewards[0])
        agent2.update(states[1], rewards[1])
        if converge:
            break
        iter += 1
    env.cal_Delta()
    return env.Delta

