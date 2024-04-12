"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
import numpy as np 
import csv
import random

logging.basicConfig()
logger = logging.getLogger("MAB Application")
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class Bandit(ABC):
    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass


class Visualization:
    def plot_cumulative_rewards(self, epsilon_greedy_rewards, thompson_rewards):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        cumulative_epsilon_greedy_rewards = np.cumsum(epsilon_greedy_rewards)
        cumulative_thompson_rewards = np.cumsum(thompson_rewards)
        plt.plot(cumulative_epsilon_greedy_rewards, label="Epsilon-Greedy")
        plt.plot(cumulative_thompson_rewards, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(np.log(cumulative_epsilon_greedy_rewards), label="Epsilon-Greedy (log scale)")
        plt.plot(np.log(cumulative_thompson_rewards), label="Thompson Sampling (log scale)")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward (log scale)")
        plt.legend()

        plt.show()

    def plot_cumulative_rewards_simple(self, epsilon_greedy_rewards, thompson_rewards):
        plt.figure(figsize=(10, 5))
        cumulative_epsilon_greedy_rewards = np.cumsum(epsilon_greedy_rewards)
        cumulative_thompson_rewards = np.cumsum(thompson_rewards)

        plt.plot(range(len(cumulative_epsilon_greedy_rewards)), cumulative_epsilon_greedy_rewards, label="Epsilon-Greedy")
        plt.plot(range(len(cumulative_thompson_rewards)), cumulative_thompson_rewards, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards")
        plt.legend()
        plt.show()

    def store_rewards_to_csv(self, epsilon_greedy_rewards, thompson_rewards):
        with open('bandit_rewards.csv', mode='w', newline='') as csv_file:
            fieldnames = ['Bandit', 'Reward', 'Algorithm']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for reward in epsilon_greedy_rewards:
                writer.writerow({'Bandit': 'EpsilonGreedy', 'Reward': reward, 'Algorithm': 'Epsilon-Greedy'})

            for reward in thompson_rewards:
                writer.writerow({'Bandit': 'ThompsonSampling', 'Reward': reward, 'Algorithm': 'Thompson Sampling'})

    def report_cumulative_reward_and_regret(self, epsilon_greedy_rewards, thompson_rewards):
        cumulative_epsilon_greedy_reward = sum(epsilon_greedy_rewards)
        cumulative_thompson_reward = sum(thompson_rewards)
        cumulative_epsilon_greedy_regret = max(Bandit_Reward) * len(epsilon_greedy_rewards) - cumulative_epsilon_greedy_reward
        cumulative_thompson_regret = max(Bandit_Reward) * len(thompson_rewards) - cumulative_thompson_reward

        logger.info(f'Cumulative Reward - Epsilon-Greedy: {cumulative_epsilon_greedy_reward}')
        logger.info(f'Cumulative Reward - Thompson Sampling: {cumulative_thompson_reward}')
        logger.info(f'Cumulative Regret - Epsilon-Greedy: {cumulative_epsilon_greedy_regret}')
        logger.info(f'Cumulative Regret - Thompson Sampling: {cumulative_thompson_regret}')


class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon=0.1):
        super().__init__(p)
        self.epsilon = epsilon
        self.estimates = [0] * len(p)
        self.bandit_rewards = p

    def pull(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.bandit_rewards) - 1)
        else:
            return np.argmax(self.estimates)

    def update(self, chosen_bandit, reward):
        self.num_trials += 1
        self.total_reward += reward
        self.estimates[chosen_bandit] += (reward - self.estimates[chosen_bandit]) / (self.num_trials)

    def experiment(self, num_trials):
        rewards = []
        for _ in range(num_trials):
            chosen_bandit = self.pull()
            reward = self.bandit_rewards[chosen_bandit]
            self.update(chosen_bandit, reward)
            rewards.append(reward)
        return rewards


class ThompsonSampling(Bandit):
    def __init__(self, rewards, alpha=1, beta=1):
        super().__init__(rewards)
        self.alpha = alpha
        self.beta = beta
        self.posteriors = [(alpha, beta) for _ in rewards]

    def pull(self):
        sampled_values = [np.random.beta(alpha, beta) for alpha, beta in self.posteriors]
        return np.argmax(sampled_values)

    def update(self, chosen_bandit, reward):
        self.num_trials += 1
        self.total_reward += reward
        alpha, beta = self.posteriors[chosen_bandit]
        self.posteriors[chosen_bandit] = (alpha + reward, beta + 1 - reward)

    def experiment(self, num_trials):
        rewards = []
        for _ in range(num_trials):
            chosen_bandit = self.pull()
            reward = self.bandit_rewards[chosen_bandit]
            self.update(chosen_bandit, reward)
            rewards.append(reward)
        return rewards


def comparison(num_trials):
    # Initialize bandit rewards
    Bandit_Reward = [1, 2, 3, 4]

    # Initialize bandit instances
    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward)
    thompson_bandit = ThompsonSampling(Bandit_Reward)

    # Run experiments
    epsilon_greedy_rewards = epsilon_greedy_bandit.experiment(num_trials)
    thompson_rewards = thompson_bandit.experiment(num_trials)

    # Visualize results
    vis = Visualization()
    vis.plot_cumulative_rewards(epsilon_greedy_rewards, thompson_rewards)
    vis.plot_cumulative_rewards_simple(epsilon_greedy_rewards, thompson_rewards)

    # Store rewards to CSV
    # vis.store_rewards_to_csv(epsilon_greedy_rewards, thompson_rewards)

    # Report cumulative reward and regret
    vis.report_cumulative_reward_and_regret(epsilon_greedy_rewards, thompson_rewards)


# Call the comparison function with the desired number of trials
comparison(20000)




if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
