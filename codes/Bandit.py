import logging
import random
from abc import ABC, abstractmethod
import csv
import numpy as np
import matplotlib.pyplot as plt
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Bandit(ABC):
    """
    Abstract base class for bandit algorithms.
    """
    @abstractmethod
    def __init__(self, p):
        """
        Initializes the Bandit object.

        Args:
            p (list): The true mean rewards of each arm.
        """
        self.true_means = p
        self.estimated_means = [0.0] * len(p)
        self.action_counts = [0] * len(p)

    @abstractmethod
    def __repr__(self):
        """
        Returns a string representation of the Bandit object.
        """
        pass

    @abstractmethod
    def pull(self):
        """
        Chooses an arm to pull based on the bandit's strategy.

        Returns:
            int: The index of the chosen arm.
        """
        pass

    @abstractmethod
    def update(self, arm, reward):
        """
        Updates the estimated mean reward of the chosen arm.

        Args:
            arm (int): The index of the chosen arm.
            reward (float): The reward received from pulling the arm.
        """
        pass

    @abstractmethod
    def experiment(self, num_trials):
        """
        Runs an experiment for a specified number of trials.

        Args:
            num_trials (int): The number of trials to run the experiment for.

        Returns:
            list: A list of rewards obtained during the experiment.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Generates a report on the bandit's performance.

        Returns:
            str: A report containing average reward and regret.
        """
        pass


class EpsilonGreedy(Bandit):
    """
    Epsilon-greedy bandit algorithm.
    """
    def __init__(self, p, epsilon=0.1):
        """
        Initializes the EpsilonGreedy object.

        Args:
            p (list): The true mean rewards of each arm.
            epsilon (float): The epsilon value for the epsilon-greedy algorithm.
        """
        super().__init__(p)
        self.epsilon = epsilon

    def __repr__(self):
        """
        Returns a string representation of the EpsilonGreedy object.
        """
        return f"EpsilonGreedy Bandit with epsilon={self.epsilon}"

    def pull(self):
        """
        Chooses an arm to pull using the epsilon-greedy strategy.

        Returns:
            int: The index of the chosen arm.
        """
        if random.random() < self.epsilon:
            return random.randint(0, len(self.true_means) - 1)
        else:
            return self.estimated_means.index(max(self.estimated_means))

    def update(self, arm, reward):
        """
        Updates the estimated mean reward of the chosen arm.

        Args:
            arm (int): The index of the chosen arm.
            reward (float): The reward received from pulling the arm.
        """
        self.action_counts[arm] += 1
        n = self.action_counts[arm]
        self.estimated_means[arm] += (1 / n) * (reward - self.estimated_means[arm])

    def experiment(self, num_trials):
        """
        Runs an experiment for a specified number of trials.

        Args:
            num_trials (int): The number of trials to run the experiment for.

        Returns:
            list: A list of rewards obtained during the experiment.
        """
        rewards = []
        for t in range(1, num_trials + 1):
            self.epsilon = 1 / t  # Decay epsilon
            arm = self.pull()
            reward = self.true_means[arm]
            self.update(arm, reward)
            rewards.append(reward)
        return rewards

    def report(self):
        """
        Generates a report on the bandit's performance.

        Returns:
            str: A report containing average reward and regret.
        """
        avg_reward = sum(self.estimated_means) / len(self.estimated_means)
        avg_regret = max(self.true_means) - avg_reward
        return f"EpsilonGreedy Results: Average Reward={avg_reward:.2f}, Average Regret={avg_regret:.2f}"


class ThompsonSampling(Bandit):
    """
    Thompson Sampling bandit algorithm.
    """
    def __init__(self, p):
        """
        Initializes the ThompsonSampling object.

        Args:
            p (list): The true mean rewards of each arm.
        """
        super().__init__(p)
        self.alpha = [1] * len(p)
        self.beta = [1] * len(p)

    def __repr__(self):
        """
        Returns a string representation of the ThompsonSampling object.
        """
        return "ThompsonSampling Bandit"

    def pull(self):
        """
        Chooses an arm to pull using the Thompson Sampling strategy.

        Returns:
            int: The index of the chosen arm.
        """
        sampled_means = [random.betavariate(alpha, beta) for alpha, beta in zip(self.alpha, self.beta)]
        return sampled_means.index(max(sampled_means))

    def update(self, arm, reward):
        """
        Updates the parameters of the chosen arm based on the received reward.

        Args:
            arm (int): The index of the chosen arm.
            reward (int): The reward received from pulling the arm.
        """
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def experiment(self, num_trials):
        """
        Runs the Thompson Sampling experiment for a specified number of trials.

        Args:
            num_trials (int): The number of trials to run the experiment for.

        Returns:
            list: A list of rewards obtained during the experiment.
        """
        rewards = []
        for _ in range(num_trials):
            arm = self.pull()
            reward = self.true_means[arm]
            self.update(arm, reward)
            rewards.append(reward)
        return rewards

    def report(self):
        """
        Generates a report on the bandit's performance.

        Returns:
            str: A report containing average reward and regret.
        """
        avg_reward = sum(self.alpha) / (sum(self.alpha) + sum(self.beta))
        avg_regret = max(self.true_means) - avg_reward
        return f"ThompsonSampling Results: Average Reward={avg_reward:.2f}, Average Regret={avg_regret:.2f}"


class Visualization:
    """
    Class for visualizing bandit algorithms.
    """
    def plot_learning_process(self, epsilon_greedy_rewards, thompson_rewards):
        """
        Plot the learning process of Epsilon-Greedy and Thompson Sampling bandits.

        Args:
            epsilon_greedy_rewards (list): Rewards obtained from Epsilon-Greedy bandit.
            thompson_rewards (list): Rewards obtained from Thompson Sampling bandit.
        """
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

    def plot_cumulative_rewards(self, epsilon_greedy_rewards, thompson_rewards):
        """
        Plot the cumulative rewards of Epsilon-Greedy and Thompson Sampling bandits.

        Args:
            epsilon_greedy_rewards (list): Rewards obtained from Epsilon-Greedy bandit.
            thompson_rewards (list): Rewards obtained from Thompson Sampling bandit.
        """
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
        """
        Store rewards to a CSV file.

        Args:
            epsilon_greedy_rewards (list): Rewards obtained from Epsilon-Greedy bandit.
            thompson_rewards (list): Rewards obtained from Thompson Sampling bandit.
        """
        # Specify the path to the "results" folder located outside of the "codes" folder
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)  # Create the "results" folder if it doesn't exist

        # Specify the path to the CSV file within the "results" folder
        csv_file_path = os.path.join(results_dir, 'bandit_rewards.csv')

        # Write rewards to the CSV file
        with open(csv_file_path, mode='w', newline='') as csv_file:
            fieldnames = ['Bandit', 'Reward', 'Algorithm']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for reward in epsilon_greedy_rewards:
                writer.writerow({'Bandit': 'EpsilonGreedy', 'Reward': reward, 'Algorithm': 'Epsilon-Greedy'})

            for reward in thompson_rewards:
                writer.writerow({'Bandit': 'ThompsonSampling', 'Reward': reward, 'Algorithm': 'Thompson Sampling'})

    def report_cumulative_reward_and_regret(self, epsilon_greedy_rewards, thompson_rewards):
        """
        Generate a report of cumulative reward and regret.

        Args:
            epsilon_greedy_rewards (list): Rewards obtained from Epsilon-Greedy bandit.
            thompson_rewards (list): Rewards obtained from Thompson Sampling bandit.
        """
        cumulative_epsilon_greedy_reward = sum(epsilon_greedy_rewards)
        cumulative_thompson_reward = sum(thompson_rewards)
        cumulative_epsilon_greedy_regret = max(Bandit_Reward) * len(epsilon_greedy_rewards) - cumulative_epsilon_greedy_reward
        cumulative_thompson_regret = max(Bandit_Reward) * len(thompson_rewards) - cumulative_thompson_reward

        logger.info(f'Cumulative Reward - Epsilon-Greedy: {cumulative_epsilon_greedy_reward}')
        logger.info(f'Cumulative Reward - Thompson Sampling: {cumulative_thompson_reward}')
        logger.info(f'Cumulative Regret - Epsilon-Greedy: {cumulative_epsilon_greedy_regret}')
        logger.info(f'Cumulative Regret - Thompson Sampling: {cumulative_thompson_regret}')


if __name__ == "__main__":
    Bandit_Reward = [1, 2, 3, 4]
    num_trials = 20000

    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward)
    epsilon_greedy_rewards = epsilon_greedy_bandit.experiment(num_trials)

    thompson_bandit = ThompsonSampling(Bandit_Reward)
    thompson_rewards = thompson_bandit.experiment(num_trials)

    vis = Visualization()
    vis.plot_learning_process(epsilon_greedy_rewards, thompson_rewards)
    vis.plot_cumulative_rewards(epsilon_greedy_rewards, thompson_rewards)
    vis.store_rewards_to_csv(epsilon_greedy_rewards, thompson_rewards)
    vis.report_cumulative_reward_and_regret(epsilon_greedy_rewards, thompson_rewards)
