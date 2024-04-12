############################### LOGGER
import logging
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

logging.basicConfig
logger = logging.getLogger("MAB Application")

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

#--------------------------------------#

class Bandit(ABC):
    @abstractmethod
    def __init__(self, p):
        self.true_means = p  # True means of each arm
        self.estimated_means = [0.0] * len(p)  # Estimated means of each arm
        self.action_counts = [0] * len(p)  # Number of times each arm is pulled

    @abstractmethod
    def __repr__(self):
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

    @abstractmethod
    def report(self):
        pass

#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon=0.1):
        # Initialize the EpsilonGreedy bandit with parameters p and epsilon
        super().__init__(p)
        # Initialize epsilon parameter
        self.epsilon = epsilon

    def __repr__(self):
        # Return a string representation of the EpsilonGreedy bandit
        return f"EpsilonGreedy Bandit with epsilon={self.epsilon}"

    def pull(self):
        # Pull an arm using the Epsilon-Greedy strategy
        # If a random number is less than epsilon, explore (choose a random arm)
        if random.random() < self.epsilon:
            return random.randint(0, len(self.true_means) - 1)
        # Otherwise, exploit (choose the arm with the highest estimated mean)
        else:
            return np.argmax(self.estimated_means)

    def update(self, arm, reward):
        # Update the estimated mean of the chosen arm based on the received reward
        self.action_counts[arm] += 1  # Increment the count of the chosen arm
        n = self.action_counts[arm]     # Get the updated count of the chosen arm
        # Update the estimated mean of the chosen arm using the incremental formula
        self.estimated_means[arm] += (1 / n) * (reward - self.estimated_means[arm])

    def experiment(self, num_trials):
        # Run the Epsilon-Greedy experiment for a specified number of trials
        rewards = []
        for t in range(1, num_trials + 1):
            self.epsilon = 1 / t  # Decay epsilon over time
            arm = self.pull()      # Choose an arm using Epsilon-Greedy strategy
            reward = self.true_means[arm]  # Get the true reward for the chosen arm
            self.update(arm, reward)       # Update the bandit based on the received reward
            rewards.append(reward)          # Collect the reward for analysis
        return rewards

    def report(self):
        # Report the average reward and regret of the EpsilonGreedy bandit
        # Calculate the average reward using the estimated means
        avg_reward = sum(self.estimated_means) / len(self.estimated_means)
        # Calculate the average regret by comparing the maximum true mean with the average reward
        avg_regret = max(self.true_means) - avg_reward
        # Format the results into a string for reporting
        return f"EpsilonGreedy Results: Average Reward={avg_reward:.2f}, Average Regret={avg_regret:.2f}"

#--------------------------------------#

class ThompsonSampling(Bandit):
    def __init__(self, p):
        # Initialize the ThompsonSampling bandit with parameters p
        super().__init__(p)
        # Initialize alpha and beta parameters for each arm
        self.alpha = [1] * len(p)  # Successes
        self.beta = [1] * len(p)   # Failures

    def __repr__(self):
        # Return a string representation of the ThompsonSampling bandit
        return "ThompsonSampling Bandit"

    def pull(self):
        # Pull an arm using Thompson Sampling algorithm
        # Sample means from the beta distribution for each arm
        sampled_means = [random.betavariate(alpha, beta) for alpha, beta in zip(self.alpha, self.beta)]
        # Choose the arm with the highest sampled mean
        return np.argmax(sampled_means)

    def update(self, arm, reward):
        # Update the parameters of the chosen arm based on the received reward
        # If the reward is 1 (success), update alpha (successes) for the chosen arm
        if reward == 1:
            self.alpha[arm] += 1
        # If the reward is 0 (failure), update beta (failures) for the chosen arm
        else:
            self.beta[arm] += 1

    def experiment(self, num_trials):
        # Run the Thompson Sampling experiment for a specified number of trials
        rewards = []
        for _ in range(num_trials):
            # Pull an arm and get the reward
            arm = self.pull()
            reward = self.true_means[arm]  # Get the true reward for the chosen arm
            # Update the bandit based on the received reward
            self.update(arm, reward)
            # Collect the reward for analysis
            rewards.append(reward)
        return rewards

    def report(self):
        # Report the average reward and regret of the ThompsonSampling bandit
        # Calculate the average reward using the updated alpha and beta parameters
        avg_reward = sum(self.alpha) / (sum(self.alpha) + sum(self.beta))
        # Calculate the average regret by comparing the maximum true mean with the average reward
        avg_regret = max(self.true_means) - avg_reward
        # Format the results into a string for reporting
        return f"ThompsonSampling Results: Average Reward={avg_reward:.2f}, Average Regret={avg_regret:.2f}"


#--------------------------------------#

class Visualization:
    def plot_learning_process(self, epsilon_greedy_rewards, thompson_rewards):
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

#--------------------------------------#

def comparison(num_trials):
    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward)
    thompson_bandit = ThompsonSampling(Bandit_Reward)

    epsilon_greedy_rewards = epsilon_greedy_bandit.experiment(num_trials)
    thompson_rewards = thompson_bandit.experiment(num_trials)

    vis = Visualization()
    vis.plot_learning_process(epsilon_greedy_rewards, thompson_rewards)
    vis.plot_cumulative_rewards(epsilon_greedy_rewards, thompson_rewards)
    # vis.store_rewards_to_csv(epsilon_greedy_rewards, thompson_rewards)
    vis.report_cumulative_reward_and_regret(epsilon_greedy_rewards, thompson_rewards)

if __name__ == "__main__":
    Bandit_Reward = [1, 2, 3, 4]
    comparison(20000)
