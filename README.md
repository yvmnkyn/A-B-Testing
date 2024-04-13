# A-B-Testing

This repository contains Python code implementing two popular algorithms: Epsilon-Greedy and Thompson Sampling. It also includes visualization and reporting functions to analyze the performance of these algorithms.

## Project Structure

The project is organized as follows:

- **codes:** This directory contains the Python scripts for A/B testing using Epsilon-Greedy and Thompson Sampling.
  - `Bandit.py`: Contains the implementation of bandit algorithms.
  - `logs.py`: Contains custom logging configuration.
  - Other Python scripts related to the A/B testing implementation.
- **.gitignore:** Specifies intentionally untracked files to ignore.
- **README.md:** This file you are currently reading.
- **requirements.txt:** Specifies the dependencies required to run the code.
- **results:** Contains the resulted .csv files.
- **docs:** Contains the Homework .pdf file

## Usage

1. **codes:** This directory contains the Python scripts for A/B testing using Epsilon-Greedy and Thompson Sampling.
    - `Bandit.py`: Contains the implementation of bandit algorithms.
    - `logs.py`: Contains custom logging configuration.
    - Other Python scripts related to the A/B testing implementation.
2. **.gitignore:** Specifies intentionally untracked files to ignore.
3. **README.md:** This file you are currently reading.
4. **requirements.txt:** Specifies the dependencies required to run the code.


## Instructions

1. Clone the repository:

    ```bash
    git clone <repository_url>
    ```

2. Navigate to the `codes` directory:

    ```bash
    cd codes
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the main script:

    ```bash
    python Bandit.py
    ```

## Suggestions 
Algorithm Parameter Tuning:
Explore techniques for automatic parameter tuning, such as grid search or Bayesian optimization, to optimize algorithm parameters like epsilon for Epsilon Greedy and precision for Thompson Sampling.
Implement a mechanism to dynamically adjust algorithm parameters during the experiment based on performance metrics and feedback.

