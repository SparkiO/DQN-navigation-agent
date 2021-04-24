# Project I: Navigation
## Introduction
The project uses Deep Reinforcement Learning algorithm to train an agent to navigate and collect fruits in a large, square world. 
![](https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)

## Project Details
#### State-Space
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

-   **`0`**  - move forward.
-   **`1`**  - move backward.
-   **`2`**  - turn left.
-   **`3`**  - turn right.
#### Rewards and Completion
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.
## Getting Started & Instructions
1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)

- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Place the file in the in the root folder, and unzip (or decompress) the file.

3. The crucial dependencies used in the project are:
-- unityagents: instructions to install: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md
-- NumPy: Numerical Python library, pip installation command: 'pip install numpy'
-- Matplotlib: Python's basic plotting library, pip installation command: 'pip install matplotlib'

The code takes form of a Jupyter Notebook. To train the agent, just run each cell one by one from the top.
The weights and plotting figures are saved in appropriate files. 
