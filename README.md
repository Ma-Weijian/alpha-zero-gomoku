# Gomoku Agent Trainers
A thread pool implementation of AlphaZero and AlphaStar.

Also have some data augmentation schemes on gomoku and implemented self-supervised learning methods like Simsiam on backbones of the policy-value networks. 

## Updates
### 2021.10
Added one line of code and removed ~45% of mutex lock requirements, brings about 20%+ speed improvements.
### 2022.4
Implemented a light-weight league training method which suits limited training hardware environments (such as PCs having single GPU).
### 2022.5
Added Simsiam on training backbones of policy-value networks.

## Features
* Easy Free-style Gomoku with no specific limitations
* Tree/Root Parallelization with Virtual Loss and LibTorch
* Gomoku and MCTS are written in C++
* SWIG for Python C++ extension

## Args
Edit config.py for everyting except training paradigms.

## Packages

* Python 3.7
* PyTorch 1.11.0
* LibTorch 1.11.0
* SWIG 4.0.1
* CMake 3.16+
* GCC 9.4.0+
* Others please refer to requirements.txt

## Run
```
# Add LibTorch/SWIG to environment variable $PATH

# Compile Python extension
# 注意这边需要在find\_package(Torch REQUIRED)前面加上链接到你的conda中torch的CMAKE\_PREFIX\_PATH.
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=path/to/libtorch -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc" -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Run
cd ..
python run_agent.py train               # train model via self-play
python run_agent.py league_train        # train model via league training
python run_agent.py play                # play with human
```

## GUI
Agent first.

![](https://github.com/Ma-Weijian/gomoku-agent-trainer/blob/master/assets/gomoku_gui.png)

## References
1. Mastering the Game of Go without Human Knowledge
2. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
3. Parallel Monte-Carlo Tree Search
4. An Analysis of Virtual Loss in Parallel MCTS
5. A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm
6. github.com/suragnair/alpha-zero-general
7. Exploring Simple Siamese Representation Learning
