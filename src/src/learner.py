from collections import deque
from os import path, mkdir
import threading
import time
import math
import numpy as np
import pickle
import concurrent.futures
import random
from functools import reduce
import whr

import sys
sys.path.append('../build')
from library import MCTS, Gomoku, NeuralNetwork

from neural_network import NeuralNetWorkWrapper
from gomoku_gui import GomokuGUI
from game_interface import GameInterface

prev_match_results = []

def tuple_2d_to_numpy_2d(tuple_2d):
    # help function
    # convert type
    res = [None] * len(tuple_2d)
    for i, tuple_1d in enumerate(tuple_2d):
        res[i] = list(tuple_1d)
    return np.array(res)


class Leaner():
    def __init__(self, config):
        # see config.py
        # gomoku
        self.action_size = config['action_size']

        self.game_engine = GameInterface(config)

        # train
        self.num_iters = config['num_iters']
        self.num_eps = config['num_eps']
        self.num_train_threads = config['num_train_threads']
        self.check_freq = config['check_freq']
        self.num_contest = config['num_contest']
        self.update_threshold = config['update_threshold']
        self.num_explore = config['num_explore']
        self.use_simsiam = config['use_simsiam']

        self.examples_buffer = deque([], maxlen=config['examples_buffer_max_len'])

        # mcts
        self.num_mcts_threads = config['num_mcts_threads']
        self.libtorch_use_gpu = config['libtorch_use_gpu']

        # neural network
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.nnet = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                                         config['num_channels'], config['n'], self.action_size, config['use_simsiam'], config['simsiam_loss_factor'], config['train_use_gpu'], self.libtorch_use_gpu)


    def learn(self):
        # train the model by self play
        match_result = whr.Base(config={'w2': 30})
        current_model = 1
        for match in prev_match_results:
            current_model = match[0]
            for i in range(match[2]):
                match_result.create_game('player_'+'{}'.format(match[0]), 'player_'+'{}'.format(match[1]), 'B', match[5])
            for i in range(match[3]):
                match_result.create_game('player_'+'{}'.format(match[0]), 'player_'+'{}'.format(match[1]), 'W', match[5])
            for i in range(match[4]):
                match_result.create_game('player_'+'{}'.format(match[0]), 'player_'+'{}'.format(match[1]), 'D', match[5])
        if prev_match_results != []:
            fianl_match = prev_match_results[-1]
            if float(fianl_match[2]) / (fianl_match[2] + fianl_match[3]) > self.update_threshold:
                current_model += 1
        
        
        if path.exists(path.join('models', 'checkpoint.example')):
            print("loading checkpoint...")
            self.nnet.load_model()
            self.load_samples()
        else:
            # save torchscript
            self.nnet.save_model()
            self.nnet.save_model('models', "best_checkpoint")
        
        for itr in range(1, self.num_iters + 1):
            print("ITER :: {}".format(itr), time.time())

            # self play in parallel
            libtorch = NeuralNetwork('./models/checkpoint.pt',
                                     self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads)
            itr_examples = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_train_threads) as executor:
                futures = [executor.submit(self.game_engine.play, itr, 1 if itr % 2 else -1, libtorch, libtorch, -1, k == 1) for k in range(1, self.num_eps + 1)]
                for k, f in enumerate(futures):
                    examples, _, _ = f.result()
                    itr_examples += examples

                    # decrease libtorch batch size
                    remain = min(len(futures) - (k + 1), self.num_train_threads)
                    libtorch.set_batch_size(max(remain * self.num_mcts_threads, 1))
                    print("EPS: {}, EXAMPLES: {}".format(k + 1, len(examples)))

            # release gpu memory
            del libtorch

            # prepare train data
            self.examples_buffer.append(itr_examples)
            train_data = reduce(lambda a, b : a + b, self.examples_buffer)
            random.shuffle(train_data)

            # train neural network
            epochs = self.epochs * (len(itr_examples) + self.batch_size - 1) // self.batch_size
            self.nnet.train(train_data, self.batch_size, int(epochs))
            self.nnet.save_model()
            self.save_samples()

            # compare performance
            if itr % self.check_freq == 0:
                libtorch_current = NeuralNetwork('./models/checkpoint.pt',
                                         self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2)
                libtorch_best = NeuralNetwork('./models/best_checkpoint.pt',
                                              self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2)

                one_won, two_won, draws = self.game_engine.contest(libtorch_current, libtorch_best, self.num_contest)
                print(one_won, two_won, draws)
                for i in range(one_won):
                    match_result.create_game('player_'+'{}'.format(current_model), 'player_'+'{}'.format(current_model-1), 'B', itr)
                for i in range(two_won):
                    match_result.create_game('player_'+'{}'.format(current_model), 'player_'+'{}'.format(current_model-1), 'W', itr)  
                for i in range(draws):
                    match_result.create_game('player_'+'{}'.format(current_model), 'player_'+'{}'.format(current_model-1), 'D', itr)              
                match_result.iterate_until_converge(verbose=False)
                for i in match_result.get_ordered_ratings():
                    print(i)

                if one_won + two_won > 0 and float(one_won) / (one_won + two_won) > self.update_threshold:
                    print('ACCEPTING NEW MODEL')
                    self.nnet.save_model('models', "best_checkpoint")
                    current_model += 1
                else:
                    print('REJECTING NEW MODEL')

                # release gpu memory
                del libtorch_current
                del libtorch_best

    def load_samples(self, folder="models", filename="checkpoint.example"):
        """load self.examples_buffer
        """

        filepath = path.join(folder, filename)
        with open(filepath, 'rb') as f:
            self.examples_buffer = pickle.load(f)

    def save_samples(self, folder="models", filename="checkpoint.example"):
        """save self.examples_buffer
        """

        if not path.exists(folder):
            mkdir(folder)

        filepath = path.join(folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.examples_buffer, f, -1)
