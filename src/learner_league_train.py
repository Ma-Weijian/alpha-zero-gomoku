from collections import deque
from nis import match
from os import path, mkdir
import threading
import time
import math
import numpy as np
import pickle
import concurrent.futures
import random
from functools import reduce
import math
import os

import sys
sys.path.append('../build')
from library import MCTS, Gomoku, NeuralNetwork

from neural_network import NeuralNetWorkWrapper
from gomoku_gui import GomokuGUI
from gomoku_rating_utils import GomokuTracker

prev_match_results = []

def tuple_2d_to_numpy_2d(tuple_2d):
    # help function
    # convert type
    res = [None] * len(tuple_2d)
    for i, tuple_1d in enumerate(tuple_2d):
        res[i] = list(tuple_1d)
    return np.array(res)


class League_Trainer():
    def __init__(self, config):
        # see config.py
        # gomoku
        self.n = config['n']
        self.n_in_row = config['n_in_row']
        self.gomoku_gui = GomokuGUI(config['n'], config['human_color'])
        self.action_size = config['action_size']

        # train
        self.num_iters = config['num_iters']
        self.num_eps = config['num_eps']
        self.num_train_threads = config['num_train_threads']
        self.check_freq = config['check_freq']
        self.num_contest = config['num_contest']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.temp = config['temp']
        self.update_threshold = config['update_threshold']
        self.num_explore = config['num_explore']
        self.noise_min = config['noise_min']
        self.noise_max = config['noise_max']

        self.examples_buffer_main_agent = deque([], maxlen=config['examples_buffer_max_len'])
        # These two buffers should be cleaned when a checkpoint is saved
        self.examples_buffer_main_exploiter = deque([], maxlen=config['examples_buffer_max_len'])
        self.examples_buffer_league_exploiter = deque([], maxlen=config['examples_buffer_max_len'])

        # train with simsiam
        self.simsiam_move_rate = config['simsiam_move_rate']
        self.simsiam_turn_rate = config['simsiam_turn_rate']
        self.simsiam_flip_rate = config['simsiam_flip_rate']
        

        # mcts
        self.num_mcts_sims = config['num_mcts_sims']
        self.c_puct = config['c_puct']
        self.c_virtual_loss = config['c_virtual_loss']
        self.num_mcts_threads = config['num_mcts_threads']
        self.libtorch_use_gpu = config['libtorch_use_gpu']

        # neural network
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.nnet_main_agent = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                         config['num_channels'], config['n'], self.action_size, config['use_simsiam'], config['simsiam_loss_factor'], config['train_use_gpu'], self.libtorch_use_gpu)
        self.nnet_main_exploiter = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                         config['num_channels'], config['n'], self.action_size, config['use_simsiam'], config['simsiam_loss_factor'], config['train_use_gpu'], self.libtorch_use_gpu)
        self.nnet_league_exploiter = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                         config['num_channels'], config['n'], self.action_size, config['use_simsiam'], config['simsiam_loss_factor'], config['train_use_gpu'], self.libtorch_use_gpu)

        self.num_warmup = config['num_warmup']
        self.main_agent_selfplay_rate = config['main_agent_selfplay_rate']
        self.main_agent_pfsp_rate = config['main_agent_pfsp_rate'] 

        self.main_agent_discard_rate = config['main_agent_discard_rate']

        self.apply_turn = config['apply_turn']

        self.main_agent_models = []
        self.discarded_main_agent = []
        self.league_exploiter_models = []
        self.main_exploiter_models = []

        if not os.path.exists('./models'):
            os.mkdir('./models')
        self.main_agent_dir = os.path.join('./models', 'main_agent')
        if not os.path.exists(self.main_agent_dir):
            os.mkdir(self.main_agent_dir)
        self.league_exploiter_dir = os.path.join('./models', 'league_exploiter')
        if not os.path.exists(self.league_exploiter_dir):
            os.mkdir(self.league_exploiter_dir)
        self.main_exploiter_dir = os.path.join('./models', 'main_exploiter')
        if not os.path.exists(self.main_exploiter_dir):
            os.mkdir(self.main_exploiter_dir)      

        self.w2 = config['w2']

        self.gomoku_tracker = GomokuTracker(self.w2, self.update_threshold)

        # start gui
        t = threading.Thread(target=self.gomoku_gui.loop)
        t.start()
    
    def init_models(self):
        if path.exists(path.join('models', 'checkpoint.example')):
            print("loading checkpoint...")
            self.nnet_main_agent.load_model()
            self.load_samples()
        else:
            # save torchscript
            self.nnet_main_agent.save_model()
    
    def selfplay_warmup(self):
        self.init_models()

        for itr in range(1, self.num_warmup + 1):
            print("ITER :: {}".format(itr), time.time())

            # self play in parallel
            # This is the main agent.
            libtorch = NeuralNetwork('./models/checkpoint.pt',
                                     self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads)
            itr_examples = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_train_threads) as executor:
                futures = [executor.submit(self.self_play, itr, 1 if itr % 2 else -1, libtorch, libtorch, k, k == 1) for k in range(1, self.num_eps + 1)]
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
            self.examples_buffer_main_agent.append(itr_examples)
            train_data = reduce(lambda a, b : a + b, self.examples_buffer_main_agent)
            random.shuffle(train_data)

            # train neural network
            epochs = self.epochs * (len(itr_examples) + self.batch_size - 1) // self.batch_size
            self.nnet_main_agent.train(train_data, self.batch_size, int(epochs))
            self.nnet_main_agent.save_model()
            self.save_samples(buffer=self.examples_buffer_main_agent)
    
    def init_league_models(self):
        self.nnet_main_agent.save_model(folder=self.main_agent_dir, filename="main_ckpt_"+"{}".format(self.num_warmup))
        self.save_samples(buffer=self.examples_buffer_main_agent, folder=self.main_agent_dir, filename="main_ckpt_"+"{}".format(self.num_warmup)+".example")
        self.main_agent_models.append("main_ckpt_"+"{}".format(self.num_warmup))

        # init main exploiter and league exploiter as the main agent
        self.nnet_main_agent.save_model(folder=self.main_exploiter_dir, filename="main_exploit_ckpt_"+"{}".format(self.num_warmup))
        self.save_samples(buffer=self.examples_buffer_main_agent, folder=self.main_exploiter_dir, filename="main_exploit_ckpt_"+"{}".format(self.num_warmup)+".example")
        self.main_exploiter_models.append("main_exploit_ckpt_"+"{}".format(self.num_warmup))
        self.nnet_main_agent.save_model(folder=self.league_exploiter_dir, filename="league_exploit_ckpt_"+"{}".format(self.num_warmup))
        self.save_samples(buffer=self.examples_buffer_main_agent, folder=self.league_exploiter_dir, filename="league_exploit_ckpt_"+"{}".format(self.num_warmup)+".example")
        self.league_exploiter_models.append("league_exploit_ckpt_"+"{}".format(self.num_warmup))
    
    def select_agents_in_league(self, itr):
        id_dict = {}

        # decide main agent's opponent
        # print("main_ckpt_"+"{}".format((itr-1)-(itr-1)%self.check_freq))
        main_agent_id = "main_ckpt_"+"{}".format((itr-1)-(itr-1)%self.check_freq)
        main_agent_path = os.path.join(self.main_agent_dir, main_agent_id+".pt")
        self.nnet_main_agent.load_model(self.main_agent_dir, main_agent_id)
        id_dict["main_agent_id"] = main_agent_id
        # print(1111111)
        libtorch_main_agent = NeuralNetwork(main_agent_path,
                                     self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2)
        dice = random.random()
        libtorch_main_opponent = None
        if dice < self.main_agent_selfplay_rate:
            # print(222222)
            main_opponent_id = main_agent_id
            libtorch_main_opponent = NeuralNetwork(main_agent_path,
                                     self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2)
        elif self.main_agent_selfplay_rate < dice and dice < self.main_agent_pfsp_rate:
            main_opponent_id = self.gomoku_tracker.find_opponent_hard(itr, main_agent_id, self.main_agent_models)
            main_opponent_path = os.path.join(self.main_agent_dir, main_opponent_id+'.pt')
            # print(333333)
            libtorch_main_opponent = NeuralNetwork(main_opponent_path,
                                 self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2)
        else:
            opponent_list = self.discarded_main_agent + self.main_exploiter_models
            main_opponent_id = random.sample(opponent_list, 1)[0]
            main_opponent_path = None
            if main_opponent_id in self.discarded_main_agent:
                main_opponent_path = os.path.join(self.main_agent_dir, main_opponent_id+'.pt')
            else:
                main_opponent_path = os.path.join(self.main_exploiter_dir, main_opponent_id+'.pt')
                
            # print(444444)
            libtorch_main_opponent = NeuralNetwork(main_opponent_path,
                                     self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2)   
        id_dict["main_opponent_id"] = main_opponent_id            
            
        # train league exploiters
        league_id = random.sample(self.league_exploiter_models, 1)[0]
        league_opponents = self.main_agent_models+self.main_exploiter_models
        league_opponent_id = self.gomoku_tracker.find_opponent_hard(itr, league_id, league_opponents)
        league_path = os.path.join(self.league_exploiter_dir, league_id+'.pt')
        league_opponent_path = None
        if league_opponent_id in self.main_exploiter_models:
            league_opponent_path = os.path.join(self.main_exploiter_dir, league_opponent_id+'.pt')
            self.nnet_league_exploiter.load_model(self.main_exploiter_dir, league_opponent_id)
        else:
            league_opponent_path = os.path.join(self.main_agent_dir, league_opponent_id+'.pt')
            self.nnet_league_exploiter.load_model(self.main_agent_dir, league_opponent_id)

        # print(555555)
        libtorch_league = NeuralNetwork(league_path,
                                     self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2)
        # print(666666)            
        libtorch_league_opponent = NeuralNetwork(league_opponent_path,
                                     self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2) 
        id_dict['league_id'], id_dict['league_opponent_id'] = league_id, league_opponent_id
            
        # train_main_exploiters
        main_exploiter_id = random.sample(self.main_exploiter_models, 1)[0]
        main_exploiter_path = os.path.join(self.main_exploiter_dir, main_exploiter_id+'.pt')
        self.nnet_main_exploiter.load_model(self.main_exploiter_dir, main_exploiter_id)
        # print(777777)
        libtorch_main_exploiter = NeuralNetwork(main_exploiter_path,
                                     self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2)
        libtorch_exploiter_rival = None
        exploit_opponent_id = None
        if random.random() < 0.5 and self.gomoku_tracker.calc_beat_rate(itr, main_exploiter_id, main_agent_id) > 0.2:
            libtorch_exploiter_rival = libtorch_main_agent
            exploit_opponent_id = main_agent_id
        else:
            exploit_opponent_id = self.gomoku_tracker.find_opponent_rel(itr, main_exploiter_id, self.main_agent_models)
            exploit_opponent_path = os.path.join(self.main_agent_dir, exploit_opponent_id+'.pt')
            # print(888888)
            libtorch_exploiter_rival = NeuralNetwork(exploit_opponent_path,
                                     self.libtorch_use_gpu, self.num_mcts_threads * self.num_train_threads // 2) 
        id_dict['main_exploiter_id'], id_dict['exploit_opponent_id'] = main_exploiter_id, exploit_opponent_id

        contest_list = [(libtorch_main_agent, libtorch_main_opponent), (libtorch_main_opponent, libtorch_main_agent),
                (libtorch_league, libtorch_league_opponent), (libtorch_league_opponent, libtorch_league),
                (libtorch_main_exploiter, libtorch_exploiter_rival), (libtorch_exploiter_rival, libtorch_main_exploiter)]
        
        return contest_list, id_dict

    def shuffle_data_and_train(self, itr, id_dict, itr_examples_all):
        # prepare train data
        itr_examples_main_agent = itr_examples_all[0]
        itr_examples_main_exploiter = itr_examples_all[1]
        itr_examples_league_exploiter = itr_examples_all[2]
        self.examples_buffer_main_agent.append(itr_examples_main_agent)
        self.examples_buffer_league_exploiter.append(itr_examples_league_exploiter)
        self.examples_buffer_main_exploiter.append(itr_examples_main_exploiter)
        train_data_main_agent = reduce(lambda a, b : a + b, self.examples_buffer_main_agent)
        train_data_league_exploiter = reduce(lambda a, b : a + b, self.examples_buffer_league_exploiter)
        train_data_main_exploiter = reduce(lambda a, b : a + b, self.examples_buffer_main_exploiter)
        random.shuffle(train_data_main_agent)
        random.shuffle(train_data_main_exploiter)
        random.shuffle(train_data_league_exploiter)

        # train neural network
        epochs_main_agent = self.epochs * (len(itr_examples_main_agent) + self.batch_size - 1) // self.batch_size
        epochs_main_exploiter = self.epochs * (len(itr_examples_main_exploiter) + self.batch_size - 1) // self.batch_size
        epochs_league_exploiter = self.epochs * (len(itr_examples_league_exploiter) + self.batch_size - 1) // self.batch_size
        print("Training main agent.")
        self.nnet_main_agent.train(train_data_main_agent, self.batch_size, int(epochs_main_agent))
        print("Training main exploiter.")
        self.nnet_main_exploiter.train(train_data_main_exploiter, self.batch_size, int(epochs_main_exploiter))
        print("Training league exploiter.")
        self.nnet_league_exploiter.train(train_data_league_exploiter, self.batch_size, int(epochs_league_exploiter))
        self.nnet_main_agent.save_model(self.main_agent_dir, "main_ckpt_"+"{}".format(itr-itr%self.check_freq))
        self.nnet_main_exploiter.save_model(self.main_exploiter_dir, id_dict["main_exploiter_id"])
        self.nnet_league_exploiter.save_model(self.league_exploiter_dir, id_dict["league_id"])

        self.save_samples(self.examples_buffer_main_agent, self.main_agent_dir, "main_ckpt_"+"{}".format(itr-itr%self.check_freq)+'.example')
        self.save_samples(self.examples_buffer_main_agent, self.main_exploiter_dir, id_dict["main_exploiter_id"]+'.example')
        self.save_samples(self.examples_buffer_main_agent, self.league_exploiter_dir, id_dict["league_id"]+'.example')

    def update_league_members(self, itr, id_dict):
        # Create virtual match to show that the two main models are equivalent
        if itr % self.check_freq == 0:
            self.main_agent_models.append(id_dict["main_agent_id"])
            self.gomoku_tracker.create_game("main_ckpt_"+"{}".format(itr-itr%self.check_freq), id_dict["main_agent_id"], "D", itr)

        # judge whether main exploiters should be added to the queue
        if itr % (2*self.check_freq) == 0:
            self.nnet_main_exploiter.save_model(self.main_exploiter_dir, "main_exploit_ckpt_"+"{}".format(itr-itr%self.check_freq))
            self.main_exploiter_models.append("main_exploit_ckpt_"+"{}".format(itr-itr%self.check_freq))
            self.gomoku_tracker.create_game("main_exploit_ckpt_"+"{}".format(itr-itr%self.check_freq),id_dict["main_exploiter_id"], "D", itr)
        else:
            for i in self.main_agent_models:
                if id_dict["main_exploiter_id"] != i and self.gomoku_tracker.calc_beat_rate(itr,  id_dict["main_exploiter_id"], i) < 0.7:
                    break
            else:
                self.nnet_main_exploiter.save_model(self.main_exploiter_dir, "main_exploit_ckpt_"+"{}".format(itr))
                self.main_exploiter_models.append("main_exploit_ckpt_"+"{}".format(itr))
                self.gomoku_tracker.create_game("main_exploit_ckpt_"+"{}".format(itr-itr%self.check_freq),id_dict["main_exploiter_id"], "D", itr)


        # judge whether league exploiters should be added to the queue
        if itr % (2*self.check_freq) == 0:
            self.nnet_league_exploiter.save_model(self.league_exploiter_dir, "league_exploit_ckpt_"+"{}".format(itr-itr%self.check_freq))
            self.league_exploiter_models.append("league_exploit_ckpt_"+"{}".format(itr-itr%self.check_freq))
            self.gomoku_tracker.create_game("league_exploit_ckpt_"+"{}".format(itr-itr%self.check_freq),id_dict["league_id"], "D", itr)
        else:
            for i in self.main_agent_models + self.league_exploiter_models:
                if id_dict["main_exploiter_id"] != i and self.gomoku_tracker.calc_beat_rate(itr, id_dict["main_exploiter_id"], i) < 0.7:
                    break
            else:
                self.nnet_league_exploiter.save_model(self.league_exploiter_dir, "league_exploit_ckpt_"+"{}".format(itr))
                self.league_exploiter_models.append("league_exploit_ckpt_"+"{}".format(itr))
                self.gomoku_tracker.create_game("league_exploit_ckpt_"+"{}".format(itr),id_dict["league_id"], "D", itr)

            
        # remove discarded main agents
        for i in self.main_agent_models:
            print(id_dict["main_agent_id"], i, self.gomoku_tracker.calc_beat_rate(itr, id_dict["main_agent_id"], i))
            if self.gomoku_tracker.calc_beat_rate(itr, id_dict["main_agent_id"], i) > self.main_agent_discard_rate:
                self.main_agent_models.remove(i)
                self.discarded_main_agent.append(i)

    def get_exploration_rate(self, iter):
        expl_rate = self.noise_min + 0.5 * (self.noise_max-self.noise_min) * (1+math.cos(iter/self.check_freq * math.pi))
        return expl_rate

    def league_train(self):
        self.gomoku_tracker.init_result_trackers(prev_match_results)
        
        # init every model in main agent, main exploiter and league exploiter 
        # as the model pretrained in 20 iterations
        self.selfplay_warmup()

        # init the warmup neural network as the main agent, main exploiter and league exploiter
        self.init_league_models()

        # write some virtual contest results and get the original gomoku status
        self.gomoku_tracker.init_league_results(self.num_warmup)

        # print(self.main_agent_models)
        # print(self.main_exploiter_models)
        # print(self.league_exploiter_models)

        # real league training
        for itr in range(self.num_warmup + 1, self.num_iters + 1):
            print("ITER :: {}".format(itr), time.time())

            contest_list, id_dict = self.select_agents_in_league(itr)

            itr_examples_main_agent = []
            itr_examples_main_exploiter = []
            itr_examples_league_exploiter = []
            # self play all the agents in parallel


            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_train_threads) as executor:
                futures = []
                assert self.num_eps % len(contest_list) == 0
                for i in range(self.num_eps):
                    futures.append(
                        executor.submit(self.self_play, itr, 1, contest_list[i%len(contest_list)][0], contest_list[i%len(contest_list)][1], i%len(contest_list), i == 1)
                    )
                for k, f in enumerate(futures):
                    examples, index, winner = f.result()
                    if index == 0 or index == 1:
                        itr_examples_main_agent += examples
                        print("EPS: {}, main agent,  EXAMPLES: {}".format(k + 1, len(examples)))
                        # do not record self play
                        if id_dict["main_agent_id"]  != id_dict["main_opponent_id"]:
                            if winner == 0:
                                self.gomoku_tracker.create_game(id_dict["main_agent_id"], id_dict["main_opponent_id"], "D", itr)
                                print(id_dict["main_agent_id"], id_dict["main_opponent_id"], "D", itr)
                            elif winner == 1 and index == 0 or winner == -1 and index == 1:
                                self.gomoku_tracker.create_game(id_dict["main_agent_id"], id_dict["main_opponent_id"], "B", itr)
                                print(id_dict["main_agent_id"], id_dict["main_opponent_id"], "B", itr)
                            else:
                                self.gomoku_tracker.create_game(id_dict["main_agent_id"], id_dict["main_opponent_id"], "W", itr)
                                print(id_dict["main_agent_id"], id_dict["main_opponent_id"], "W", itr)
                    elif index == 2 or index == 3:
                        itr_examples_league_exploiter += examples
                        print("EPS: {}, league exploiter,  EXAMPLES: {}".format(k + 1, len(examples)))  
                        # do not record self play
                        if id_dict["league_id"] != id_dict["league_opponent_id"]:
                            if winner == 0:
                                self.gomoku_tracker.create_game(id_dict["league_id"], id_dict["league_opponent_id"], "D", itr)
                                print(id_dict["league_id"], id_dict["league_opponent_id"], "D", itr)
                            elif winner == 1 and index == 0 or winner == -1 and index == 1:
                                self.gomoku_tracker.create_game(id_dict["league_id"], id_dict["league_opponent_id"], "B", itr)
                                print(id_dict["league_id"], id_dict["league_opponent_id"], "B", itr)
                            else:
                                self.gomoku_tracker.create_game(id_dict["league_id"], id_dict["league_opponent_id"], "W", itr)
                                print(id_dict["league_id"], id_dict["league_opponent_id"], "W", itr)
                    else:                      
                        itr_examples_main_exploiter += examples
                        print("EPS: {}, main exploiter,  EXAMPLES: {}".format(k + 1, len(examples)))
                        # do not record self play
                        if id_dict["main_exploiter_id"] != id_dict["exploit_opponent_id"]:
                            if winner == 0:
                                self.gomoku_tracker.create_game(id_dict["main_exploiter_id"], id_dict["exploit_opponent_id"], "D", itr)
                                print(id_dict["main_exploiter_id"], id_dict["exploit_opponent_id"], "D", itr)
                            elif winner == 1 and index == 0 or winner == -1 and index == 1:
                                self.gomoku_tracker.create_game(id_dict["main_exploiter_id"], id_dict["exploit_opponent_id"], "B", itr)
                                print(id_dict["main_exploiter_id"], id_dict["exploit_opponent_id"], "B", itr)
                            else:
                                self.gomoku_tracker.create_game(id_dict["main_exploiter_id"], id_dict["exploit_opponent_id"], "W", itr)
                                print(id_dict["main_exploiter_id"], id_dict["exploit_opponent_id"], "W", itr)

            self.gomoku_tracker.iterate_until_converge(verbose=False)

            # release gpu memory
            for i in contest_list:
                for j in i:
                    del j
            del contest_list
            '''
            if libtorch_exploiter_rival != libtorch_main_agent:
                del libtorch_exploiter_rival
            del libtorch_main_agent
            del libtorch_main_opponent
            del libtorch_league
            del libtorch_league_opponent
            del libtorch_main_exploiter
            '''
            itr_examples_all = [itr_examples_main_agent, itr_examples_main_exploiter, itr_examples_league_exploiter]
            self.shuffle_data_and_train(itr, id_dict, itr_examples_all)

            # check if we should add new members to the league or remove someone from the main agents 
            self.update_league_members(itr, id_dict)

            for i in self.gomoku_tracker.get_ordered_ratings():
                print(i)


            
    def self_play(self, iter, first_color, libtorch_player1, libtorch_player2, index, show):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.
        """
        start_time = time.time()
        train_examples = []

        player1 = MCTS(libtorch_player1, self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size)
        player2 = MCTS(libtorch_player2, self.num_mcts_threads, self.c_puct,
            self.num_mcts_sims, self.c_virtual_loss, self.action_size)
        players = [player2, None, player1]
        player_index = 1

        gomoku = Gomoku(self.n, self.n_in_row, first_color)

        if show:
            self.gomoku_gui.reset_status()

        episode_step = 0
        while True:
            episode_step += 1
            player = players[player_index + 1]

            # get action prob
            if episode_step <= self.num_explore:
                prob = np.array(list(player.get_action_probs(gomoku, self.temp)))
            else:
                prob = np.array(list(player.get_action_probs(gomoku, 0)))

            # generate sample
            board = tuple_2d_to_numpy_2d(gomoku.get_board())
            last_action = gomoku.get_last_move()
            cur_player = gomoku.get_current_color()

            train_examples.append([board, prob, last_action, cur_player])

            # dirichlet noise
            legal_moves = list(gomoku.get_legal_moves())
            noise = self.get_exploration_rate(iter) * np.random.dirichlet(self.dirichlet_alpha * np.ones(np.count_nonzero(legal_moves)))

            prob = (1-self.get_exploration_rate(iter)) * prob
            j = 0
            for i in range(len(prob)):
                if legal_moves[i] == 1:
                    prob[i] += noise[j]
                    j += 1
            prob /= np.sum(prob)

            # execute move
            action = np.random.choice(len(prob), p=prob)

            if show:
                self.gomoku_gui.execute_move(cur_player, action)
            gomoku.execute_move(action)
            player1.update_with_move(action)
            player2.update_with_move(action)

            # next player
            player_index = -player_index

            # is ended
            ended, winner = gomoku.get_game_status()
            if ended == 1:
                # b, last_action, cur_player, p, v
                # print("A game ends, duration:", time.time()-start_time)
                final_board, final_prob, final_action, _ = train_examples[-1]
                dir = self.get_aug_scale(final_board, final_prob, final_action)
                turn_45 = self.apply_turn
                if turn_45:
                    if self.n - dir[0] - dir[1] <= self.n // 2 + 1 and self.n - dir[2] - dir[3] <= self.n // 2 + 1:  
                        turn_45 = (random.random() < 0.1)
                        if turn_45:
                            turned_board, turned_prob, turned_action = self.turn(final_board, final_prob, final_action, dir)
                            turned_dir = self.get_aug_scale(turned_board, turned_prob, turned_action)
                    else:
                        turn_45 = False

                augmented_examples = []
                # print(dir, len(train_examples))
                for board, prob, last_action, temp_player in train_examples:
                    if not turn_45 or last_action == -1:
                        res = self.get_move_aug(board, prob, last_action, temp_player, dir)
                    else:
                        turned_board, turned_prob, turned_action = self.turn(board, prob, last_action, dir)
                        # print(turned_board, turned_prob, turned_action, turned_dir)
                        res = self.get_move_aug(turned_board, turned_prob, turned_action, temp_player, turned_dir)
                    # print(len(res), len(augmented_examples))
                    for moved_board, moved_prob, moved_last_act, res_player in res:
                        sym = self.get_symmetries(moved_board, moved_prob, moved_last_act)
                        for b, p, a in sym:
                            simsiam_augmented_board, simsiam_augmented_action = self.simsiam_augment(b, a)
                            augmented_examples.append([b, simsiam_augmented_board, a, simsiam_augmented_action, res_player, p])
                # print(len(augmented_examples))

                # b, simsiam_augmented_board, a, simsiam_augmented_action, res_player, p, reward = res_player*winner
                return [(x[0], x[1], x[2], x[3], x[4], x[5], x[4] * winner) for x in augmented_examples], index, winner

    def contest(self, network1, network2, num_contest):
        """compare new and old model
           Args: player1, player2 is neural network
           Return: one_won, two_won, draws
        """
        one_won, two_won, draws = 0, 0, 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_train_threads) as executor:
            futures = [executor.submit(\
                self._contest, network1, network2, 1 if k <= num_contest // 2 else -1, k == 1) for k in range(1, num_contest + 1)]
            for f in futures:
                winner = f.result()
                if winner == 1:
                    one_won += 1
                elif winner == -1:
                    two_won += 1
                else:
                    draws += 1

        return one_won, two_won, draws

    def _contest(self, network1, network2, first_player, show):
        # create MCTS
        player1 = MCTS(network1, self.num_mcts_threads, self.c_puct,
            self.num_mcts_sims, self.c_virtual_loss, self.action_size)
        player2 = MCTS(network2, self.num_mcts_threads, self.c_puct,
                    self.num_mcts_sims, self.c_virtual_loss, self.action_size)

        # prepare
        players = [player2, None, player1]
        player_index = first_player
        gomoku = Gomoku(self.n, self.n_in_row, first_player)
        if show:
            self.gomoku_gui.reset_status()

        # play
        while True:
            player = players[player_index + 1]

            # select best move
            prob = player.get_action_probs(gomoku)
            best_move = int(np.argmax(np.array(list(prob))))

            # execute move
            gomoku.execute_move(best_move)
            if show:
                self.gomoku_gui.execute_move(player_index, best_move)

            # check game status
            ended, winner = gomoku.get_game_status()
            if ended == 1:
                return winner

            # update search tree
            player1.update_with_move(best_move)
            player2.update_with_move(best_move)

            # next player
            player_index = -player_index
    
    def simsiam_augment(self, board, last_action):
        dice_move = random.random()
        fake_pi = np.zeros(self.n * self.n)
        fake_player = 1
        if dice_move < self.simsiam_move_rate:
            dir = self.get_aug_scale(board, fake_pi, last_action)
            temp_aug = self.get_move_aug(board, fake_pi, last_action, fake_player, dir)
            choice = random.sample(temp_aug, 1)[0]
            board, last_action = choice[0], choice[2]
        
        dice_turn, dice_flip = random.random(), random.random()
        if dice_turn < self.simsiam_turn_rate or dice_flip < self.simsiam_flip_rate:
            temp_turn = self.get_symmetries(board, fake_pi, last_action)
            idx = -1
            if not dice_turn < self.simsiam_turn_rate:
                # it must has been fliped here
                idx = 7
            else:
                idx = random.randint(0, 2)
                if dice_flip < self.simsiam_flip_rate:
                    idx += 4
            choice = temp_turn[idx]
            board, last_action = choice[0], choice[2]


        return board, last_action

    def get_symmetries(self, board, pi, last_action):
        # mirror, rotational
        assert(len(pi) == self.action_size)  # 1 for pass

        # print(last_action)
        pi_board = np.reshape(pi, (self.n, self.n))
        last_action_board = np.zeros((self.n, self.n))
        # print(board)
        # print(pi)
        # print(last_action, last_action // self.n, last_action % self.n)
        last_action_board[last_action // self.n][last_action % self.n] = 1
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                newAction = np.rot90(last_action_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                    newAction = np.fliplr(newAction)
                l += [(newB, newPi.ravel(), np.argmax(newAction) if last_action != -1 else -1)]
        return l

    def get_aug_scale(self, board, pi, last_action):
        if last_action == -1:
            return [0, 0, 0, 0]
        assert(len(pi) == self.action_size)  # 1 for pass
        pi = pi.reshape(self.n, self.n)
        # print(board)
        # print(pi)
        # dir: up down, left, right        
        # threshold = self.n_in_row-1
        threshold = 0
        # print(board.shape, pi.shape, last_action)

        last_x, last_y = last_action // self.n, last_action % self.n
        act_dir = [max(0,last_x-threshold), max(0,self.n-1-threshold-last_x), max(0,last_y-threshold), max(0,self.n-1-threshold-last_y)]
        # print(last_action, act_dir)
        # print(act_dir)
        board_dir = []
        for i in range(len(board)):
            if not (np.count_nonzero(board[i]) == 0):
                board_dir.append(max(0,i-threshold))
                break
        for i in range(len(board)-1, -1, -1):
            if not (np.count_nonzero(board[i]) == 0):
                board_dir.append(max(0,len(board)-1-i-threshold))
                break
        board_trans = np.transpose(board)
        for i in range(len(board_trans)):
            if not (np.count_nonzero(board_trans[i]) == 0):
                board_dir.append(max(0,i-threshold))
                break
        for i in range(len(board_trans)-1, -1, -1):
            if not (np.count_nonzero(board_trans[i]) == 0):
                board_dir.append(max(0,len(board_trans)-1-i-threshold))
                break
        pi_dir = []
        for i in range(len(pi)):
            if not (np.count_nonzero(pi[i]) == 0):
                pi_dir.append(max(0,i-threshold))
                break
        else:
            pi_dir.append(len(pi))
        for i in range(len(pi)-1, -1, -1):
            if not (np.count_nonzero(pi[i]) == 0):
                pi_dir.append(max(0,len(pi)-1-i-threshold))
                break
        else:
            pi_dir.append(len(pi))
        pi_trans = np.transpose(pi)
        for i in range(len(pi_trans)):
            if not (np.count_nonzero(pi_trans[i]) == 0):
                pi_dir.append(max(0,i-threshold))
                break
        else:
            pi_dir.append(len(pi))
        for i in range(len(pi_trans)-1, -1, -1):
            if not (np.count_nonzero(pi_trans[i]) == 0):
                pi_dir.append(max(0,len(pi_trans)-1-i-threshold))
                break
        else:
            pi_dir.append(len(pi))
        
        # print(board_dir, pi_dir)
        dir = []
        for i in range(len(act_dir)):
            dir.append(min(act_dir[i], board_dir[i], pi_dir[i]))
        # print(dir)
        return dir

    def turn(self, board, pi, last_action, dir):
        pi = pi.reshape(self.n, self.n)
        offset = self.n // 2
        temp_board = np.zeros([self.n // 2 + 1, self.n // 2 + 1])
        temp_pi = np.zeros([self.n // 2 + 1, self.n // 2 + 1])
        # print(temp_board.shape, temp_pi.shape)
        # print(self.n-dir[0]-dir[1], self.n-dir[2]-dir[3], dir)
        temp_board[:self.n-dir[0]-dir[1], :self.n-dir[2]-dir[3]] = board[dir[0]:self.n-dir[1], dir[2]:self.n-dir[3]]
        temp_pi[:self.n-dir[0]-dir[1], :self.n-dir[2]-dir[3]] = pi[dir[0]:self.n-dir[1], dir[2]:self.n-dir[3]]


        turned_board = np.zeros([self.n, self.n])
        turned_pi = np.zeros([self.n, self.n])

        for i in range(len(temp_board)):
            for j in range(len(temp_board)):
                new_i, new_j = i + offset -j, i + j
                turned_board[new_i][new_j] = temp_board[i][j]
                turned_pi[new_i][new_j] = temp_pi[i][j]
        turned_pi = turned_pi.reshape(self.n * self.n)

        last_x, last_y = last_action // self.n, last_action % self.n
        x, y = last_x - dir[0], last_y - dir[2]
        turned_x, turned_y = x + offset - y, x + y 
        action = turned_x * self.n + turned_y

        return turned_board, turned_pi, action

    def get_move_aug(self, board, pi, last_action, player, dir):
        if last_action == -1:
            return [(board, pi, last_action, player)]
        assert(len(pi) == self.action_size)  # 1 for pass
        pi = pi.reshape(self.n, self.n)

        last_x, last_y = last_action // self.n, last_action % self.n
        # print(dir)
        useful_board = board[dir[0]:self.n-dir[1], dir[2]:self.n-dir[3]]
        useful_pi = pi[dir[0]:self.n-dir[1], dir[2]:self.n-dir[3]]
        # print(useful_board.shape, useful_pi.shape, dir)
        updown_dir = dir[0] + dir[1] + 1
        leftright_dir = dir[2] + dir[3] +1
        size_x, size_y = useful_board.shape[0], useful_board.shape[1]
        final_move_aug = []
        for i in range(updown_dir):
            for j in range(leftright_dir):
                temp_board = np.zeros([self.n, self.n])
                temp_pi = np.zeros([self.n, self.n])
                temp_board[i:i+size_x, j:j+size_y] = useful_board
                temp_pi[i:i+size_x, j:j+size_y] = useful_pi
                x, y = last_x + i - dir[0], last_y + j - dir[2]
                # print(last_action, x, y, dir)
                action = x*self.n + y
                temp_pi = temp_pi.reshape(self.n*self.n)
                final_move_aug.append((temp_board, temp_pi, action, player))
        
        return final_move_aug

    def play_with_human(self, human_first=True, checkpoint_name="best_checkpoint"):
        # load best model
        libtorch_best = NeuralNetwork('./models/best_checkpoint.pt', self.libtorch_use_gpu, 12)
        mcts_best = MCTS(libtorch_best, self.num_mcts_threads * 3, \
             self.c_puct, self.num_mcts_sims * 6, self.c_virtual_loss, self.action_size)

        # create gomoku game
        human_color = self.gomoku_gui.get_human_color()
        gomoku = Gomoku(self.n, self.n_in_row, human_color if human_first else -human_color)

        players = ["alpha", None, "human"] if human_color == 1 else ["human", None, "alpha"]
        player_index = human_color if human_first else -human_color

        self.gomoku_gui.reset_status()

        while True:
            player = players[player_index + 1]

            # select move
            if player == "alpha":
                prob = mcts_best.get_action_probs(gomoku)
                best_move = int(np.argmax(np.array(list(prob))))
                self.gomoku_gui.execute_move(player_index, best_move)
            else:
                self.gomoku_gui.set_is_human(True)
                # wait human action
                while self.gomoku_gui.get_is_human():
                    time.sleep(0.1)
                best_move = self.gomoku_gui.get_human_move()

            # execute move
            gomoku.execute_move(best_move)

            # check game status
            ended, winner = gomoku.get_game_status()
            if ended == 1:
                break

            # update tree search
            mcts_best.update_with_move(best_move)

            # next player
            player_index = -player_index

        print("HUMAN WIN" if winner == human_color else "ALPHA ZERO WIN")

    def load_samples(self, folder="models", filename="checkpoint.example"):
        """load self.examples_buffer
        """

        filepath = path.join(folder, filename)
        with open(filepath, 'rb') as f:
            self.examples_buffer = pickle.load(f)

    def save_samples(self, buffer, folder="models", filename="checkpoint.example"):
        """save self.examples_buffer
        """

        if not path.exists(folder):
            mkdir(folder)
        else:
            os.system("rm " + folder + "/*.example")

        filepath = path.join(folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(buffer, f, -1)
