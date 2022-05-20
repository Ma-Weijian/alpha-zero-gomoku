from collections import deque
from nis import match
from os import path, mkdir
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
from library import NeuralNetwork

from neural_network import NeuralNetWorkWrapper
from gomoku_rating_utils import GomokuTracker
from game_interface import GameInterface

prev_match_results = []

class League_Trainer():
    def __init__(self, config):
        # see config.py
        self.game_engine = GameInterface(config)

        # train
        self.num_iters = config['num_iters']
        self.num_eps = config['num_eps']
        self.num_train_threads = config['num_train_threads']
        self.check_freq = config['check_freq']
        self.num_contest = config['num_contest']
        self.update_threshold = config['update_threshold']
        self.action_size = config['action_size']


        self.examples_buffer_main_agent = deque([], maxlen=config['examples_buffer_max_len'])
        # These two buffers should be cleaned when a checkpoint is saved
        self.examples_buffer_main_exploiter = deque([], maxlen=config['examples_buffer_max_len'])
        self.examples_buffer_league_exploiter = deque([], maxlen=config['examples_buffer_max_len'])


        # mcts
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
                futures = [executor.submit(self.game_engine.play, itr, 1 if itr % 2 else -1, libtorch, libtorch, k, k == 1) for k in range(1, self.num_eps + 1)]
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


    def league_train(self):
        self.gomoku_tracker.init_result_trackers(prev_match_results)
        
        # init every model in main agent, main exploiter and league exploiter 
        # as the model pretrained in num_warmup iterations
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
                        executor.submit(self.game_engine.play, itr, 1, contest_list[i%len(contest_list)][0], contest_list[i%len(contest_list)][1], i%len(contest_list), i == 1)
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
