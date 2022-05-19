import time
import numpy as np
import concurrent.futures
from data_augmentation import AugmentUtils
from gomoku_gui import GomokuGUI
import threading
import math

import sys
sys.path.append('../build')
from library import MCTS, Gomoku, NeuralNetwork

class GameInterface():
    def __init__(self, config) -> None:
        self.augment_utils = AugmentUtils(config)
        # gomoku
        self.n = config['n']
        self.n_in_row = config['n_in_row']
        self.action_size = config['action_size']
        self.gomoku_gui = GomokuGUI(config['n'], config['human_color'])
        # start gui
        t = threading.Thread(target=self.gomoku_gui.loop)
        t.start()

        # mcts
        self.num_mcts_sims = config['num_mcts_sims']
        self.c_puct = config['c_puct']
        self.c_virtual_loss = config['c_virtual_loss']
        self.num_mcts_threads = config['num_mcts_threads']
        self.libtorch_use_gpu = config['libtorch_use_gpu']

        # train
        self.temp = config['temp']
        self.num_explore = config['num_explore']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.noise_min = config['noise_min']
        self.noise_max = config['noise_max']
        self.check_freq = config['check_freq']
        self.num_train_threads = config['num_train_threads']


    def tuple_2d_to_numpy_2d(self, tuple_2d):
        # help function
        # convert type
        res = [None] * len(tuple_2d)
        for i, tuple_1d in enumerate(tuple_2d):
            res[i] = list(tuple_1d)
        return np.array(res)

    def get_exploration_rate(self, iter):
        expl_rate = self.noise_min + 0.5 * (self.noise_max-self.noise_min) * (1+math.cos(iter/self.check_freq * math.pi))
        return expl_rate

    def play(self, iter, first_color, libtorch_player1, libtorch_player2, index, show):
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
            board = self.tuple_2d_to_numpy_2d(gomoku.get_board())
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
                # there are
                # b, simsiam_augmented_board, a, simsiam_augmented_action, res_player, p, reward = res_player*winner
                # in augmented_board
                augmented_examples = self.augment_utils.augment_received_examples(train_examples, winner)
                return augmented_examples, index, winner

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
