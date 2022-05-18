config = {
    # gomoku
    'n': 15,                                    # board size
    'n_in_row': 5,                              # n in row

    # mcts
    'libtorch_use_gpu' : True,                  # libtorch use cuda
    'num_mcts_threads': 1,                      # mcts threads number
    'num_mcts_sims': 1600,                      # mcts simulation times
    'c_puct': 5,                                # puct coeff
    'c_virtual_loss': 3,                        # virtual loss coeff

    # neural_network
    'train_use_gpu' : True,                     # train neural network using cuda
    'lr': 0.001,                                # learning rate
    'l2': 0.0001,                               # L2
    'num_channels': 256,                        # convolution neural network channel size
    'num_layers' : 8,                           # residual layer number
    'epochs': 0.6,                              # train epochs
    'batch_size': 512,                          # batch size for training, not for simulation

    # train
    'num_iters': 10000,
    'num_eps': 12,                              # self play times in per iter, must be multiples of 6
    'num_train_threads': 1,                     # self play in parallel
    'num_explore': 5,                           # explore step in a game
    'temp': 1,                                  # temperature
    'dirichlet_alpha': 0.3,                     # action noise in self play games
    'examples_buffer_max_len': 20,              # max length of examples buffer
    'noise_min': 0,
    'noise_max': 0.4,

    # for league training
    'num_warmup': 40,                           # warmup_round of before league training
    'check_freq': 40,                           # test model frequency

    # alphazero evaluations
    'update_threshold': 0.55,                   # update model threshold
    'num_contest': 10,                          # new/old model compare times, for alphazero-like rating evaluation

    # simsiam hyperparams
    'use_simsiam': True,
    'simsiam_loss_factor': 1,
    'simsiam_move_rate': 0.5,
    'simsiam_turn_rate': 0.5,
    'simsiam_flip_rate': 0.25,

    'apply_turn': False,

    # for whr rating
    'w2': 30,

    # test
    'human_color': 1,                           # human player's color

    # league_training
    'main_agent_selfplay_rate' : 0.35,
    'main_agent_pfsp_rate' : 0.85,              # 0.35 + 0.5

    # The original alphastar paper says that the main agents are discarded 
    # if it is beaten by the main agent at the rate of 1.
    # However, it is impossibile under account of WHR
    # so we use an approximate rate instead.
    'main_agent_discard_rate' : 0.98
}

# action size
config['action_size'] = config['n'] ** 2
