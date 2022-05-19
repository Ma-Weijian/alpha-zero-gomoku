import numpy as np
import random

class AugmentUtils():
    def __init__(self, config) -> None:
        self.n = config['n']
        self.apply_turn = config['apply_turn']
        self.action_size = config['action_size']


        # train with simsiam
        self.simsiam_move_rate = config['simsiam_move_rate']
        self.simsiam_turn_rate = config['simsiam_turn_rate']
        self.simsiam_flip_rate = config['simsiam_flip_rate']

    def augment_received_examples(self, train_examples, winner):
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

        # print(dir, len(train_examples))
        temp_examples = []
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
                    temp_examples.append([b, simsiam_augmented_board, a, simsiam_augmented_action, res_player, p])
                # print(len(augmented_examples))
        return [(x[0], x[1], x[2], x[3], x[4], x[5], x[4] * winner) for x in temp_examples]

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

