from tabnanny import verbose
import whr

class GomokuTracker():
    def __init__(self, w2, update_model_threshold):
        self.match_result = whr.Base(config={'w2': w2})
        self.current_model = 1
        self.update_model_threshold = update_model_threshold

    def init_result_trackers(self, prev_match_results):
        start = 0
        for match in prev_match_results:
            self.current_model = match[0]
            start = match[5]
            for i in range(match[2]):
                self.match_result.create_game('player_'+'{}'.format(match[0]), 'player_'+'{}'.format(match[1]), 'B', match[5])
            for i in range(match[3]):
                self.match_result.create_game('player_'+'{}'.format(match[0]), 'player_'+'{}'.format(match[1]), 'W', match[5])
            for i in range(match[4]):
                self.match_result.create_game('player_'+'{}'.format(match[0]), 'player_'+'{}'.format(match[1]), 'D', match[5])
        if prev_match_results != []:
            fianl_match = prev_match_results[-1]
            if float(fianl_match[2]) / (fianl_match[2] + fianl_match[3]) > self.update_model_threshold:
                self.current_model += 1
        
        return start

    def init_league_results(self, num_warmup):
        # whr library does not support the 0th day
        day = max(num_warmup, 1)
        self.match_result.create_game("main_ckpt_"+"{}".format(num_warmup), "main_exploit_ckpt_"+"{}".format(num_warmup), "D", day)
        self.match_result.create_game("league_exploit_ckpt_"+"{}".format(num_warmup), "main_exploit_ckpt_"+"{}".format(num_warmup), "D", day)
        self.match_result.create_game("main_ckpt_"+"{}".format(num_warmup), "league_exploit_ckpt_"+"{}".format(num_warmup), "D", day)
        self.match_result.iterate_until_converge(verbose=False)

    
    def update_result_tracker(self, iteration, one_won, two_won, draws):
        for i in range(one_won):
            self.match_result.create_game('player_'+'{}'.format(self.current_model), 'player_'+'{}'.format(self.current_model-1), 'B', iteration)
        for i in range(two_won):
            self.match_result.create_game('player_'+'{}'.format(self.current_model), 'player_'+'{}'.format(self.current_model-1), 'W', iteration)  
        for i in range(draws):
            self.match_result.create_game('player_'+'{}'.format(self.current_model), 'player_'+'{}'.format(self.current_model-1), 'D', iteration)              
        self.match_result.iterate_until_converge(verbose=False)

        if one_won + two_won > 0 and float(one_won) / (one_won + two_won) > self.update_model_threshold:
            self.current_model += 1
        
        return self.match_result.get_ordered_ratings()

    def find_opponent_hard(self, itr, id, rival_list):
        # print(itr, id, rival_list)
        prob = [self.calc_beat_rate(itr, id, i) for i in rival_list]
        # print(rival_list, prob)
        hard = [(1-i)**2 for i in prob]
        return rival_list[hard.index(max(hard))]
    
    def find_opponent_rel(self, itr, id, rival_list):
        # print(itr, id, rival_list)
        prob = [self.calc_beat_rate(itr, id, i) for i in rival_list]
        # print(rival_list, prob)
        rel = [(1-i)*i for i in prob]
        return rival_list[rel.index(max(rel))]

    def calc_beat_rate(self, itr, player1, player2):
        # print(itr, player1, player2)
        virtual_game = whr.Game(player1, player2, "B", itr)
        eval = whr.Evaluate(self.match_result)
        return eval.evaluate_single_game(virtual_game)

    def create_game(self, player1, player2, result, timestamp):
        self.match_result.create_game(player1, player2, result, timestamp)
    
    def iterate_until_converge(self, verbose=False):
        self.match_result.iterate_until_converge(verbose=False)

    def get_ordered_ratings(self):
        self.match_result.iterate_until_converge(verbose=False)
        return self.match_result.get_ordered_ratings()