# coding: utf-8
import sys
sys.path.append('..')
sys.path.append('./src')

import learner
import config
import learner_league_train
import os

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "play", 'league_train']:
        print("[USAGE] python run_agent.py train|play|league_train")
        exit(1)

    if sys.argv[1] == "train" or sys.argv[1] == "play":
        runner = learner.Leaner(config.config)
    else:
        runner = learner_league_train.League_Trainer(config.config)


    if sys.argv[1] == "train":
        runner.learn()
    elif sys.argv[1] == "league_train":
        runner.league_train()
    elif sys.argv[1] == "play":
        for i in range(10):
            print("GAME: {}".format(i + 1))
            runner.play_with_human(human_first=i % 2)
            os.sleep(5)
