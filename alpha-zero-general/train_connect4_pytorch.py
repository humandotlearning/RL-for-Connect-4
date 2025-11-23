import logging
import time

import coloredlogs

from Coach import Coach
from connect4.Connect4Game import Connect4Game as Game
from connect4.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 300,
    # IMPROVEMENT: Increased from 10 to 50 games per iteration
    # Reason: More self-play games per iteration provides:
    # - Better training signal with more diverse positions
    # - More stable learning (less variance in updates)
    # - Better exploration of the state space
    # Original 10 was too low for reliable convergence
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.

    # IMPROVEMENT: Increased from 15 to 20 moves
    # Reason: Longer exploration phase (temp=1) helps discover more diverse strategies
    # Connect 4 benefits from exploration in opening and mid-game
    'tempThreshold': 20,        #

    # IMPROVEMENT: Increased from 0.6 to 0.55 (more lenient acceptance)
    # Reason: Original threshold was too strict, may reject good models due to variance
    # With 100 arena games, 55% win rate is statistically significant
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.

    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.

    # IMPROVEMENT: Increased from 25 to 100 simulations during self-play
    # Reason: More MCTS simulations = better policy quality during self-play
    # 25 was too low - model was learning from weak play
    # 100 is good balance between quality and speed
    # For online play, can use 200-400 for even stronger play
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.

    # IMPROVEMENT: Increased from 40 to 100 games
    # Reason: More arena games = more reliable evaluation
    # 40 games had too much variance - good models could be rejected by luck
    # 100 games gives ~95% confidence in win rate estimation
    'arenaCompare': 100,         # Number of games to play during arena play to determine if new net will be accepted.

    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(6)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    log.info('Loading best model weights from %s/%s...', args.checkpoint, 'best.pth.tar')
    nnet.load_checkpoint(args.checkpoint, 'best.pth.tar')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    import os
    examples_files = [f for f in os.listdir(args.checkpoint) if f.endswith('.examples')]
    if examples_files:
        examples_files.sort(key=lambda f: os.path.getmtime(os.path.join(args.checkpoint, f)), reverse=True)
        latest_examples = examples_files[0]
        base_name = latest_examples[:-9]  # strip '.examples'
        args.load_folder_file = (args.checkpoint, base_name)
        log.info("Selected trainExamples file: %s", os.path.join(args.load_folder_file[0], args.load_folder_file[1] + '.examples'))
    else:
        log.warning('No .examples files found in %s; continuing without train examples history.', args.checkpoint)
    log.info("Loading 'trainExamples' from file if available...")
    c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    start_time = time.time()
    c.learn()
    end_time = time.time()
    log.info('Learning process completed in %.2f seconds', end_time - start_time)


if __name__ == "__main__":
    main()
