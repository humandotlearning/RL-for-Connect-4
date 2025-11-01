# main_connect4.py
from Coach import Coach
from utils import dotdict

# Use the built-in Connect4 game + PyTorch net shipped in the repo
from connect4.Connect4Game import Connect4Game as Game
from connect4.pytorch.NNet import NNetWrapper as NNet

# --- baseline training settings (good first run) ---
args = dotdict({
    # Self-play / learning loop
    'numIters': 50,                     # training iterations
    'numEps': 100,                      # self-play games per iter
    'tempThreshold': 10,                # moves before temp->0
    'updateThreshold': 0.55,            # accept new model if win-rate >= 55%
    'maxlenOfQueue': 200000,            # replay buffer size
    'numMCTSSims': 200,                 # simulations per move in self-play
    'arenaCompare': 40,                 # games to compare new vs old nets
    'cpuct': 1.25,                      # exploration constant

    # Checkpoints / history
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    # NN training params
    'lr': 1e-3,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 256,
    'cuda': True,                       # flip to False if on CPU
})

# 6x7 standard board
g = Game(6, 7)

# Build network & (optionally) load a checkpoint
nnet = NNet(g)
if args.load_model:
    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

# Orchestrate training
coach = Coach(g, nnet, args)
coach.learn()
