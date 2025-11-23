import os 
import sys 
import time
import numpy as np 
from tqdm import tqdm
from utils import *
import torch 
import torch.optim as optim 

from NeuralNet import NeuralNet 

from .Connect4NNet import Connect4Net as c4nnet 

args = dotdict({
    # IMPROVEMENT: Slightly reduced initial learning rate from 0.001 to 0.002
    # Reason: Will use scheduler to decay, so start slightly higher
    # Scheduler will reduce this over time for fine-tuning
    'lr': 0.002,

    # IMPROVEMENT: Reduced dropout from 0.3 to 0.2
    # Reason: 0.3 was too aggressive, causing underfitting
    # Connect 4 has limited state space, less regularization needed
    # 0.2 provides good balance between generalization and learning capacity
    'dropout': 0.2,

    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,

    # IMPROVEMENT: Reduced from 20 to 15 residual blocks
    # Reason: 20 blocks is overkill for Connect 4 (much simpler than Go)
    # - Faster training and inference
    # - Less overfitting risk
    # - Still enough capacity for strong play
    # AlphaGo Zero used 40 blocks for 19x19 Go; Connect 4 is ~100x simpler
    'num_residual_layers': 15
})

class NNetWrapper( NeuralNet):
    def __init__(self, game):
        super().__init__(game)
        self.nnet = c4nnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        optimizer = optim.Adam( self.nnet.parameters(), lr=args.lr)

        # IMPROVEMENT: Add learning rate scheduler for better convergence
        # Reason: Cosine annealing helps fine-tune the network in later epochs
        # - Starts with higher LR for faster initial learning
        # - Gradually reduces to avoid overshooting optimal weights
        # - Cosine schedule is smooth and well-tested for deep RL
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()

            self.pis = AverageMeter()
            self.vs = AverageMeter()
            self.total_loss = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')

            for _ in t:

                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)

                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                self.pis.update(l_pi.item(), boards.size(0))
                self.vs.update(l_v.item(), boards.size(0))
                self.total_loss.update(total_loss.item(), boards.size(0))
                t.set_postfix(Loss_pi=self.pis, Loss_v=self.vs, Loss_total=self.total_loss)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), 5.0)
                optimizer.step()

            # Step the scheduler after each epoch
            scheduler.step()
            print(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')


    def predict(self, board):
        start = time.time()
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)

        self.nnet.eval()

        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).detach().cpu().numpy()[0], v.detach().cpu().numpy()[0]

    
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

        

        

        


            
