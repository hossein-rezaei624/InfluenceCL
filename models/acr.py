import torch  ## back to here again
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.acr_loss import SupConLoss
from utils.acr_transforms_aug import transforms_aug

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Class-Adaptive Sampling Policy.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--E', type=int, default=5,
                        help='Epoch for strategies')
    
    return parser



class Acr(ContinualModel):
    NAME = 'acr'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(Acr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.transform = None
        self.task = 0
        self.epoch = 0


    def begin_train(self, dataset):
        self.n_sample_per_task = dataset.get_examples_number()//dataset.N_TASKS
    
    def begin_task(self, dataset, train_loader):
        self.epoch = 0
        self.task += 1
        
    
    def end_epoch(self, dataset, train_loader):
        self.epoch += 1    

    def observe(self, inputs, labels, not_aug_inputs, index_):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()

        
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))


        
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=inputs[:real_batch_size],
                             labels=labels[:real_batch_size])
        
        
        return loss.item()
