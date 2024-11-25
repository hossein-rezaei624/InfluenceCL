import torch
from utils.ifs_buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
from utils.ntk_generator import get_kernel_fn
from utils.influence_ntk import InfluenceNTK
import jax
from utils.acr_loss import SupConLoss
from utils.acr_transforms_aug import transforms_aug


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--sel_epoch', type=int, nargs='+', default=[49],
                        help='Epoch for sample selection')
    parser.add_argument('--mu', type=float, default=0.5,
                        help='Probability of already-in-coreset case.')
    parser.add_argument('--nu', type=float, default=0.01,
                        help='Weight for second-order influence functions.')
    parser.add_argument('--lmbda', type=float, default=1e-3,
                        help='Regularization coefficient for NTK.')
    parser.add_argument('--norig', action='store_true',
                        help='Not store original image in the buffer')
    return parser






class Acr(ContinualModel):
    NAME = 'acr'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(Acr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.epoch = 0
        self.kernel_fn = get_kernel_fn(backbone)
        self.ifs = InfluenceNTK()

    def begin_train(self, dataset):
        self.n_sample_per_task = dataset.get_examples_number()//dataset.N_TASKS
    
    def begin_task(self, dataset, train_loader):
        self.epoch = 0
        self.task += 1
        self.ifs.out_dim = dataset.N_CLASSES_PER_TASK * self.task
    
    def end_epoch(self, dataset, train_loader):
        self.epoch += 1

    def observe(self, inputs, labels, not_aug_inputs, index_):
        
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        

        # batch update
        batch_x, batch_y = inputs, labels
        batch_x_aug = torch.stack([transforms_aug[self.args.dataset](batch_x[idx].cpu())
                                   for idx in range(batch_x.size(0))])
        batch_x = batch_x.to(self.device)
        batch_x_aug = batch_x_aug.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_x_combine = torch.cat((batch_x, batch_x_aug))
        batch_y_combine = torch.cat((batch_y, batch_y))
            
        logits, feas= self.net.pcrForward(batch_x_combine)
        novel_loss = 0*self.loss(logits, batch_y_combine)

    
        
        if self.buffer.is_empty():
            feas_aug = self.net.pcrLinear.L.weight[batch_y_combine]

            feas_norm = torch.norm(feas, p=2, dim=1).unsqueeze(1).expand_as(feas)
            feas_normalized = feas.div(feas_norm + 0.000001)

            feas_aug_norm = torch.norm(feas_aug, p=2, dim=1).unsqueeze(1).expand_as(
                feas_aug)
            feas_aug_normalized = feas_aug.div(feas_aug_norm + 0.000001)
            cos_features = torch.cat([feas_normalized.unsqueeze(1),
                                      feas_aug_normalized.unsqueeze(1)],
                                     dim=1)
            PSC = SupConLoss(temperature=0.09, contrast_mode='proxy')
            novel_loss += PSC(features=cos_features, labels=batch_y_combine)

        
        else:
            indexes, mem_x, mem_y = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True)    
        
            mem_x_aug = torch.stack([transforms_aug[self.args.dataset](mem_x[idx].cpu())
                                     for idx in range(mem_x.size(0))])
            mem_x = mem_x.to(self.device)
            mem_x_aug = mem_x_aug.to(self.device)
            mem_y = mem_y.to(self.device)
            mem_x_combine = torch.cat([mem_x, mem_x_aug])
            mem_y_combine = torch.cat([mem_y, mem_y])


            mem_logits, mem_fea= self.net.pcrForward(mem_x_combine)

            combined_feas = torch.cat([mem_fea, feas])
            combined_labels = torch.cat((mem_y_combine, batch_y_combine))
            combined_feas_aug = self.net.pcrLinear.L.weight[combined_labels]

            combined_feas_norm = torch.norm(combined_feas, p=2, dim=1).unsqueeze(1).expand_as(combined_feas)
            combined_feas_normalized = combined_feas.div(combined_feas_norm + 0.000001)

            combined_feas_aug_norm = torch.norm(combined_feas_aug, p=2, dim=1).unsqueeze(1).expand_as(
                combined_feas_aug)
            combined_feas_aug_normalized = combined_feas_aug.div(combined_feas_aug_norm + 0.000001)
            cos_features = torch.cat([combined_feas_normalized.unsqueeze(1),
                                      combined_feas_aug_normalized.unsqueeze(1)],
                                     dim=1)
            PSC = SupConLoss(temperature=0.09, contrast_mode='proxy')
            novel_loss += PSC(features=cos_features, labels=combined_labels)

        
        novel_loss.backward()
        self.opt.step()


        if self.epoch in self.args.sel_epoch:
            inputs = inputs if self.args.norig else not_aug_inputs
            if self.buffer.num_seen_examples < self.args.buffer_size:
                self.buffer.add_data(examples=inputs[:real_batch_size],
                                     labels=labels[:real_batch_size])
            else:
                inc_weight = real_batch_size / self.buffer.num_seen_examples
                buf_inputs, buf_labels = self.buffer.get_all_data()
                inputs = torch.cat((inputs[:real_batch_size], buf_inputs))
                labels = torch.cat((labels[:real_batch_size], buf_labels))
                chosen_indexes = self.ifs.select(inputs.cpu(), labels.cpu(), self.buffer.buffer_size, self.kernel_fn,
                                                 self.args.lmbda, self.args.mu, self.args.nu, inc_weight)[0]
                out_indexes = np.setdiff1d(np.arange(self.buffer.buffer_size), chosen_indexes - real_batch_size)
                in_indexes = chosen_indexes[chosen_indexes < real_batch_size]
                self.buffer.replace_data(out_indexes, inputs[in_indexes], labels[in_indexes])
                self.buffer.num_seen_examples += real_batch_size
        
        
        return novel_loss.item()
