import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.casp_loss import SupConLoss
from utils.casp_transforms_aug import transforms_aug


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Class-Adaptive Sampling Policy.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Casp(ContinualModel):
    NAME = 'casp'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(Casp, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)


    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]
        
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
        ##novel_loss = 0*self.loss(logits, batch_y_combine)
        novel_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True) * 2
        print("novel_loss", novel_loss)
        self.opt.zero_grad()

        if not self.buffer.is_empty():
            mem_x, mem_y = self.buffer.get_data(
                self.args.minibatch_size, transform=None)
        
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
        
        # update mem
        self.buffer.add_data(examples=inputs[:real_batch_size],
                             labels=labels[:real_batch_size])

        return novel_loss.item()

