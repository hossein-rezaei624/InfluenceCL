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


    
soft_1 = nn.Softmax(dim=1)

def distribute_samples(probabilities, M):
    # Normalize the probabilities
    total_probability = sum(probabilities.values())
    normalized_probabilities = {k: v / total_probability for k, v in probabilities.items()}

    # Calculate the number of samples for each class
    samples = {k: round(v * M) for k, v in normalized_probabilities.items()}
    
    # Check if there's any discrepancy due to rounding and correct it
    discrepancy = M - sum(samples.values())
    
    # Adjust the number of samples in each class to ensure the total number of samples equals M
    for key in samples:
        if discrepancy == 0:
            break    # Stop adjusting if there's no discrepancy
        if discrepancy > 0:
            # If there are less samples than M, add a sample to the current class and decrease discrepancy
            samples[key] += 1
            discrepancy -= 1
        elif discrepancy < 0 and samples[key] > 0:
            # If there are more samples than M and the current class has samples, remove one and increase discrepancy
            samples[key] -= 1
            discrepancy += 1

    return samples    # Return the final classes distribution

    
def distribute_excess(lst):
    # Calculate the total excess value
    total_excess = sum(val - 500 for val in lst if val > 500)

    # Number of elements that are not greater than 500
    recipients = [i for i, val in enumerate(lst) if val < 500]

    num_recipients = len(recipients)

    # Calculate the average share and remainder
    avg_share, remainder = divmod(total_excess, num_recipients)

    lst = [val if val <= 500 else 500 for val in lst]
    
    # Distribute the average share
    for idx in recipients:
        lst[idx] += avg_share
    
    # Distribute the remainder
    for idx in recipients[:remainder]:
        lst[idx] += 1
    
    # Cap values greater than 500
    for i, val in enumerate(lst):
        if val > 500:
            return distribute_excess(lst)
            break

    return lst



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
        novel_loss = 0*self.loss(logits, batch_y_combine)
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

