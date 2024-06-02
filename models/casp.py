import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.casp_loss import SupConLoss
from utils.casp_transforms_aug import transforms_aug

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

    
def distribute_excess(lst, check_bound):
    # Calculate the total excess value
    total_excess = sum(val - check_bound for val in lst if val > check_bound)

    # Number of elements that are not greater than check_bound
    recipients = [i for i, val in enumerate(lst) if val < check_bound]

    num_recipients = len(recipients)

    # Calculate the average share and remainder
    avg_share, remainder = divmod(total_excess, num_recipients)

    lst = [val if val <= check_bound else check_bound for val in lst]
    
    # Distribute the average share
    for idx in recipients:
        lst[idx] += avg_share
    
    # Distribute the remainder
    for idx in recipients[:remainder]:
        lst[idx] += 1
    
    # Cap values greater than check_bound
    for i, val in enumerate(lst):
        if val > check_bound:
            return distribute_excess(lst, check_bound)
            break

    return lst


def adjust_values_integer_include_all(a, b):
    excess = {}
    shortage = {}
    total_excess = 0

    # Establish initial excess and shortage based on the limits in b
    for k in a:
        if k in b:
            if a[k] > b[k]:
                excess[k] = a[k] - b[k]
                total_excess += a[k] - b[k]
                a[k] = b[k]  # Adjust to the limit of b
            elif a[k] < b[k]:
                shortage[k] = b[k] - a[k]  # Available space to increase
        else:
            # If no corresponding key in b, treat as having no upper limit
            shortage[k] = float('inf')  # Theoretically unlimited capacity

    # Distribute the excess to those under the limit as integers
    while total_excess > 0 and shortage:
        per_key_excess = max(total_excess // len(shortage), 1)  # Ensure minimal distribution
        for k in list(shortage):
            if total_excess == 0:
                break
            if shortage[k] == float('inf'):
                increment = per_key_excess  # No limit, so use per_key_excess
            else:
                increment = min(shortage[k], per_key_excess)

            a[k] += increment
            total_excess -= increment

            if shortage[k] != float('inf'):
                shortage[k] -= increment
                if shortage[k] == 0:
                    del shortage[k]  # Remove key from shortage if fully adjusted

    # Ensure all values are integers
    for key in a:
        a[key] = int(a[key])

    return a


class Casp(ContinualModel):
    NAME = 'casp'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(Casp, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.epoch = 0
        self.unique_classes = set()
        self.mapping = {}
        self.reverse_mapping = {}
        self.confidence_by_class = {}
        self.confidence_by_sample = None
        self.n_sample_per_task = None
        self.class_portion = []
        self.task_portion = []
        self.dist_task_prev = None
        self.task_class = {}
        self.dist_class_prev = None
        self.predicted_epoch = 1
        self.task_conf_first = []

    def begin_train(self, dataset):
        self.n_sample_per_task = dataset.get_examples_number()//dataset.N_TASKS
    
    def begin_task(self, dataset, train_loader):
        self.epoch = 0
        self.task += 1
        self.unique_classes = set()
        for _, labels, _, _ in train_loader:
            self.unique_classes.update(labels.numpy())
            if len(self.unique_classes)==dataset.N_CLASSES_PER_TASK:
                break
        self.mapping = {value: index for index, value in enumerate(self.unique_classes)}
        self.reverse_mapping = {index: value for value, index in self.mapping.items()}
        self.confidence_by_class = {class_id: {epoch: [] for epoch in range(self.args.n_epochs)} for class_id, __ in enumerate(self.unique_classes)}
        self.confidence_by_sample = torch.zeros((self.args.n_epochs, self.n_sample_per_task))
        self.confidence_by_task = {task_id: [] for task_id in range(self.task)}
        self.task_class.update({value: (self.task - 1) for index, value in enumerate(self.unique_classes)})
    
    def end_epoch(self, dataset, train_loader):

        if self.epoch == 0 and self.task == 1:
            self.predicted_epoch = torch.mean(torch.tensor(self.task_conf_first)).item()
            print("self.predicted_epoch", self.predicted_epoch)
            self.predicted_epoch = round(self.predicted_epoch * np.log(dataset.N_CLASSES_PER_TASK) / np.log(dataset.N_TASKS))
            print("self.predicted_epoch", self.predicted_epoch)
            if self.predicted_epoch > self.args.n_epochs:
                self.predicted_epoch = self.args.n_epochs
            if self.predicted_epoch < 2:
                self.predicted_epoch = 2
            self.predicted_epoch = 3
            print("self.predicted_epoch", self.predicted_epoch)
        
        if self.epoch == (self.args.n_epochs - 1) and not self.buffer.is_empty():
            buffer_logits, _ = self.net.pcrForward(self.buffer.examples)
            soft_buffer = soft_1(buffer_logits)
            for j in range(len(self.buffer)):
                self.confidence_by_task[self.task_class[self.buffer.labels[j].item()]].append(soft_buffer[j, self.buffer.labels[j]].item())
                    
        self.epoch += 1
        
        if self.epoch == self.args.n_epochs:
            # Calculate mean confidence by class
            mean_by_class = {class_id: {epoch: torch.mean(torch.tensor(confidences[epoch])) for epoch in range(self.predicted_epoch)} for class_id, confidences in self.confidence_by_class.items()}
            
            # Calculate standard deviation of mean confidences by class
            std_of_means_by_class = {class_id: torch.mean(torch.tensor([mean_by_class[class_id][epoch] for epoch in range(self.predicted_epoch)])) for class_id, __ in enumerate(self.unique_classes)}


            mean_by_task = {task_id: torch.std(torch.tensor(confidences)) for task_id, confidences in self.confidence_by_task.items()}
   ####         std_of_means_by_task = {task_id: torch.mean(torch.tensor([mean_by_task[task_id][epoch] for epoch in range(self.args.casp_epoch)])) for task_id in range(self.task)}
            std_of_means_by_task = {task_id: mean_by_task[task_id] for task_id in range(self.task)}


            ##self.confidence_by_sample = self.confidence_by_sample[:self.predicted_epoch]
            self.confidence_by_sample = self.confidence_by_sample[:5]
            
            # Compute mean and variability of confidences for each sample
            Confidence_mean = self.confidence_by_sample.mean(dim=0)
            Variability = self.confidence_by_sample.std(dim=0)

            ##plt.scatter(Variability, Confidence_mean, s = 2)
            
            ##plt.xlabel("Variability") 
            ##plt.ylabel("Confidence") 
            
            ##plt.savefig('scatter_plot.png')

            
        
            # Sort indices based on the Confidence
            sorted_indices_1 = np.argsort(Confidence_mean.numpy())
            
            # Sort indices based on the variability
            ##sorted_indices_2 = np.argsort(Variability.numpy())
            
        
        
            ##top_indices_sorted = sorted_indices_1 #hard
            
            top_indices_sorted = sorted_indices_1[::-1].copy() #simple
        
            # Descending order
            ##top_indices_sorted = sorted_indices_2[::-1].copy() #challenging


            # Initialize lists to hold data
            all_inputs, all_labels, all_not_aug_inputs, all_indices = [], [], [], []
            
            # Collect all data
            for data_1 in train_loader:
                inputs_1, labels_1, not_aug_inputs_1, indices_1 = data_1
                all_inputs.append(inputs_1)
                all_labels.append(labels_1)
                all_not_aug_inputs.append(not_aug_inputs_1)
                all_indices.append(indices_1)
            
            # Concatenate all collected items to form complete arrays            
            all_inputs = torch.cat(all_inputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_not_aug_inputs = torch.cat(all_not_aug_inputs, dim=0)
            all_indices = torch.cat(all_indices, dim=0)

            # Convert sorted_indices_2 to a tensor for indexing
            top_indices_sorted = torch.tensor(top_indices_sorted, dtype=torch.long)

            # Find the positions of these indices in the shuffled order
            positions = torch.hstack([torch.where(all_indices == index)[0] for index in top_indices_sorted])

            # Extract inputs and labels using these positions
            ###all_images = all_inputs[positions]
            ###all_not_aug_inputs = all_not_aug_inputs[positions]
            all_images = all_not_aug_inputs[positions]
            all_labels = all_labels[positions]


            # Extract the first 12 images to display (or fewer if there are less than 12 images)
            ##images_display = [all_images[j] for j in range(100)]
    
            # Make a grid from these images
            ##grid = torchvision.utils.make_grid(images_display, nrow=10)  # Adjust nrow based on actual images
            
            # Save grid image with unique name for each batch
            ##torchvision.utils.save_image(grid, 'grid_image.png')



            # Extract the first 12 images to display (or fewer if there are less than 12 images)
            ##images_display_ = [all_not_aug_inputs[j] for j in range(100)]
    
            # Make a grid from these images
            ##grid_ = torchvision.utils.make_grid(images_display_, nrow=10)  # Adjust nrow based on actual images
            
            # Save grid image with unique name for each batch
            ##torchvision.utils.save_image(grid_, 'grid_image_not_aug_inputs.png')

            
            # Convert standard deviation of means by class to item form
            updated_std_of_means_by_class = {k: v.item() for k, v in std_of_means_by_class.items()}
            updated_std_of_means_by_class = {self.reverse_mapping[k]: v for k, v in updated_std_of_means_by_class.items()} # comment for balance
            ##updated_std_of_means_by_class = {self.reverse_mapping[k]: 1 for k, _ in updated_std_of_means_by_class.items()}   #uncomment for balance

            self.class_portion.append(updated_std_of_means_by_class)
##            self.task_portion.append(((self.confidence_by_sample.std(dim=1)).mean(dim=0)).item())
            
##            updated_task_portion = {i:value for i, value in enumerate(self.task_portion)}
##            dist_task = distribute_samples(updated_task_portion, self.args.buffer_size)

##            if self.task > 1:
##                updated_task_portion_prev = {i:value for i, value in enumerate(self.task_portion[:-1])}
##                dist_task_prev = distribute_samples(updated_task_portion_prev, self.args.buffer_size)
            
####            same_task_number = self.args.buffer_size//self.task
####            dist_task = {i:same_task_number for i in range(self.task)}
####            diff = self.args.buffer_size - same_task_number*self.task
####            for o in range(diff):
####                dist_task[o] += 1
####
####            if self.task > 1:
####                same_task_number_prev = self.args.buffer_size//(self.task - 1)
####                dist_task_prev = {i:same_task_number_prev for i in range(self.task - 1)}
####                diff_prev = self.args.buffer_size - same_task_number_prev*(self.task - 1)
####                for o in range(diff_prev):
####                    dist_task_prev[o] += 1


            ##updated_std_of_means_by_task = {k: v.item() for k, v in std_of_means_by_task.items()}  # comment for balance
            updated_std_of_means_by_task = {k: 1 for k, v in std_of_means_by_task.items()}    #uncomment for balance
            dist_task_before = distribute_samples(updated_std_of_means_by_task, self.args.buffer_size)
            
            if self.task > 1:
                dist_task = adjust_values_integer_include_all(dist_task_before.copy(), self.dist_task_prev)
            else:
                dist_task = dist_task_before
            
            dist_class = [distribute_samples(self.class_portion[i], dist_task[i]) for i in range(self.task)]
            
###            if self.task > 1:
###                dist_class_prev = [distribute_samples(self.class_portion[i], self.dist_task_prev[i]) for i in range(self.task - 1)]

            self.dist_task_prev = dist_task

            ##print("dist_class", dist_class)
            ##print("dist_task", dist_task)
            
            # Distribute samples based on the standard deviation
            dist = dist_class.pop()
            dist_last = dist.copy()
            dist = {self.mapping[k]: v for k, v in dist.items()}

            
            # Initialize a counter for each class
            counter_class = [0 for _ in range(len(self.unique_classes))]
        
            # Distribution based on the class variability
            condition = [dist[k] for k in range(len(dist))]
        
            # Check if any class exceeds its allowed number of samples
            check_bound = self.n_sample_per_task//len(self.unique_classes)
            for i in range(len(condition)):
                if condition[i] > check_bound:
                    # Redistribute the excess samples
                    condition = distribute_excess(condition, check_bound)
                    break
        
            # Initialize new lists for adjusted images and labels
            images_list_ = []
            labels_list_ = []
        
            # Iterate over all_labels and select most challening images for each class based on the class variability
            for i in range(all_labels.shape[0]):
                if counter_class[self.mapping[all_labels[i].item()]] < condition[self.mapping[all_labels[i].item()]]:
                    counter_class[self.mapping[all_labels[i].item()]] += 1
                    labels_list_.append(all_labels[i])
                    images_list_.append(all_images[i])
                if counter_class == condition:
                    break
        
            # Stack the selected images and labels
            all_images_ = torch.stack(images_list_).to(self.device)
            all_labels_ = torch.stack(labels_list_).to(self.device)
        
            
            counter_manage = [{k:0 for k, __ in dist_class[i].items()} for i in range(self.task - 1)]

            dist_class_merged = {}
            counter_manage_merged = {}
            dist_class_merged_prev = {}
            
            for d in dist_class:
                dist_class_merged.update(d)
            for f in counter_manage:
                counter_manage_merged.update(f)
            if self.task > 1:
                dist_class_merged_prev = self.dist_class_prev
                class_key = list(dist_class_merged.keys())
                temp_key = -1
                for k, value in dist_class_merged.items():
                    temp_key += 1
                    if value > dist_class_merged_prev[k]:
                        temp = value - dist_class_merged_prev[k]
                        dist_class_merged[k] -= temp
                        for hh in range(temp):
                            dist_class_merged[class_key[temp_key + hh + 1]] += 1
            
            self.dist_class_prev = dist_class_merged.copy()
            self.dist_class_prev.update(dist_last)
            if not self.buffer.is_empty():
                # Initialize new lists for adjusted images and labels
                images_store = []
                labels_store = []
                
                # Iterate over all_labels and select most challening images for each class based on the class variability
                for i in range(len(self.buffer)):
                    if counter_manage_merged[self.buffer.labels[i].item()] < dist_class_merged[self.buffer.labels[i].item()]:
                        counter_manage_merged[self.buffer.labels[i].item()] += 1
                        labels_store.append(self.buffer.labels[i])
                        images_store.append(self.buffer.examples[i])
                    if counter_manage_merged == dist_class_merged:
                        break
                
                # Stack the selected images and labels
                images_store_ = torch.stack(images_store).to(self.device)
                labels_store_ = torch.stack(labels_store).to(self.device)
                
                all_images_ = torch.cat((images_store_, all_images_))
                all_labels_ = torch.cat((labels_store_, all_labels_))

            if not hasattr(self.buffer, 'examples'):
                self.buffer.init_tensors(all_images_, all_labels_, None, None)
            
            self.buffer.num_seen_examples += self.n_sample_per_task
            
            # Update the buffer with the shuffled images and labels
            self.buffer.labels = all_labels_
            self.buffer.examples = all_images_

    def observe(self, inputs, labels, not_aug_inputs, index_):

        real_batch_size = inputs.shape[0]
        
        if self.epoch < 12: #self.predicted_epoch
            targets = torch.tensor([self.mapping[val.item()] for val in labels]).to(self.device)
            confidence_batch = []

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

        if self.epoch < 12:  #self.predicted_epoch
            casp_logits, _ = self.net.pcrForward(not_aug_inputs)
            soft_ = soft_1(casp_logits)
            # Accumulate confidences
            for i in range(targets.shape[0]):
                confidence_batch.append(soft_[i,labels[i]].item())
                
                # Update the dictionary with the confidence score for the current class for the current epoch
                self.confidence_by_class[targets[i].item()][self.epoch].append(soft_[i, labels[i]].item())
            
            # Record the confidence scores for samples in the corresponding tensor
            conf_tensor = torch.tensor(confidence_batch)
            self.confidence_by_sample[self.epoch, index_] = conf_tensor
    

        if self.epoch == (self.args.n_epochs - 1):
            casp_logits, _ = self.net.pcrForward(not_aug_inputs)
            soft_task = soft_1(casp_logits)
            for j in range(labels.shape[0]):
                self.confidence_by_task[self.task_class[labels[j].item()]].append(soft_task[j, labels[j]].item())
            

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
            mem_x, mem_y = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
        
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

        if self.epoch == 0 and self.task == 1:
            self.task_conf_first.append(novel_loss.item())
        
        novel_loss.backward()
        self.opt.step()
        
        return novel_loss.item()
