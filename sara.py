import torch
import torch.nn.functional as F
from torchattacks import *
import copy

def sara(model, 
         forward,
         inputs, 
         labels, 
         modified_miss_threshold=0.05, 
         loss_options = ['modified_miss','normalize_logits','modified_both_cleanamiss']):
    
    model.eval()
    if 'modified_miss' in loss_options:
        original_input, original_labels = copy.deepcopy(inputs), copy.deepcopy(labels)
        
        # with torch.no_grad():
        natural_logits, _ = forward(original_input)
        # _, pred = torch.max(natural_logits.data, 1)
        # correct += torch.sum(pred == original_labels).item()
            
        if "normalize_logits" in loss_options:
            natural_logits_normalized = natural_logits / torch.norm(natural_logits, p=float('inf'))
            natural_logits = natural_logits_normalized
            
        natural_true_labels = original_labels
        # natural_predicted_labels = torch.argmax(natural_logits, dim=1)
        _, natural_predicted_labels = torch.max(natural_logits.data, 1)
        
        # Find misclassified samples for current model
        misclassified_samples = []
        misclassified_labels = []
        misclassified_predicted_labels = []
                
        condition1 =  torch.max(F.softmax(natural_logits), dim=1).values> modified_miss_threshold
        condition2 =  (natural_predicted_labels != natural_true_labels)
    
        condition1=1
        misclassified_indices = (condition1 * condition2).nonzero()
            
        # Append misclassified samples and labels to the lists
        for index in misclassified_indices:
            misclassified_samples.append(original_input[index])
            misclassified_labels.append(original_labels[index])
            misclassified_predicted_labels.append(natural_predicted_labels[index])

        if "modified_both_cleanamiss" in loss_options:
            ### targeted attack
            # targeted params could be differ from other attack based apporaches
            targeted_eps = 0.03
            targeted_apha = 8/255
            targeted_steps = 20
            attack = PGD(model, eps=targeted_eps, alpha=targeted_apha, steps=targeted_steps, random_start=True)
            
            #### apply attack for misclassified samples
            if len(misclassified_samples)>0:    
                if type(misclassified_samples)==list:
                    misclassified_samples = torch.cat(misclassified_samples, dim=0)
                    misclassified_labels = torch.cat(misclassified_labels, dim=0)
                    misclassified_predicted_labels = torch.cat(misclassified_predicted_labels, dim=0)
                
                # x_hat = gradient_ascent_attack(model, misclassified_samples, misclassified_labels, misclassified_labels, epsilon=0.03, num_steps=10, alpha=0.01)
                x_hat = attack(misclassified_samples, misclassified_labels)

                        
                logits,_ = forward(x_hat)
                logits = F.log_softmax(logits, dim=1)
                # predicted_labels = torch.argmax(logits, dim=1)
                        
                _, predicted_labels = torch.max(logits.data, 1)
                # predicted_labels = torch.argmax(logits, dim=1)
                # miss2nd = (predicted_labels != misclassified_labels)
                # how many samples are now correctly classified ????????????  TODO
                
                for i, index in enumerate(misclassified_indices):        
                    check_just_true = predicted_labels[i]== misclassified_labels[i]
                    if check_just_true:
                        original_input = torch.cat((original_input, x_hat[i].unsqueeze(0)))
                        original_labels = torch.cat((original_labels, misclassified_labels[i].unsqueeze(0)))
                        print('New sample is added')
            else:
                print(f'no missclassified sample!')
    model.train()
    inputs = original_input
    labels = original_labels
    return inputs, labels
