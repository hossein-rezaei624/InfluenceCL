# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

##import itertools

from corruptions import *

# Add deprecated aliases back to NumPy
np.float = float

from torchvision.transforms import ToPILImage, PILToTensor

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.cm as cm
from matplotlib.lines import Line2D

import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar

try:
    import wandb
except ImportError:
    wandb = None

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    accs_augmented, accs_mask_classes_augmented = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        correct_augmented, correct_mask_classes_augmented, total_augmented = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    if model.NAME == 'casp':
                        outputs, _ = model.net.pcrForward(inputs)
                    else:
                        outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

                
                batch_x = inputs
                batch_y = labels

                ###batch_x = batch_x.expand(-1, 3, -1, -1) #should be uncomment for mnist
                
                # List to hold all the batches with distortions applied
                all_batches = []
                
                # Convert the batch of images to a list of PIL images
                to_pil = ToPILImage()
                batch_x_pil_list = [to_pil(img.cpu()) for img in batch_x]  
                
                distortions = [
                    gaussian_noise, shot_noise, impulse_noise, defocus_blur, motion_blur,
                    zoom_blur, fog, snow, elastic_transform, pixelate, jpeg_compression
                ]
        
                # Process each image in the batch
                for batch_idx, batch_x_pil in enumerate(batch_x_pil_list):
                    # List to hold the original and distorted images for the current batch image
                    augmented_images = []
                    
                    # Add the original image to the list
                    augmented_images.append(batch_x[batch_idx])

                    # Loop through the distortions and apply them to the current image
                    for function in distortions:
                        if function in [pixelate, jpeg_compression]:
                            # For functions returning tensors
                            img_processed = PILToTensor()(function(batch_x_pil)).to(dtype=batch_x.dtype).to("cuda") / 255.0
                        else:
                            # For functions returning images
                            img_processed = torch.tensor(function(batch_x_pil).astype(float) / 255.0, dtype=batch_x.dtype).to("cuda").permute(2, 0, 1)
        
                        # Append the distorted image
                        augmented_images.append(img_processed)
        
                    # Concatenate the original and distorted images
                    augmented_images_concatenated = torch.stack(augmented_images, dim=0)
                    all_batches.append(augmented_images_concatenated)
        
                # Concatenate all the augmented batches along the batch dimension
                batch_x_augmented = torch.cat(all_batches, dim=0)

                ###batch_x_augmented = batch_x_augmented.mean(dim=1, keepdim=True)  #should be uncomment for mnist
                
                # Repeat each label for the number of augmentations plus the original image
                batch_y_augmented = batch_y.repeat_interleave(len(distortions) + 1)
                
##                 
##                         # Extract the first 12 images to display (or fewer if there are less than 12 images)
##                         images_display = [batch_x_augmented[j] for j in range(min(12, batch_x_augmented.size(0)))]
##                 
##                         # Make a grid from these images
##                         grid = torchvision.utils.make_grid(images_display, nrow=len(images_display))  # Adjust nrow based on actual images
##                         
##                         # Save grid image with unique name for each batch
##                         torchvision.utils.save_image(grid, 'grid_image.png')
##                         

                if 'class-il' not in model.COMPATIBILITY:
                    outputs_augmented = model(batch_x_augmented, k)
                else:
                    if model.NAME == 'casp':
                        outputs_augmented, __temp = model.net.pcrForward(batch_x_augmented)
                    else:
                        outputs_augmented = model(batch_x_augmented)
                
                __augmented, pred_augmented = torch.max(outputs_augmented.data, 1)
                correct_augmented += torch.sum(pred_augmented == batch_y_augmented).item()
                total_augmented += batch_y_augmented.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs_augmented, dataset, k)
                    __augmented, pred_augmented = torch.max(outputs_augmented.data, 1)
                    correct_mask_classes_augmented += torch.sum(pred_augmented == batch_y_augmented).item()

        
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)


        accs_augmented.append(correct_augmented / total_augmented * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes_augmented.append(correct_mask_classes_augmented / total_augmented * 100)

    
    model.net.train(status)
    return accs, accs_mask_classes, accs_augmented, accs_mask_classes_augmented


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    task_class_ = {}
              
    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        name_of_run = f"{args.model}_{args.buffer_size}_{args.dataset}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, notes=args.notes, config=vars(args), name=name_of_run)
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)
    results, results_mask_classes = [], []
    results_augmented, results_mask_classes_augmented = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task, random_results_class_augmented, random_results_task_augmented = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    if hasattr(model, 'begin_train'):
        if model.NAME == 'casp' or model.NAME == 'meta_sp':
            model.begin_train(dataset)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()

        if t == 0:
            task_1 = train_loader
        if t == 1:
            
            # Step 1: Extract features, labels, and image hashes for Task 1 samples
            model.net.eval()
            features_list = []
            labels_list = []
            for data in task_1:
                with torch.no_grad():
                    inputs, labels, not_aug_inputs, index_ = data
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    labels = labels.to(model.device)
                    
                    # Extract features
                    if model.NAME == 'casp':
                        outputs, rep = model.net.pcrForward(not_aug_inputs)
                    else:
                        outputs, rep = model.net.forward(not_aug_inputs, returnt='all')
                    

                    # Move features to CPU and store
                    features_list.append(rep.cpu())
                    labels_list.append(labels.cpu())
            
            # Concatenate all features and labels
            features_list = torch.cat(features_list)
            labels_list = torch.cat(labels_list)


            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_list.numpy())
            
            # Initialize t-SNE with desired parameters
            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
            
            # Fit and transform the features
            features_2d = tsne.fit_transform(features_scaled)




            # Extract features
            if model.NAME == 'casp':
                outputs, rep = model.net.pcrForward(model.buffer.examples)
            else:
                outputs, rep = model.net.forward(model.buffer.examples, returnt='all')

            # Concatenate features and labels
            features_buffer = rep
            labels_buffer = model.buffer.labels.cpu().numpy()



            
            # Convert features to NumPy arrays
            features_task1_np = features_list.numpy()
            features_buffer_np = features_buffer.detach().cpu().numpy()
            
            # Set a small tolerance for matching features
            tolerance = 1e-6
            
            # Initialize a list to store indices of buffer samples in task 1 samples
            buffer_indices_in_task1 = []
            
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(features_task1_np)
            
            # Find the nearest neighbor in task1 features for each buffer feature
            distances, indices = nbrs.kneighbors(features_buffer_np)
            
            # Apply tolerance to filter out non-matching features
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if dist[0] < tolerance:
                    buffer_indices_in_task1.append(idx[0])
                else:
                    print(f"Buffer sample {i} did not match any task 1 sample within tolerance.")
            
            # Convert to a NumPy array
            buffer_indices_in_task1 = np.array(buffer_indices_in_task1)

            buffer_mask = np.zeros(len(labels_list), dtype=bool)
            buffer_mask[buffer_indices_in_task1] = True
            print("buffer_mask.size", buffer_mask.size)
            

            
            # Prepare color mapping
            classes = np.unique(labels_list)
            num_classes = len(classes)
            cmap = cm.get_cmap('tab10', num_classes)
            label_to_color = {label: cmap(i) for i, label in enumerate(classes)}
            colors = np.array([label_to_color[label.item()] for label in labels_list])
            
            # Plotting
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Non-buffer samples (circles)
            ax.scatter(
                features_2d[~buffer_mask, 0],
                features_2d[~buffer_mask, 1],
                c=colors[~buffer_mask],
                marker='o',
                alpha=0.7,
                edgecolors='none'
            )
            
            # Buffer samples (triangles)
            ax.scatter(
                features_2d[buffer_mask, 0],
                features_2d[buffer_mask, 1],
                c=colors[buffer_mask],
                marker='^',
                alpha=0.7,
                edgecolors='k'
            )
            
            # Legends
            marker_legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Non-buffer samples',
                       markerfacecolor='gray', markersize=10),
                Line2D([0], [0], marker='^', color='w', label='Buffer samples',
                       markerfacecolor='gray', markeredgecolor='k', markersize=10)
            ]
            
            class_legend_elements = [
                Line2D([0], [0], marker='s', color='w', label=str(class_id),
                       markerfacecolor=label_to_color[class_id], markersize=10)
                for class_id in classes
            ]
            
            # Add legends
            class_legend = ax.legend(
                handles=class_legend_elements,
                title='Classes',
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )
            ax.add_artist(class_legend)
            marker_legend = ax.legend(
                handles=marker_legend_elements,
                title='Sample Type',
                loc='best'
            )
            
            ax.set_title('t-SNE of Learned Representations from the First Task')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')


            
            plt.savefig("tsneERnew10")

            model.net.train()

            

        unique_classes_ = set()
        for _, labels_, _, _ in train_loader:
            unique_classes_.update(labels_.numpy())
            if len(unique_classes_)==dataset.N_CLASSES_PER_TASK:
                break
        
        task_class_.update({value: t for index, value in enumerate(unique_classes_)})

        
        if hasattr(model, 'begin_task'):
            if model.NAME == 'casp':
                model.begin_task(dataset, train_loader)
            else:
                model.begin_task(dataset)
        if t and not args.ignore_other_metrics:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            results_augmented[t-1] = results_augmented[t-1] + accs[2]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
                results_mask_classes_augmented[t-1] = results_mask_classes_augmented[t-1] + accs[3]

        scheduler = dataset.get_scheduler(model, args)
        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs, index_ = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    index_ = index_.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, index_)
                assert not math.isnan(loss)
                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            if scheduler is not None:
                scheduler.step()
            if hasattr(model, 'end_epoch'):
                if model.NAME == 'casp':
                    model.end_epoch(dataset, train_loader)
                else:
                    model.end_epoch(dataset)

        if hasattr(model, 'end_task'):
            model.end_task(dataset)
        
        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        results_augmented.append(accs[2])
        results_mask_classes_augmented.append(accs[3])

        
        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log:
            logger.log(mean_acc[:2], mean_acc[2:])
            logger.log_fullacc(accs[:2], accs[2:])

        if not args.nowand:
            d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1], 'RESULT_class_mean_accs_ood': mean_acc[2], 'RESULT_task_mean_accs_ood': mean_acc[3],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])},
                **{f'RESULT_class_acc_ood_{i}': a for i, a in enumerate(accs[2])},
                **{f'RESULT_task_acc_ood_{i}': a for i, a in enumerate(accs[3])}}

            wandb.log(d2)


    confidence_by_task_ = {task_id:0 for task_id in range(dataset.N_TASKS)}
    confidence_by_class_ = {class_id:0 for class_id in range(dataset.N_TASKS*dataset.N_CLASSES_PER_TASK)}
    for j in range(len(model.buffer)):
        confidence_by_task_[task_class_[model.buffer.labels[j].item()]] += 1
        confidence_by_class_[model.buffer.labels[j].item()] += 1
        
    ##print("confidence_by_task_", confidence_by_task_)
    ##print("confidence_by_class_", confidence_by_class_)



              

    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes, results_augmented, results_mask_classes_augmented)
        logger.add_forgetting(results, results_mask_classes, results_augmented, results_mask_classes_augmented)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class, results_mask_classes, random_results_task,
                          results_augmented, random_results_class_augmented, results_mask_classes_augmented, random_results_task_augmented)
        
    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
