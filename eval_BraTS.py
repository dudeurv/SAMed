from torchmetrics import Dice
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

def test_per_epoch(model, testloader, loss_fn, multimask_output, img_size):
    model.eval()
    loss_per_epoch = []
    num_classes = 4
    with torch.no_grad():
        for i_batch, (image_batch, label_batch) in enumerate(testloader):

            image_batch, label_batch = image_batch.unsqueeze(1).float().cuda(), label_batch.unsqueeze(1).cuda()
            image_batch = image_batch.repeat(1, 3, 1, 1)
            
            label_batch = F.interpolate(label_batch, size=(128, 128), mode='nearest') 
            label_batch = label_batch.squeeze(1)

            label_batch = torch.clamp(label_batch, 0, num_classes-1)
        
            logits = model(image_batch, multimask_output, img_size)
            loss = loss_fn(logits, labels)
            loss_per_epoch.append(loss.item())
            
    return torch.tensor(loss_per_epoch).mean().item()


# Define a function to calculate the confusion matrix from predictions and ground truths
def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    # Stack the ground truth and prediction arrays and transpose
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    # Compute the confusion matrix using histogram
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),  # Number of bins for each dimension (nr_labels)
        range=[(0, nr_labels), (0, nr_labels)]  # Range of labels
    )
    # Convert the confusion matrix to uint32 for consistency
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix

# Define a function to calculate the Dice coefficient from the confusion matrix
def calculate_dice(confusion_matrix):
    dices = []  # Initialize a list to store Dice scores for each class
    # Iterate over each class to calculate Dice score
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]  # Diagonal elements are true positives
        # Sum of the column for the class minus true positives gives false positives
        false_positives = confusion_matrix[:, index].sum() - true_positives
        # Sum of the row for the class minus true positives gives false negatives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        # The denominator in the Dice score formula
        denom = 2 * true_positives + false_positives + false_negatives
        # Handle the case where denominator is zero (to avoid division by zero)
        if denom == 0:
            dice = 0
        else:
            # Dice score calculation: 2 times the number of true positives divided by the denominator
            dice = 2 * float(true_positives) / denom
        dices.append(dice)  # Append the Dice score for the current class to the list
    return dices
    
    
# Define a function to test the model for an epoch and visualize the results
def vis_per_epoch(model, testloader, multimask_output, img_size):
    model.eval()  
    num_classes = 4
    fig, axs = plt.subplots(len(testloader), 3, figsize=(1*3, len(testloader)*1), subplot_kw=dict(xticks=[],yticks=[]))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)

    with torch.no_grad():
        
        for i_batch, (image_batch, label_batch) in enumerate(testloader):

            image_batch, label_batch = image_batch.unsqueeze(1).float().cuda(), label_batch.unsqueeze(1).cuda()
            image_batch = image_batch.repeat(1, 3, 1, 1)
            
            label_batch = F.interpolate(label_batch, size=(128, 128), mode='nearest') 
            label_batch = label_batch.squeeze(1)

            label_batch = torch.clamp(label_batch, 0, num_classes-1)
        
            logits = model(image_batch, multimask_output, img_size)

            prob = F.softmax(logits, dim=1)
            pred_seg = torch.argmax(prob, dim=1)
     
            confusion_matrix += calculate_confusion_matrix_from_arrays(pred_seg.cpu(), labels.cpu(), num_classes)
    
            img_num = 0
            axs[batch_idx, 0].imshow(images[img_num, 0].cpu().numpy(), cmap='gray')
            axs[batch_idx, 1].imshow(labels[img_num].cpu().numpy(), cmap='gray')
            axs[batch_idx, 2].imshow(pred_seg[img_num].cpu().numpy(), cmap='gray')

    confusion_matrix = confusion_matrix[1:, 1:]

    dices_per_class = {'dice_cls:{}'.format(cls + 1): dice
                for cls, dice in enumerate(calculate_dice(confusion_matrix))}

    plt.axis('OFF')
    plt.tight_layout()
    plt.show()
    return dices_per_class
