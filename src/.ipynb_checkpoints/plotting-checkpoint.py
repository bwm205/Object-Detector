#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import random
import matplotlib.pyplot as plt

from time import time
from torchvision import io
from matplotlib import patches

from src.utils import get_transform


def convert_bbox(bbox):
    xmin, ymin, xmax, ymax = bbox.tolist()
    width = xmax - xmin
    height = ymax - ymin

    return [xmin, ymin, width, height]
    

def show_prediction(model, categories, path=None, data_loader=None, score_filter = .5, figsize=(8,8), image_filename=None):

    #Configure model
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    time1 = time()
    
    #If given path to specific image
    if path != None:
        image = io.read_image(path)
        eval_transform = get_transform(train=False)
        x = eval_transform(image).to(device)

    #If given test data for random image selection
    elif data_loader != None:
        image = data_loader.dataset[random.randrange(0,len(data_loader))][0]
        x = image.to(device)

    #Evaluate prediction
    with torch.no_grad():     #Use no_grad to avoid updating weights when validating
        x = x[:3, ...]    #:3 ensures we only use RGB values, nothing extra such as depth
    
        predictions = model([x, ])
        pred = predictions[0]

    inference_time = time() - time1

    #Scale pixel values for plotting
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)  
    image = image[:3, ...]   #Only take RGB vals

    #Scale for adjusting annotations to image size
    scale = int(max(image.shape)/70)

    #Colours for plotting
    colours = {1: 'tab:red' , 2 : 'tab:blue', 3 : 'tab:green', 4: 'tab:orange', 5 : 'tab:purple', 6 : 'tab:brown', 7 : 'tab:pink', 8 : 'tab:gray', 9 : 'tab:olive', 10 : 'tab:cyan'}

    time2 = time()

    #Create figure and show image
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
    
    plt.imshow(image.permute(1,2,0))

    #Filter through high confidence perdictions
    for label, score, bbox in zip(pred["labels"], pred["scores"], pred['boxes']):
        if score > score_filter:
            #Plot boudning box
            xmin, ymin, width, height = convert_bbox(bbox)
            rectangle = patches.Rectangle((xmin, ymin), width, height, edgecolor=colours[label.item()], facecolor="none")
            ax.add_patch(rectangle)

            #Adjust annotation position
            xcentre = xmin + width/2.0
            
            if ymin - 2*scale > 0:   #Check if annotation within image bounds
                ycentre = ymin - scale*1.1

            else:
                ycentre = ymin + height + scale*1.1

            #Label prediction on screen
            prediction = f"{categories[label.item()]}: {score:.3f}"
            l = ax.annotate(prediction, (xcentre, ycentre), fontsize=scale/2.5, fontweight='bold', color='white', ha='center', va='center')
            l.set_bbox(dict(facecolor=colours[label.item()], alpha=0.5, edgecolor=colours[label.item()]))

    ax.axis('off')

    plotting_time = time() - time2
    print('Speed : {:.1f}ms inference, {:.1f}ms plotting'.format(inference_time*1e3, plotting_time*1e3))

    if image_filename != None:
        plt.tight_layout()
        plt.savefig(image_filename, dpi=360)

        print('Saved prediction image to {}'.format(image_filename))