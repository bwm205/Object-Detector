import torch
import random
import matplotlib.pyplot as plt

from time import time
from torchvision import io
from matplotlib import patches

from src.utils import get_transform
    

def show_prediction(model, categories, path=None, data_loader=None, score_filter = .5, figsize=(10,10), image_filename=None, device='cpu'):
    '''Plots the prediction from the model on a given image, either loaded randomly from a dataloader or from the path provided
    
    INPUT
        model : Pytorch model to perform inference
        catagories : Dictionary of category names for plot labels. Formatted such as {0 : 'Background', 1 : 'Armature'}
        path : Path to image file
        data_loader : Dataloader from which a random image is selected
        score_filter : Assign from 0-1. All predictions bellow filter are discarded
        figsize : Tuple for matplotlib figsize
        image_filename : If given, save image to this path
        device : Device on which to perform inference. Choose either cpu, cuda (for nvidia gpu) or mps (for apple silicon)'''

    #Configure model
    device = torch.device(device)
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

    plot(pred, image, categories, score_filter=score_filter, figsize=figsize, image_filename=image_filename)

    time2 = time()
    plotting_time = time() - time2
    print('Speed : {:.1f}ms inference, {:.1f}ms plotting'.format(inference_time*1e3, plotting_time*1e3))


def plot(prediction, image, categories, score_filter=0.5, figsize=(10,10), image_filename=None):
    '''Plots bourding boxes from a given prediction onto the image provided
    
    INPUT
        prediction : Prediction dictionary outputted from pytorch model, containing boxes, scores and labels for each prediction
        image : Pytorch tensor holding image on which inference was performed
        categories : Dictionary of category names for plot labels. Formatted such as {0 : 'Background', 1 : 'Armature'}
        score_filter : Assign from 0-1. All predictions bellow filter are discarded
        figsize : Tuple for matplotlib figsize
        image_filename : If given, save image to this path'''

    scale = int(max(image.shape)/70)
    colours = {1: 'tab:red' , 2 : 'tab:blue', 3 : 'tab:green', 4: 'tab:orange', 5 : 'tab:purple', 6 : 'tab:brown', 7 : 'tab:pink', 8 : 'tab:gray', 9 : 'tab:olive', 10 : 'tab:cyan'}

    #Create figure and show image
    fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)
    
    plt.imshow(image.permute(1,2,0))

    for label, score, bbox in zip(prediction["labels"], prediction["scores"], prediction['boxes']):
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
            pred = f"{categories[label.item()]}: {score:.3f}"
            l = ax.annotate(pred, (xcentre, ycentre), fontsize=max(figsize)*.8, fontweight='bold', color='white', ha='center', va='center')
            l.set_bbox(dict(facecolor=colours[label.item()], alpha=0.5, edgecolor=colours[label.item()]))

    ax.axis('off')

    #If filename give, save image
    if image_filename != None:
        plt.tight_layout()
        plt.savefig(image_filename, dpi=360)

        print('Saved prediction image to {}'.format(image_filename))


def convert_bbox(bbox):
    '''Converts bounding box from pytorch format to format used for matplotlib plotting (from xmin, ymin, xmax, ymax to xmin, ymin, width, height)
    
    INPUT
        bbox : Bouding box tensor to be converted
        
    OUTPUT
        Converted bounding box, formatted as a list'''

    xmin, ymin, xmax, ymax = bbox.tolist()
    width = xmax - xmin
    height = ymax - ymin

    return [xmin, ymin, width, height]