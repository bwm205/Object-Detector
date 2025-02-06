import os
import json
import numpy as np
import matplotlib.pyplot as plt
import io as iopython

from tqdm import tqdm
from contextlib import redirect_stdout
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from sklearn import metrics
from PIL import Image, ImageOps

import torch
import torchvision.transforms.v2 as T
from torchvision import  tv_tensors


def get_transform(train):
    '''Provides transformation and augmentation functions ready for training. Used as argument within datset initialisation.
    
    INPUT
        train : Input True to include data augmentation for training dataset.
        
    OUTPUT
        List of transforms composed together for use within dataset'''
    
    transforms = []
    
    #Training augmentations
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))         #Random horizontal flip
        transforms.append(T.RandomVerticalFlip(0.5))           #Random vertical flip
        transforms.append(T.ColorJitter(brightness=0.3))       #Adjusts brigtness by up to given factor

        
    transforms.append(T.ToDtype(torch.float, scale=True))   #Converts images to floating point dtype and scales them appropriately
    transforms.append(T.ToPureTensor())                     #Converts images to a pure tensor (not tv_tensor) once transforms have been applied
    
    return T.Compose(transforms)   #Composes transforms into a list for use


        
def collate_fn(batch):
    '''Collates together image and annotation data batches for when accessed by dataloader.
    
    INPUT
        batch : Data to be collated
        
    OUTPUT
        Collated images and target data'''

    return tuple(zip(*batch))



class Custom_Dataset(torch.utils.data.Dataset):
    '''Modified Pytorch dataset to accept data exported in COCO format. 
    When indexed, analyses COCO annotations file to extract image and corresponding bounding box annotations. 
    Should be converted to subset to subdivide data and then a dataloader for use in training
    '''
    
    def __init__(self, annotations_path, transforms = None):
        '''Initialises dataset from COCO annotations file
        
        INPUT
            annotations_path : Path to coco annotations file, produced directly from labelling software or from preprocessing
            transforms : get_transform() function giving list of pytorch transformations'''

        #Annotations contains annotations and filenames
        with open(annotations_path, 'r') as file:
            self.annotations = json.load(file)   

        self.location = self.annotations['location']
        self.transforms = transforms  #Any augmentation transforms

    
    def __getitem__(self, ind):
        '''Accesses given image and annotation data using annotations file. Converts bboxes from COCO to pytorch format and applies transformations to image.
          
        INPUT
            ind : Index corresponding to for accessing from list of images'''
        
        #Load images from annotations path
        image_path = os.path.join(self.location, self.annotations['images'][ind]['file_name'])
        image = self.load_image(image_path)
        
        image_ind = self.annotations['images'][ind]['id']

        #Find relevant annotations
        image_annotations = [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_ind]

        #Extract annotations
        boxes = torch.zeros((len(image_annotations), 4), dtype=torch.float32)
        labels = torch.zeros(len(image_annotations), dtype=torch.int64)
        area = torch.zeros(len(image_annotations), dtype=torch.float32)
        iscrowd = torch.zeros(len(image_annotations), dtype=torch.int64)
        
        
        for i, ann in enumerate(image_annotations):
            boxes[i] = torch.tensor(ann['bbox'])
            labels[i] = ann['category_id']
            area[i] = ann['area']
            iscrowd[i] = ann['iscrowd']

        #Convert bbox to correct foramt
        boxes = self.convert_bbox(boxes)

        #Wrap sample and targets (ie: target bounding boxes and masks) into torchvision tv_tensors for data augmentation and training
        image = tv_tensors.Image(image)

        #Combine in target dictionary
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=T.functional.get_size(image)),
            "labels": labels,
            "image_id": image_ind,
            "area": area,
            "iscrowd": iscrowd,
        }

        #If given, use our custom transforms function for data processing and augmentation
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
        
    
    def __len__(self):
        '''Number of images within dataset'''

        return len(self.annotations['images'])
    

    def convert_bbox(self, bbox):
        'Converts bounding box from COCO to pytorch format for use in training/testing'

        xmin, ymin, width, height = bbox.unbind(1)

        return torch.stack((xmin, ymin, xmin + width, ymin + height), dim=1)
    
    def load_image(self, path):
        '''Loads image from given path, converts to pytorch format and removes any transforms within the image metadata (such as rotations)
        which may produce conflict with bounding box coordinates
        
        INPUT
            path : Path to image file'''
        
        with Image.open(path) as img:
            # Automatically apply EXIF orientation
            img = ImageOps.exif_transpose(img)

            #Transform to pytorch tensor
            transform = T.ToTensor()
            tensor_image = transform(img)

            return tensor_image
    
    
    
class Evaluator():
    '''Used to assess the performance of an object detection model using COCO evaluation, a widely utilised methodology for assessing performance under varying conditions.
    The class assesses and tracks model precision and recall during training and provides appropriate plots, along with a confusion matrix'''
    
    def __init__(self, data_loader):
        '''Initialises evaluator, extracting annotations from test dataloader (ensuring in COCO format) and initialises COCO evaluator
        
        INPUT
            data_loader : Dataloader containing test data for evaluation'''

        self.data_loader = data_loader
        self.dataset = {'images': [], 'categories': [], 'annotations': []}
        self.predictions = []
        
        self.annotation_id = 1

        #Append images, categories and annotations to dataset
        for images, targets in self.data_loader:
            for image, target in zip(images,targets):     #Second loop in case extra batch size
                self.append_target(image,target)

        #Find categories
        try:
            self.dataset['categories'] = data_loader.dataset.dataset.annotations['categories']  #Try statement ensures works if subsets are used or not
        except:
            self.dataset['categories'] = data_loader.dataset.annotations['categories']

        #Convert dataset to COCO objects
        self.coco_targets = COCO()
        self.coco_targets.dataset = self.dataset
        
        with redirect_stdout(iopython.StringIO()):    #Supresses print
            self.coco_targets.createIndex()

        self.mAP50_history = [0]

    @property
    def mAP50(self):
        '''Returns most recent Mean Average Precision with IOU threshold = .5'''
        return self.mAP50_history[-1]
        

    def evaluate(self, model, device = 'cpu', score_filter=0.5):
        '''Evaluates models model using COCO method and updates assesment metrics
        
        INPUT
            model : Object detection model
            device : Device on which to run inference. Choose either cpu, cuda (for nvidia gpu) or mps (for apple silicon)
            score_filter : Score hreshold below which all predictions are discarded'''

        #Clear current predictions
        self.predictions = []

        #Configure model
        device = torch.device(device)
        model.to(device)
        model.eval()

        print('-------------------------------\nEvaluating\n-------------------------------')
        
        #Loop through validation data in batches
        for images, targets in tqdm(self.data_loader):
            #Send data to appropriate device
            images = list(image.to(device) for image in images)

            #Calculate model outputs
            with torch.no_grad():
                predictions = model(images)

            #If nescessary, convert data back to cpu
            if device.type != 'cpu':
                cpu_device = torch.device('cpu')
                predictions = [{k: v.to(cpu_device) for k, v in t.items()} for t in predictions]
            
            #Append predictions in COCO format
            for prediction, target in zip(predictions, targets):
                self.append_prediction(prediction, target, score_filter=score_filter)

        #Define COCO evaluator and evaluate
        coco_predictions = self.coco_targets.loadRes(self.predictions)
        self.coco_eval = COCOeval(self.coco_targets, coco_predictions, iouType='bbox')
        
        #Evaluate and display
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

        if self.coco_eval.stats[1] != self.mAP50:    #if not equal to most recent mAP50 then we are training, hence update history
            self.mAP50_history.append(self.coco_eval.stats[1])
        

    def append_target(self, image, target):
        '''Appends target image information to self.dataset in COCO format
        
        INPUT
            image : Image to be appended
            target : Image annotations to be appended'''
        
        #Format image
        formatted_image = {}
        formatted_image['id'] = target['image_id']
        formatted_image['width'] = image.shape[-1]
        formatted_image['height'] = image.shape[-2]

        self.dataset['images'].append(formatted_image)

        #Format target bbox/category
        for i in range(len(target['labels'])):
            formatted_target = {}
            formatted_target['id'] = self.annotation_id
            formatted_target['image_id'] = target['image_id']
            formatted_target['category_id'] = target['labels'][i].item()
            formatted_target['area'] = target['area'][i].item()
            formatted_target['iscrowd'] = target['iscrowd'][i].item()
            formatted_target['bbox'] = self.convert_bbox(target['boxes'][i]).tolist()

            self.dataset['annotations'].append(formatted_target)
            self.annotation_id += 1


    def append_prediction(self, prediction, target, score_filter=0.5):
        '''Appends current prediction to total predictions in COCO format
        
        INPUT
            prediction : Prediction from model
            target : Target annotations
            score_filter : Score hreshold below which all predictions are discarded'''

        for i in range(len(prediction['labels'])):
            if prediction['scores'][i].item() > score_filter:
                formatted_prediction = {}
                formatted_prediction['image_id'] = target['image_id']
                formatted_prediction['category_id'] = prediction['labels'][i].item()
                formatted_prediction['score'] = prediction['scores'][i].item()
                formatted_prediction['bbox'] = self.convert_bbox(prediction['boxes'][i]).tolist()

                self.predictions.append(formatted_prediction)       

    
    def convert_bbox(self, box):
        '''Converts bounding box from Pytorch format (xmin, ymin, xmax, ymax) to COCO format (xmin, xmax, width, height)
        
        INPUT
            box : Bounding box, formatted as pytorch tensor
            
        OUTPUT
            Formatted bounding box'''

        xmin, ymin, xmax, ymax = box
        
        return torch.tensor([xmin, ymin, xmax - xmin, ymax - ymin])


    def plot(self, figsize = (4,10), image_filename=None):
        '''Plots the mean average precision (mAP50) of the model tracked over all training epochs
        
        INPUT
            figsize : Size of the matplotlib figure
            image_filename : Path at which to save plot'''
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        xs = np.arange(1, len(self.mAP50_history[1:])+1, 1)

        ax.plot(xs, self.mAP50_history[1:])   #Dont plot epoch 0
        
        ax.set_xlabel('epochs', fontsize=max(figsize)*1.5)
        ax.set_ylabel('mAP50', fontsize=max(figsize)*1.5)

        if image_filename != None:
            plt.tight_layout()
            plt.savefig(image_filename, dpi=360)

            print('Saved mAP50 plot to {}'.format(image_filename))  

    
    def confusion_matrix(self, IOU_thresh=0.5, image_filename=None):
        '''Plots the confusion matrix for the current predictions, visually displaying the quality of predictions.
        
        INPUT
            IOU_thresh : Bounding box only associated with prediction if Intersection Over Union exceeds this threshold
            image_filename : Path at which to save plot'''

        pred_labels = []
        true_labels = []
        
        for image in self.dataset['images']:
            #predictions and annotations for given image
            predictions = [prediction for prediction in self.predictions if prediction["image_id"] == image['id']]
            annotations = [annotation for annotation in self.dataset['annotations'] if annotation["image_id"] == image['id']]
        
            for prediction in predictions:
                pred_labels.append(prediction['category_id'])

                #IOUs for prediction with all annotations
                IOUs = [calc_IOU(prediction['bbox'], annotation['bbox'], prediction['area'], annotation['area']) for annotation in annotations]

                #If any IOU crosses threshold, find max
                if len(IOUs) > 0 and max(IOUs) >= IOU_thresh:
                    max_ind = np.argmax(IOUs)
                    true_labels.append(annotations[np.argmax(IOUs)]['category_id'])
                    annotations.pop(max_ind)
                #If none meet threshold, add as false positive
                else:
                    true_labels.append(0)
            
            #Any remaining annotations, add false negative
            if len(annotations) > 0:
                for annotation in annotations:
                    pred_labels.append(0)  # Predicted as background
                    true_labels.append(annotation['category_id'])
        
        #Produce confusion matrix
        confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels)
    
        display_labels =  [category['name'] for category in self.dataset['categories']] 
        
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
        
        cm_display.plot()

        #If requested, save plot
        if image_filename != None:
            plt.tight_layout()
            plt.savefig(image_filename, dpi=360)

            print('Saved confusion matrix to {}'.format(image_filename))
    
    
class Loss_Tracker():
    '''Tracks and plots the model's loss over training epochs''' 

    def __init__(self):
        '''Initialises Loss Tracker'''

        self.iteration = 0   #Iteration number
        self.epoch = 1       #Epoch number
        
        self.loss_cache = {'total_loss' : []}    #Stores all values of current epoch loss
        self.history = {'total_loss' : []}       #Stores epoch loss history

    
    @property
    def loss(self):
        '''Returns the mean of the total losses for a given epoch'''
        return np.array(self.loss_cache['total_loss']).mean().item()


    def update(self, losses):
        '''Updates the loss history with the inputted losses.
        
        INPUT
            losses : Dictionary of losses from model. Should be formatted such as {'loss_classifier': tensor(0.0808)}'''

        #If first iteration, initialise the loss types
        if self.iteration == 0 and self.epoch == 1:
            self.initialise_loss_types(losses)       
            
        #One more iteration has passes
        self.iteration += 1

        #Log total loss
        self.loss_cache['total_loss'].append(sum(loss for loss in losses.values()).item())
        
        #Log individual loss types
        for key in self.loss_types:
            self.loss_cache[key].append(losses[key].item())

    
    def initialise_loss_types(self, losses):
        '''On first iteration, find names of all loss types for logging
        
        INPUT
            losses : Dictionary of loss types'''

        #Append each loss type as key
        for key in losses.keys():
            self.loss_cache[key] = []
            self.history[key] = []

        self.loss_types = losses.keys()


    def mark_epoch(self):
        '''Mark the end of the epoch and average losses to give epoch loss'''

        #Append epoch average for each loss type
        self.history['total_loss'].append(self.loss)

        for key in self.loss_types:
            self.history[key].append(np.array(self.loss_cache[key]).mean().item())
            self.loss_cache[key] = []

        #Update markers
        self.epoch += 1
        self.iteration = 0


    def plot(self, keys=['total_loss'], figsize = (10,10), image_filename=None):
        '''Plot either total loss or individual loss types, tracked over all training epochs
        
        INPUT
            keys : Losss types to plot. Either keep as total loss, or input individual loss types
            figsize : Size of matplotlib figure
            image_filename : Path at which to save plot'''

        #Create figure
        fig, axes = plt.subplots(len(keys), 1, figsize=figsize, sharex=True)
        
        if len(keys) <= 1:
            axes = [axes]

        #Loop through selected loss types and plot
        for i, key in enumerate(keys):
            ax = axes[i]

            xs = np.arange(1,self.epoch,1)
            ys = self.history[key]
            
            ax.plot(xs,ys)
            ax.set_ylabel(key, fontsize=max(figsize)*1.5)

        axes[-1].set_xlabel('epoch', fontsize=max(figsize)*1.5) #Add tau label at bottom

        if image_filename != None:
            plt.tight_layout()
            plt.savefig(image_filename, dpi=360)

            print('Saved loss plot to {}'.format(image_filename))

        plt.show()



class Early_Stopper:
    '''Determines whether to stop training early if model precision (mAP50) fails to improve'''
    
    def __init__(self, patience=100, max_delta=0):
        '''Initialises Early Stopper
        
        INPUT
            patience :  The number of epochs of failed improvement after which to stop training
            max_delta : Expected variance in mAP50. Standard variation in precision will not be penalised as failing to improve'''

        self.patience = patience      #Number of epochs after which to stop training
        self.max_delta = max_delta    #Tolerance for early stoper improvement
        self.counter = 0              #How many epochs since last best performance
        self.max_mAP50 = 0            #Current best mAP50
        self.improved = False         #Tracks if performance improved over epoch
        

    def early_stop(self, mAP50):
        '''Assess if precision has improved and hence whether to stop training
        
        INPUT
            mAP50 : The mean average precision of model, calculated with IOU threshold of .5 at the end of an epoch
            
        OUTPUT
            Boolean giving whether to stop training'''

        #If mAP50 improves, restart counter
        if mAP50 > self.max_mAP50:
            self.max_mAP50 = mAP50
            self.counter = 0
            self.improved = True

        #If mAP50 fails to improve, check for early stop
        elif mAP50 < (self.max_mAP50 + self.max_delta):
            self.counter += 1
            self.improved = False
            
            if self.counter >= self.patience:
                return True
                
        return False
    


def calc_IOU(bbox1, bbox2, bbox1_area=None, bbox2_area=None):
    '''Calculates the intersection over union between two bouding boxes (analagous to the overlap)
    
    INPUT
        bbox1 & bbox2 : Bounding boxes, given in as a pytorch tensors in pytorch format
        bbox1_area & bbox2_area : Areas of inputted bounding boxes. If not given, calculated within function

    OUTPUT
        iou : Calculated intersection over union'''

    #Find overlap mins and maxes
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    yB = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    #Calculate overlap
    intersection = max(0, xB - xA) * max(0, yB - yA)

    #Calculate bbox areas if not given
    if None in (bbox1_area, bbox2_area):
        bbox1_area = bbox1[2] * bbox1[3]
        bbox2_area = bbox2[2] * bbox2[3]

    #Calc IOU
    iou = intersection / (bbox1_area + bbox2_area - intersection)

    return iou 