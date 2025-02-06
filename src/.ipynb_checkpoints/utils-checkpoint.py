#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import torch
import torchvision.transforms.v2 as T
from torchvision import io, tv_tensors


def get_transform(train):
    '''Provides transformation and augmentation functions ready for training'''
    
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
    return tuple(zip(*batch))



class Custom_Dataset(torch.utils.data.Dataset):
    '''Modifying pytorch dataset to accept custom datasets.
    Includes functionality for both my object detection and instance segmentation. 
    Should modify that to accept bounding box coords later, since only dealing with object detection.'''
    
    def __init__(self, annotations_path, transforms = None, image_size=None):

        #Annotations contains annotations and filenames
        with open(annotations_path, 'r') as file:
            self.annotations = json.load(file)   

        self.location = self.annotations['location']
        self.transforms = transforms  #Any augmentation transforms

    
    def __getitem__(self, ind):
        
        #Load images from annotations path
        image_path = os.path.join(self.location, self.annotations['images'][ind]['file_name'])
        image = io.read_image(image_path)
        
        image_ind = ind

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
            "image_id": ind,
            "area": area,
            "iscrowd": iscrowd,
        }

        #If given, use our custom transforms function for data processing and augmentation
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
        
    
    def __len__(self):
        return len(self.annotations['images'])

    def convert_bbox(self, bbox):
        xmin, ymin, width, height = bbox.unbind(1)

        return torch.stack((xmin, ymin, xmin + width, ymin + height), dim=1)
    
    
    
class Evaluator():
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.dataset = {'images': [], 'categories': [], 'annotations': []}
        self.predictions = []
        
        self.annotation_id = 1

        #Append images, categories and annotations to dataset
        for images, targets in self.data_loader:
            for image, target in zip(images,targets):     #Second loop in case extra batch size
                self.append_target(image,target)

        #Find categories
        self.dataset['categories'] = data_loader.dataset.dataset.annotations['categories']

        #Convert dataset to COCO objects
        self.coco_targets = COCO()
        self.coco_targets.dataset = self.dataset
        
        with redirect_stdout(iopython.StringIO()):    #Supresses print
            self.coco_targets.createIndex()

        self.mAP50_history = [0]

    @property
    def mAP50(self):
        return self.mAP50_history[-1]
        

    def evaluate(self, model, device = 'cpu'):
        '''Evaluate models model using COCO method and updates assesment metrics'''

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
                self.append_prediction(prediction, target)

        #Define COCO evaluator and evaluate
        coco_predictions = self.coco_targets.loadRes(self.predictions)
        coco_eval = COCOeval(self.coco_targets, coco_predictions, iouType='bbox')
        
        #Evaluate and display
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if coco_eval.stats[1] != self.mAP50:    #if not equal to most recent mAP50 then we are training, hence update history
            self.mAP50_history.append(coco_eval.stats[1])
            
        

    def append_target(self, image, target):
        '''Appends target image information to self.dataset in COCO format'''
        
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


    def append_prediction(self, prediction, target):
        '''Appends current prediction to total predictions in COCO format'''

        for i in range(len(prediction['labels'])):
            formatted_prediction = {}
            formatted_prediction['image_id'] = target['image_id']
            formatted_prediction['category_id'] = prediction['labels'][i].item()
            formatted_prediction['score'] = prediction['scores'][i].item()
            formatted_prediction['bbox'] = self.convert_bbox(prediction['boxes'][i]).tolist()

            self.predictions.append(formatted_prediction)       

    
    def convert_bbox(self, box):
        '''Converts bounding box from Pytorch format (xmin, ymin, xmax, ymax) to COCO format (xmin, xmax, width, height)'''

        xmin, ymin, xmax, ymax = box
        
        return torch.tensor([xmin, ymin, xmax - xmin, ymax - ymin])


    def plot(self, figsize = (4,10), image_filename=None):
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        xs = np.arange(1, len(self.mAP50_history[1:])+1, 1)

        ax.plot(xs, self.mAP50_history[1:])   #Dont plot epoch 0
        
        ax.set_xlabel('epochs', fontsize=max(figsize)*1.5)
        ax.set_ylabel('mAP50', fontsize=max(figsize)*1.5)   

        if image_filename != None:
            plt.tight_layout()
            plt.savefig(image_filename, dpi=360)

            print('Saved mAP50 plot to {}'.format(image_filename))  

    
    def confusion_matrix(self, image_filename=None):
        pred_labels = []
        true_labels = []
        
        for image in self.dataset['images']:
            predictions = [prediction for prediction in self.predictions if prediction["image_id"] == image['id']]
            annotations = [annotation for annotation in self.dataset['annotations'] if annotation["image_id"] == image['id']]
        
            for prediction in predictions:
                IOUs = np.zeros(len(annotations))
        
                for i, annotation in enumerate(annotations):
                    IOUs[i] = calc_IOU(prediction['bbox'], annotation['bbox'], prediction['area'], annotation['area'])
        
                if any(map(lambda x: x >= .5, IOUs)):
                    pred_labels.append(prediction['category_id'])
                    true_labels.append(annotations[np.argmax(IOUs)]['category_id'])
        
        confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels)
        
        display_labels =  [category['name'] for category in self.dataset['categories'] if category['name'].lower() != 'background'] 
        
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = display_labels)
        
        cm_display.plot()
        plt.show() 

        if image_filename != None:
            plt.tight_layout()
            plt.savefig(image_filename, dpi=360)

            print('Saved confusion matrix to {}'.format(image_filename)) 
        
    
    
class Loss_Tracker():
    '''Return the average loss''' 
    def __init__(self, n_print=25):
        self.iteration = 0   #Iteration number
        self.epoch = 1       #Epoch number
        
        self.loss_cache = {'total_loss' : []}    #Stores all values of current epoch loss
        self.history = {'total_loss' : []}       #Stores epoch loss history

        self.n_print = n_print

    
    @property
    def loss(self):
        return np.array(self.loss_cache['total_loss']).mean()


    def update(self, losses):

        #If first iteration, initialise the loss types
        if self.iteration == 0 and self.epoch == 1:
            self.initialise_loss_types(losses)       
            
        self.iteration += 1

        self.loss_cache['total_loss'].append(sum(loss for loss in losses.values()).item())
        
        for key in self.loss_types:
            self.loss_cache[key].append(losses[key].item())

        if self.iteration % self.n_print == 0:
            print(f"Iteration #{self.iteration} loss: {self.loss}")

    
    def initialise_loss_types(self, losses):
        #Append each loss type as key
        for key in losses.keys():
            self.loss_cache[key] = []
            self.history[key] = []

        self.loss_types = losses.keys()


    def mark_epoch(self):
        #Append epoch average for each loss type
        self.history['total_loss'].append(self.loss)

        for key in self.loss_types:
            self.history[key].append(np.array(self.loss_cache[key]).mean())
            self.loss_cache[key] = []

        #Update markers
        self.epoch += 1
        self.iteration = 0


    def plot(self, keys=['total_loss'], figsize = (10,10), image_filename=None):
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

        axes[-1].set_xlabel('epoch', fontsize=figsize[0]*1.5) #Add tau label at bottom

        if image_filename != None:
            plt.tight_layout()
            plt.savefig(image_filename, dpi=360)

            print('Saved loss plot to {}'.format(image_filename))

        plt.show()



class Early_Stopper:
    
    def __init__(self, patience=100, min_delta=0, model_filename=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_mAP50 = 0

        if model_filename != None:
            self.model_filename = model_filename
        

    def early_stop(self, mAP50, model=None):
        #If mAP50 improves, restart counter
        if mAP50 > self.min_mAP50:
            self.min_mAP50 = mAP50
            self.counter = 0

            #If requested, save current best edition of model
            if model != None and self.model_filename != None:
                torch.save(model, self.model_filename)

                print('mAP50 improved. Saved new best model to {}'.format(self.model_filename))

        #If mAP50 fails to improve, check for early stop
        elif mAP50 < (self.min_mAP50 + self.min_delta):
            self.counter += 1
            
            if self.counter >= self.patience:
                return True
                
        return False


def calc_IOU(bbox1, bbox2, bbox1_area=None, bbox2_area=None):

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