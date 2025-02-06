import numpy as np

from time import time

import torch
from torchvision.ops import box_iou


class Deployable_Network(torch.nn.Module):
    '''Experimental class attempting to tackle several model problems at deployment. Includes two forms of bounding box ensembling 
    (non-maximum supression and weighted bounding box fusion) to tackle the model making too many predictions. Includes prioritisation system 
    to prioritise predicting class over another if any ambiguity. Includes experimental section attempting to run model on a moving belt - using belt
    speed and camera properties, notices if the same object appears translated by the belt in two consecutive image and only outputs bounding boxes for
    ejection one the object leaves the field of view'''

    def __init__(self, ObjectDetector, similar_classes):
        '''Initialises the deployable model
        
        INPUT
            ObjectDetector : Base object detector. Must be in pytorch format, outputting a dictionay of bounding boxes, category labels and prediction scores
            similar classes : List of tuples containing similar classes. For instance, (1,3) will instruct category ID 1 to be prioritised over 3'''

        super().__init__()

        #Initialise object detector
        self.ObjectDetector = ObjectDetector

        #List of similar class tuples
        self.similar_classes = similar_classes
        self.num_classes = self.ObjectDetector.roi_heads.box_predictor.cls_score.out_features
    
        #Ensure in evaluation mode
        self.eval()

        #Initially assume not live inference
        self.live_inference = False


    def init_live_inference(self, image_dimension, camera_fov, camera_height, belt_speed, belt_direction):
        '''Initialises live inference on a belt moving at a constant speed perpendictular to a given image edge 

        INPUT
            image_dimension : Tuple giving image dimension in pixels (height, width)
            camera_fov : Horizontal camera field of view in radians
            camera height : Camera height above belt in meters
            belt_speed : Speed of belt in meters/second
            belt direction : String giving direction of belt movement with respect to the image. Choose up, down, left or right'''

        self.belt_direction = belt_direction   #Direction of belt travel wrt plotted image
        self.distance = 0            #Distance belt has travelled between inference 
        self.live_inference = True
        self.image_height = image_dimension[0]
        self.image_width = image_dimension[1]
        
        #Convert belt speed to pixels per second
        image_width_meters = 2 * camera_height * np.tan(camera_fov/2)
        meter_to_pixel = image_width_meters / self.image_width
        self.belt_speed = belt_speed / meter_to_pixel

        self.keep_predictions = {'boxes' : torch.zeros((0,4)), 'scores' : torch.zeros(0), 'labels' : torch.zeros(0)}   #predictions within camera view
        self.return_boxes = {'boxes' : torch.zeros((0,4)), 'scores' : torch.zeros(0), 'labels' : torch.zeros(0)}       #predictions out of camera view
        self.recent_time = time() #Time of most recent inference


    def forward(self, images, score_thresh=0.5, ensemble_boxes='wbf'):
        '''Performs inference on list of images using the initialsied object detctor. Predictions are fused using either weighted boundng box fusion 
        or non-maximum supression and outputted as a dictionary of boxes, scores and labels
        
        INPUT
            images : List of pytorch tensors images on which to perform inference. Should be appropriately preprocessed
            score_thresh : Minimum score value for a prediction, bellow which all predictions are cut off
            ensemble_boxes : Method to ensemble bounding boxes. Choose 'wbf' for weighted bounding box fusion and 'nms' for non-maximum supression
            
        OUTPUT
            predictions : List of predictions for each image. Predictions formatted as a dictionary containing boxes, scores and labels'''

        if self.training:
            raise RuntimeError("Model should only be used for inference. Ensure model in evaluation mode with .eval() method")

        #Run object detection
        with torch.no_grad():
            predictions = self.ObjectDetector(images)

        #Box ensembling for each class
        for i, prediction in enumerate(predictions):

            #Filter by score threshold
            mask = prediction['scores'] > score_thresh
            prediction['boxes'], prediction['scores'], prediction['labels'] = prediction['boxes'][mask], prediction['scores'][mask], prediction['labels'][mask]

            if ensemble_boxes == 'nms':
                predictions = self.perform_nms(prediction)

            elif ensemble_boxes == 'wbf':
                prediction = self.perform_weighted_boxes_fusion(prediction)

            for group in self.similar_classes:
                #Check if group nums are found in prediction
                if all(num in prediction['labels'] for num in group):
                    prediction = self.fine_classification(prediction, group)

        predictions[i] = prediction

        if self.live_inference:
            self.run_live_inference(predictions[0])

        return predictions
    
    
    def perform_nms(self, prediction, thresh_IoU=0.5, ordered=True):
        '''Performs non-maximum supression on each category of the inputted prediction
        
        INPUT
            prediction : Dictionary containing prediction boxes, scores and labels
            thresh_IoU : Threshold IoU, used in NMS algorithm, above which two bounding boxes are considered to overlap
            ordered : Set to True if inputted predictions are already ordered by their score. If not, they will be ordered during NMS
            
        OUTPUT
            prediction : Final prediction which has undergone NMS'''

        keep_inds = []

        for i in range(self.num_classes):
            #Find relavent classes
            mask = prediction['labels'] == i

            #If less than two objects, no need for nms
            if mask.sum().item() <= 1:
                keep_inds.extend(torch.where(mask == 1)[0].tolist())

            #Else ompute nms for classes
            else:
                temp_inds = self.nms(prediction['boxes'][mask], prediction['scores'][mask], thresh_IoU=thresh_IoU, ordered=ordered)
                keep_inds.extend(torch.where(mask == 1)[0][temp_inds].tolist())

        #Filter out by nms
        prediction['boxes'], prediction['scores'], prediction['labels'] = prediction['boxes'][keep_inds], prediction['scores'][keep_inds], prediction['labels'][keep_inds]

        return prediction


    def nms(self, boxes, scores, thresh_IoU=0.5, ordered=True):
        '''Performs non-maximum supression algorithm. Moving from highest to lowest confidence bounding box, calculates the IoU between all bounding boxes
        and removes overlapping boxes of a lower confidence
        
        INPUT
            boxes : Tensor of bounding boxes
            scores : Tensor of bounding box confidences
            thresh_IoU : Threshold IoU, used in NMS algorithm, above which two bounding boxes are considered to overlap
            ordered : Set to True if inputted predictions are already ordered by their score. If not, they will be ordered during NMS
            
        OUTPUT
            keep_inds : List containing indexes for only high confidence bounding boxes which have not been removed'''

        #Order bboxes if needed
        if ordered:
            order = torch.tensor([i for i in range(len(scores))])    

        else:
            order = scores.argsort(descending=True)

        #For keeping filtered bboxes and scores
        keep_inds = []

        while len(order) > 0:
            #Select the maximum ind and add bbox to keep
            max_ind = order[0]
            keep_inds.append(max_ind.item())

            #Remove highest score bbox
            order = order[1:]

            IoU = box_iou(boxes[order], boxes[max_ind].unsqueeze(0)).squeeze(-1)

            mask = IoU < thresh_IoU
            order = order[mask]

        return keep_inds
    
    
    def perform_weighted_boxes_fusion(self, prediction, IoU_thresh=0.4, ordered=True, num_models=1):
        '''Performs weighted bounding box fusion (WBF) on each category of the inputted prediction
        
        INPUT
            prediction : Dictionary containing prediction boxes, scores and labels
            thresh_IoU : Threshold IoU, used in WBF algorithm, above which two bounding boxes are considered to overlap
            ordered : Set to True if inputted predictions are already ordered by their score. If not, they will be ordered during WBF
            num_models : Number of models which have performed inference on the image
            
        OUTPUT
            new_prediction : Final prediction, filtered from wbf'''
        
        new_prediction = {'boxes' : [], 'scores' : [], 'labels' : []}

        for i in range(self.num_classes):
            #Find relavent classes
            mask = prediction['labels'] == i

            if mask.sum().item() == 0:
                continue

            elif mask.sum().item() == 1:
                new_prediction['boxes'].append(prediction['boxes'][mask])
                new_prediction['scores'].append(prediction['scores'][mask])
                new_prediction['labels'].append(prediction['labels'][mask])

            else:
                fused_boxes, fused_scores = self.weighted_boxes_fusion(prediction['boxes'][mask], prediction['scores'][mask], IoU_thresh=IoU_thresh, ordered=ordered, num_models=num_models)

                new_prediction['boxes'].append(fused_boxes)
                new_prediction['scores'].append(fused_scores)
                new_prediction['labels'].append(torch.ones(len(fused_scores)) * i)

        #If any object in image, concatenate together
        if len(new_prediction['boxes']) != 0:
            new_prediction = {key: torch.cat(value, dim=0) for key, value in new_prediction.items()}

        return new_prediction


    def weighted_boxes_fusion(self, boxes, scores, IoU_thresh=0.5, ordered=True, num_models=1):
        '''Performs weighted bounding box fusion. Moving from the highest to lowest bounding boxes, calculates the IoU with all other bounding boxes and
        combines an overlapping boxes via a weighted sum (weighted by the box scores)
        
        INPUT
            boxes : Tensor of bounding boxes
            scores : Tensor of bounding box confidences
            thresh_IoU : Threshold IoU, used in WBF algorithm, above which two bounding boxes are considered to overlap
            ordered : Set to True if inputted predictions are already ordered by their score. If not, they will be ordered during WBF
            num_models : Number of models which produced the inputted predictions
            
        OUTPUT
            fused boxes : Tensor giving bounding box locations for final fused boxes
            fused scores : Tensor giving confidence scores for all fuseed bounding boxes'''
        
        #Order bboxes if needed
        if ordered:
            order = torch.tensor([i for i in range(len(scores))])    
        else:
            order = scores.argsort(descending=True)

        #Order the boxes by score
        max_ind = order[0]
        cluster_inds = [[max_ind.item()]]  #For indexing boxes and scores within cluster
        fused_boxes = [boxes[max_ind]]

        #Remove highest score bbox
        order = order[1:]

        while len(order) > 0:
            #Highest scoring box
            max_ind = order[0]
            order = order[1:]    #Remove highest score bbox

            #Find maximum IoU
            IoUs = box_iou(boxes[max_ind].unsqueeze(0), torch.stack(fused_boxes)).squeeze(0)
            max_IoU_ind = IoUs.argmax()

            #If intersection, add to cluster and update fused box
            if IoUs[max_IoU_ind] > IoU_thresh:
                cluster_inds[max_IoU_ind].append(max_ind.item())
                fused_boxes[max_IoU_ind] = self.fuse_boxes(boxes[cluster_inds[max_IoU_ind]], scores[cluster_inds[max_IoU_ind]])  #Calcualte nwe fused box

            #If not intersection, start new cluster
            else:
                fused_boxes.append(boxes[max_ind])
                cluster_inds.append([max_ind.item()])
        
        #Combine list into tensor
        fused_boxes = torch.stack(fused_boxes)

        #Average scores for cluster and scale by cluster size and num models used
        fused_scores = torch.tensor([scores[cluster].mean() for cluster in cluster_inds])

        if self.live_inference:
            #Number of bounding boxes fused to get each final bounding box
            cluster_size = torch.tensor([min(len(cluster), num_models) for cluster in cluster_inds])

            #Find predictions in section of image (belt) which has was in the previous image, but has been translated by the moving belt
            mask = self.find_duplicate_predictions(fused_boxes)

            fused_scores[mask] = fused_scores[mask] * cluster_size[mask] / num_models

        return fused_boxes, fused_scores
    

    def fuse_boxes(self, boxes, scores):
        '''Combines a set of bounding boxes via a weighted sum, weighted via box scores
        
        INPUT
            boxes : Tensor holding bounding boxes
            scores : Tensor holding scores
            
        OUTPUT
            Bounding box coords for final fused box'''

        #Numerator and denominator of weighted sum
        numerator = (boxes.T * scores).sum(dim=1)
        denominator = scores.sum()

        return numerator / denominator
    

    def find_duplicate_predictions(self, boxes):
        '''Calculates which predictions were in the a section of the image additionally appearing in the previouslu analysed image
        and which have been translated out of view of the camera via the belt
        
        INPUT
            boxes : Tensor containing predicted bounding boxes
            
        OUTPUT
            mask : Boolean mask giving which bounding boxes are in the target section'''

        #Find boxes in range of two predictions
        if self.belt_direction == 'up':
            mask = boxes[:,3] < self.image_height - self.distance

        elif self.belt_direction == 'down':
            mask = boxes[:,1] > self.distance

        elif self.belt_direction == 'left': 
            mask = boxes[:,2] < self.image_width - self.distance

        elif self.belt_direction == 'right':
            mask = boxes[:,0] > self.distance

        return mask
    

    def fine_classification(self, prediction, group, IoU_thresh=0.5):
        '''Performs fine classification on ambiguous predictions, prioritising the chosen category within group. If two predictions of ID within group overlap,
        only the priority ID will be kept
        
        INPUT
            prediction : Dictionary containing prediction boxes, scores and labels
            group : Tuple giving the group category IDs. First ID is the priority
            IoU : Threshold IoU above which two bounding boxes are considered to overlap
            
        OUTPUT
            prediction : Final prediction after prioritisation method'''
        
        priority_inds = [i for i, label in enumerate(prediction['labels']) if label == group[0]]    #Inds of priority class
        secondary_inds = [i for i, label in enumerate(prediction['labels']) if label == group[1]]   #Inds of secondary class
        keep = [i for i, label in enumerate(prediction['labels']) if label not in group]            #Leep all labels not currenly analysed

        for priority_ind in priority_inds:
            #Calc IoUs
            IoUs = box_iou(prediction['boxes'][priority_ind].unsqueeze(0), prediction['boxes'][secondary_inds]).squeeze(0)

            #Find all instances not overlapping
            secondary_inds = [secondary_ind for secondary_ind, IoU in zip(secondary_inds, IoUs) if IoU < IoU_thresh]
        
        keep += priority_inds + secondary_inds
        prediction['boxes'], prediction['labels'], prediction['scores'] = prediction['boxes'][keep], prediction['labels'][keep], prediction['scores'][keep]
        
        return prediction
    

    def run_live_inference(self, prediction):
        '''Runs live inference on a given prdiction, translating prevous predicrtions according to the belt speed and direction, combines these with 
        most recent predictions and returns predictions out of frame for ejection
        
        INPUT
            prediction : Dictionary containing most recent prediction boxes, scores and labels'''
        
        #Translate bounding boxes by belt speed
        self.translate_boxes()

        #Combine old and new boudning boxes
        combined_prediction = {'boxes' : torch.concatenate((self.keep_predictions['boxes'], prediction['boxes'])), 'scores' : torch.concatenate((self.keep_predictions['scores'], prediction['scores'])), 'labels' : torch.concatenate((self.keep_predictions['labels'], prediction['labels']))}
        self.keep_predictions = self.perform_weighted_boxes_fusion(combined_prediction, num_models=2)

    
    def translate_boxes(self):
        '''Translates bounding boxes from previous predictions. Uses the inputted belt speed, direction and the time passed since last prediction to 
        calculate how far the belt (and therefore the objects) have translated'''

        #Check if any bboxes yet
        if len(self.keep_predictions['boxes']) == 0:
            return None

        #Find distance translated by belt
        self.distance = self.belt_speed * (time() - self.recent_time)

        #Update bounding time (seconds)
        self.recent_time = time()

        #Translate bounding boxes and filter those outside image
        if self.belt_direction == 'up':
            self.keep_predictions['boxes'][:,1] -= self.distance
            self.keep_predictions['boxes'][:,3] -= self.distance
            self.filter_bboxes(lambda x: x[:,3] < 0)

        elif self.belt_direction == 'down':
            self.keep_predictions['boxes'][:,1] += self.distance
            self.keep_predictions['boxes'][:,3] += self.distance
            self.filter_bboxes(lambda x: x[:,1] < self.image_height)

        elif self.belt_direction == 'left': 
            self.keep_predictions['boxes'][:,0] -= self.distance
            self.keep_predictions['boxes'][:,2] -= self.distance
            self.filter_bboxes(lambda x: x[:,2] < 0)

        elif self.belt_direction == 'right':
            self.keep_predictions['boxes'][:,0] += self.distance
            self.keep_predictions['boxes'][:,2] += self.distance
            self.filter_bboxes(lambda x: x[:,0] > self.image_width)


    def filter_bboxes(self, condition):
        '''Filters bounding boxes based on a given condition, usually if the prediction is out of frame. If condition is satisfied, add to self.return_boxes
        for ejection. If condition is not satisfied, add to self.keep_predictions to combine with latest predictions'''

        #Find boxes outside of image
        mask = condition(self.keep_predictions['boxes'])

        #Seperate boxes in and out of image
        self.return_boxes = {'boxes' : self.keep_predictions['boxes'][mask], 'scores' : self.keep_predictions['scores'][mask], 'labels' : self.keep_predictions['labels'][mask]}
        self.keep_predictions = {'boxes' : self.keep_predictions['boxes'][~mask], 'scores' : self.keep_predictions['scores'][~mask], 'labels' : self.keep_predictions['labels'][~mask]}