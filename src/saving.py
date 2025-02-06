import json
import os
import shutil

import torch


class Model_Manager():
    '''Provides functionality for saving and loading pytorch models. 
    Tracks model versions when saving, saves training data/results and provides functionality for model backups during training'''
    
    def __init__(self, model_location):
        '''Initialises model manager
        
        INPUT
            model_location : Path to folder where model versions are located'''

        #Change directory to when model versions are located
        self.change_directory(model_location)

        self.backup_version = None   #For use when backing up during training
        self.best_predictions = None   #For tracking evaluation metrics for best model
        self.best_coco_eval = None

    @property
    def latest_version(self):
        '''Searches current directory to find latest model version'''
        #All model filenames, ignoring hidden files
        files = [name for name in os.listdir(self.model_location) if not name.startswith('.')]

        #If no versions, return 0 as current version
        if len(files) == 0:
            return 0

        #Get versions by trimming 'v_' from front of model filename
        versions = [file[2:] for file in files if file[:2] == 'v_']

        return int(max(versions))
        

    def change_directory(self, model_location):
        '''Change current saving directory
        
        INPUT
            model_location : Path to folder where model versions are located'''
        
        #See if model filename contains location. If not, save to current working directory
        location, _ = os.path.split(model_location)
        if len(location) == 0:
            location = os.getcwd()
            self.model_location = os.path.join(location, model_location)

        else:
            self.model_location = model_location

        #If doesn't exist, make file
        make_folder(self.model_location)


    def save(self, model, version=None, data_loaders=None, data_loader_names=None, loss_tracker=None, evaluator=None):
        '''Saves model to latest version or the version given, along with train/test/val split used during training (saved as .json files) and any loss/evaluation plots
        
        INPUT
            model : Pytorch model to save
            version : Integer for desired model version to save as. If None, save as latest version
            data_loaders : List of custom pytorch dataloaders from which to extract subset data
            data_loader_names : Names of data_loaders, used for filenames when saving training split
            loss_tracker : Loss tracker object to save loss over training plot
            evaluator : Evaluator object to save mAP50 over training plot and confusion matrix'''
        
        #Find latest version
        if version == None:
            version = self.latest_version

            #Check if currently backing up to this version?
            if version != self.backup_version:
                version += 1

        #Folder for model
        model_filename = os.path.join(self.model_location, 'v_' + str(version))

        #Check if version already exists, if yes then request to overwrite
        if os.path.isdir(model_filename) and version != self.backup_version:   #if overiding backup, don't ask for permission to remove
            rewrite = ''
            while rewrite != 'y' and rewrite != 'N':
                rewrite =  input('Version {} already exists. Would you like to overwrite? [y/N]'.format(version))
                if rewrite == 'y':
                    shutil.rmtree(model_filename)
                elif rewrite == 'N' :
                    print('Input new version or use version=None for automatic version control')
                    return None
                else:
                    print('Invalid input')
                                    
        #If doesn't exist, make file
        make_folder(model_filename)

        #Save model
        torch.save(model, os.path.join(model_filename, 'weights.pt'))

        #Save loss plots
        if loss_tracker != None or evaluator != None:
            #Location of model
            plot_path = os.path.join(model_filename, 'plots')
            if not os.path.isdir(plot_path):
                os.mkdir(plot_path)

            #Contains final results for assessing model performance
            final_results = {}

            #Save confusion matrix + mAP50 over time
            if evaluator != None:
                #If saved, assign the evaluation from best model to current evaluator so best evaluation metrics are saved
                if self.best_predictions != None and self.best_coco_eval != None:
                    evaluator.predictions = self.best_predictions
                    evaluator.coco_eval = self.best_coco_eval

                evaluator.confusion_matrix(IOU_thresh=0.5, image_filename=os.path.join(plot_path, 'confusion matrix.png'))
                evaluator.plot(figsize = (10,10), image_filename=os.path.join(plot_path, 'mAP50 plot.png'))

                final_results['Average Precision'] = evaluator.coco_eval.stats[1]
                final_results['Average Recall'] = evaluator.coco_eval.stats[8]

            #Save loss plots
            if loss_tracker != None:
                loss_tracker.plot(keys=['total_loss'], figsize = (10,10), image_filename=os.path.join(plot_path, 'total loss.png'))
                loss_tracker.plot(keys=loss_tracker.loss_types, figsize = (8,16), image_filename=os.path.join(plot_path, 'loss types.png'))
                final_results['loss'] = loss_tracker.history

            #Save final results
            with open(os.path.join(model_filename, 'results.json'), 'w') as f:
                    json.dump(final_results, f)

        #If given, use dataloaders to save train, test, val ... images as .json files
        if data_loaders != None:
            #Check if file available for storing .json logs
            path = os.path.join(model_filename, 'previous_annotations')
            os.mkdir(path)

            #If names not provided, assume given in train_test_val
            if data_loader_names == None:
                data_loader_names = ['train', 'test', 'val'][:len(data_loaders)]

            #Loop through dataloaders and produce .json files for relevant data
            for data_loader, data_loader_name in zip(data_loaders, data_loader_names):
                annotations = self.analyse_dataloader(data_loader)

                annotations_path = os.path.join(path, data_loader_name + '.json')
                with open(annotations_path, 'w') as f:
                    json.dump(annotations, f)

        print('Saved model to {}'.format(model_filename))

        #If not backing up, remove any legacy backups
        backup_filename = os.path.join(model_filename, 'backup.pt')
        if os.path.isfile(backup_filename):
            os.remove(backup_filename)
            self.backup_version = None   #Since we are no longer backing up to that version


    def load(self, version=None, give_annotations=False):
        '''Loads either the latest model or the version given, along with .json files for the respective train/test split used during training (used to produce dataloaders)
        
        INPUT
            version : Integer for desired model version to load. If None, load latest version
            give_annotations : If True, return the .json files corresponding to subset split used during training
            
        OUTPUT
            model : Pytorch model to load'''
        
        #Find latest model
        if version == None:
            version = self.latest_version

        #Folder for model   
        model_filename = os.path.join(self.model_location, 'v_' + str(version))

        #Try loading either weights or backup
        if os.path.isfile(os.path.join(model_filename, 'weights.pt')):
            model = torch.load(os.path.join(model_filename, 'weights.pt'), weights_only=False)

        elif os.path.isfile(os.path.join(model_filename, 'backup.pt')):
            model = torch.load(os.path.join(model_filename, 'backup.pt'), weights_only=False)

        else:
            print('Model not found within {}. Check model_filename is a valid path'.format(model_filename))
            return None
        
        print('Model loaded from {}'.format(model_filename))

        #Give paths to annotations used in training
        if give_annotations:
            annotation_paths = []

            #Paths to annotations
            annotations_dir = os.path.join(model_filename, 'previous_annotations')
            filenames = os.listdir(annotations_dir)

            for filename in filenames:
                annotations_path = os.path.join(annotations_dir, filename)
                annotation_paths.append(annotations_path)

            return model, annotation_paths
        
        return model
    

    def backup(self, model, version, evaluator=None):
        '''Backup model weights. For use during training
        
        INPUT
            model : Pytorch model to backup
            version : Integer for desired model version to backup to. If None, backup to new version'''
        
        #Find latest version
        if version == None:
            version = self.latest_version

            #Check if currently backing up to this version?
            if version != self.backup_version:
                version += 1

        #Folder for model
        model_filename = os.path.join(self.model_location, 'v_' + str(version))

        #If doesn't exist, make file
        make_folder(model_filename)

        #Save model
        torch.save(model, os.path.join(model_filename, 'backup.pt'))

        #Assign as current backup version
        self.backup_version = version

        #Save best evaluator so far
        if evaluator != None:
            self.best_predictions = evaluator.predictions
            self.best_coco_eval = evaluator.coco_eval

        print('Backed up model to {}'.format(model_filename))
    

    def analyse_dataloader(self, data_loader):
        '''Extract images, their respective annotations and categories from dataloaders and formats a COCO .json file
        
        INPUT
            data_loader : Custom dataloader from which to extract subset data'''

        #All annotations
        try:
            all_annotations = data_loader.dataset.dataset.annotations
        except:
            all_annotations = data_loader.dataset.annotations

        inds = data_loader.dataset.indices
        
        #List all annotations to keep
        keep_ids = [all_annotations['images'][ind]['id'] for ind in inds]

        #Search through and keep desired annotations
        new_annotations = {'images' : [], 'categories' : all_annotations['categories'], 'annotations' : [], 'location' : all_annotations['location']}
        new_annotations['images'] = [image for image in all_annotations['images'] if image['id'] in keep_ids]
        new_annotations['annotations'] = [annotation for annotation in all_annotations['annotations'] if annotation['image_id'] in keep_ids]

        return new_annotations
    


def make_folder(path):
    '''Creates folder at given path
    
    INPUT
        path : Path for desired folder'''
    #Check if path exists
    if not os.path.isdir(path):
        location, _ = os.path.split(path)
        make_folder(location)            #If not, move back through path to check if previous exists
        
        #Create path
        os.mkdir(path)