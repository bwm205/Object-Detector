import os
import shutil
import PIL
import json

from tqdm import tqdm


def preprocess(location, folder_names, image_size=None):
    '''Preprocesses data, combining multiple datasets (COCO format) and formatting appropriately to produce a single .json file for initialising a custom pytorch dataset

    INPUT
        location : Directory where datasets are stored
        folder_names : Names of folders for all datasets
        image_size : Size down to which all images will be downscaled. Given as a single integer
    
    OUTPUT
        new_annotations_path : Annotations .json file giving all image locations and annotations in COCO format. Use by inputting into dataset'''

    assert type(folder_names) == list, 'folder_names variable should be formatted as a list'

    #If multiple dataset folders given, concatenate together
    if len(folder_names) > 1:
        annotations = concatenate_datasets(location, folder_names)
        annotations['location'] = location

    elif len(folder_names) == 1:
        annotations_path = os.path.join(location, folder_names[0], "result.json")
        with open(annotations_path, 'r') as file:
            annotations = json.load(file)
            annotations['location'] = os.path.join(location, folder_names[0])

    #Ensure all annotations are consecutive
    make_consecutive(annotations)

    #Format to ensure background label = 0
    is_background = False
    
    for category in annotations['categories']:
        if category['name'].lower() == 'background':
            swap_labels(annotations, 0, category['id'])
            
            is_background = True
            break

    #If background not present in labels, make to 0th label
    if not is_background:
        insert_background_label(annotations)

    #If given size, rescale images
    if image_size != None:
        rescale(annotations, location, image_size)

    new_annotations_path = os.path.join(os.getcwd(), "formatted_annotations.json")
    with open(new_annotations_path, 'w') as f:
        json.dump(annotations, f)

    return new_annotations_path


def concatenate_datasets(location, folder_names):
    '''Combines annotations from several datasets (COCO format), relabeling annotations, finding all unique classes and updating image paths for use within training.
    
    INPUT
        location : Directory where datasets are stored
        folder_names : Names of folders for all datasets
        
    OUTPUT
        new_annotations : '''

    new_annotations = {'images' : [], 'categories' : [], 'annotations' : []}  #New .json file with all data
    all_categories = []  #List of categories for reference

    #For editing ids
    num_images = 0
    num_annotations = 0

    #Loop through all datasets
    for folder_name in folder_names:
        #Annotations contains annotations and filenames
        annotations_path = os.path.join(location, folder_name, "result.json")
        with open(annotations_path, 'r') as file:
            annotations = json.load(file)

        #Append all images to new annotations with new ids and paths
        for image in annotations['images']:
            image['id'] += num_images
            image['file_name'] = os.path.join(folder_name, image['file_name'])   #images in category dataset hence path must lead there

            new_annotations['images'].append(image)

        category_reference = {}   #For changing categories in annotations

        #Loop through categories to check if new
        for category in annotations['categories']:
            name = category['name'].lower()
            
            if name in all_categories:
                category_reference[category['id']] = all_categories.index(name)

            elif name not in all_categories:
                category_reference[category['id']] = len(all_categories)
                new_annotations['categories'].append({'id' : len(all_categories), 'name' : category['name']})
                all_categories.append(name)

        #Append all annotations to new annotations with new ids
        for annotation in annotations['annotations']:
            annotation['id'] += num_annotations
            annotation['image_id'] += num_images
            annotation['category_id'] = category_reference[annotation['category_id']]

            new_annotations['annotations'].append(annotation)

        #Update number of annotations
        num_images = len(new_annotations['images'])
        num_annotations = len(new_annotations['annotations'])

    return new_annotations


def make_consecutive(annotations):
    '''Ensures all category IDs are in consecutive order
    
    INPUT
        annotations : Dictionary containing image data, image annotations and list of all categories'''

    old_categories = [category['id'] for category in annotations['categories']]

    total_categories = len(annotations['categories'])
    i = 0

    while i < total_categories:
        #Search for id
        for category in annotations['categories']:
            #If found, break and begin search for next
            if category['id'] == i:
                i += 1
                break
    
        #If not found, reduce all nescessary ids by 1
        else:
            for category2 in annotations['categories']:
                    if category2['id'] >= i: category2['id'] -= 1

    #Sort to ensure in order
    annotations['categories'] = sorted(annotations['categories'], key=lambda x: x['id'])

    #For referencing annotations to prevent continuous looping
    new_categories = [category['id'] for category in annotations['categories']]
    category_reference = {old : new for old, new in zip(old_categories, new_categories)}

    #Change annotations to new ids
    if old_categories != new_categories:
        for annotation in annotations['annotations']:
            annotation['category_id'] = category_reference[annotation['category_id']]


def swap_labels(annotations, id1, id2):
    '''Relabels all annotations with category id1 to category id2
    
    INPUT
        annotations : Dictionary containing image data, image annotations and list of all categories
        id1 & id2 : Integer category ids to be swapped'''
        
    #Swap labels in listed categories
    annotations['categories'][id1]['name'], annotations['categories'][id2]['name'] = annotations['categories'][id2]['name'], annotations['categories'][id1]['name']

    #Swap id1 and id2 labels in all annotations con
    for annotation in annotations['annotations']:
        if annotation['category_id'] == id1:
            annotation['category_id'] = id2

        elif annotation['category_id'] == id2:
            annotation['category_id'] = id1


def insert_background_label(annotations):
    '''Insert background label at the 0th category ID. Pytorch requires this for training
    
    INPUT
        annotations : Dictionary containing image data, image annotations and list of all categories'''

    #If we have zeroth label, shift up all labels to make room
    if any(category['id'] == 0 for category in annotations['categories']):
        for category in annotations['categories']:
            category['id'] += 1

        for annotation in annotations['annotations']:
            annotation['category_id'] += 1

    annotations['categories'].append({'id': 0, 'name': 'Background'})


def relabel(target_categories, new_category, annotations_path):
    '''OLD FUNCTION. Relabels all annotations for target categories to a new category
    
    INPUT
        target_categories : List of target categories to change
        new_category : New category ID
        annotations_path : Path to annotations .json folder'''

    #Open annotations file  (Consider changing function to take annotations variable instead of the path. Open outside of function)
    with open(annotations_path, 'r') as file:
        annotations = json.load(file)

    #Order categories by ID
    annotations['categories'] = sorted(annotations['categories'], key=lambda x: x["id"])

    target_categories = [element.lower() for element in target_categories]   #Categories to relabel
    category_reference = {}      #For changing annotation IDs
    new_categories = []          #New categories list for annotations file

    found_first_target = False     #Checks if first category found
    decrease_number = 0            #For changing category IDs if not a target
    new_category_id = 0            #Assigned during loop to target category with minimum ID

    for category in annotations['categories']:

        if category['name'].lower() not in target_categories:
            category_reference[category['id']] = category['id'] - decrease_number   #decrease number ensures consecutivity
            new_categories.append(category)
        
        elif category['name'].lower() in target_categories:
            #First target, append new label to categories and use ID as ID for all targets
            if not found_first_target:
                category['name'] = new_category
                new_categories.append(category)
                new_category_id = category['id']

                found_first_target = True

            #Above first target, need to reduce all others by more
            else: decrease_number += 1

            category_reference[category['id']] = new_category_id

    #Change categories
    annotations['categories'] = new_categories

    #Change annotation category IDs
    for i, annotation in enumerate(annotations['annotations']):
        annotations['annotations'][i]['category_id'] = category_reference[annotation['category_id']]

    #Save changed annotations
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f)


def rescale(annotations, location, image_size):
    '''FUNCTION NOT FULLY TESTED. Resizes all images to given image size using a lanczos transformation.
    
    INPUT
        annotations : Dictionary containing image data, image annotations and list of all categories
        Location : location to save resized images
        image_size : Image size to rescale to. Given as a single integer'''

    #If needed, wipe old data
    folder_name = 'resized_images'
    path = os.path.join(location, folder_name)
    
    if os.path.isdir(path):
        shutil.rmtree(path)

    os.mkdir(path)

    print('Resizing Images...')

    for i, image_info in tqdm(enumerate(annotations['images'])):
        #Load image
        image_path = os.path.join(location, image_info['file_name'])
        image = PIL.Image.open(image_path)

        #Calculate scaling factor
        width, height = image.size
        scale_factor = image_size / max(width, height)
        width = int(width*scale_factor)
        height = int(height*scale_factor)

        #Rescale
        image = image.resize((width, height), resample=PIL.Image.LANCZOS)

        #Save rescaled image to folder
        head, tail = os.path.split(image_info['file_name'])
        new_path = os.path.join(location, folder_name, tail)
        image.save(new_path)

        #Adjust annotations
        image_info['file_name'] = os.path.join(folder_name, tail)
        image_info['width'] = width
        image_info['height'] = height

        #Adjust bbox size
        for annotation in annotations['annotations']:
            if annotation['image_id'] == image_info['id']:
                annotation['bbox'] = [val * scale_factor for val in annotation['bbox']]

                annotation['area'] = annotation['bbox'][-1] * annotation['bbox'][-2]  