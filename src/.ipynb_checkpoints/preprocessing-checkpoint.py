# -*- coding: utf-8 -*-
import os
import shutil
import PIL
import json

from tqdm import tqdm


def concatenate_datasets(location, folder_names):
    '''Combines annotations from several datasets, relabeling annotations, finding all unique classes and updating image paths'''

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


def swap_labels(annotations, id1, id2):
        
    annotations['categories'][id1]['name'], annotations['categories'][id2]['name'] = annotations['categories'][id2]['name'], annotations['categories'][id1]['name']

    for annotation in annotations['annotations']:
        if annotation['category_id'] == id1:
            annotation['category_id'] = id2

        elif annotation['category_id'] == id2:
            annotation['category_id'] = id1


def insert_background_label(annotations):
    '''Insert background label to 0th id'''

    #If we have zeroth label, shift up all labels to make room
    if any(category['id'] == 0 for category in annotations['categories']):
        for category in annotations['categories']:
            category['id'] += 1

        for annotation in annotations['annotations']:
            annotation['category_id'] += 1

    annotations['categories'].append({'id': 0, 'name': 'Background'})


def make_consecutive(annotations):
    '''Ensures all category IDs are in consecutive order'''

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


def rescale(annotations, location, image_size):

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
                    


def preprocess(location, folder_names, image_size=None):
    '''Preprocesses data, returning single .json file for datasets in folder_names at given location'''

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

    if image_size != None:
        rescale(annotations, location, image_size)

    new_annotations_path = os.path.join(os.getcwd(), "formatted_annotations.json")

    with open(new_annotations_path, 'w') as f:
        json.dump(annotations, f)

    return new_annotations_path