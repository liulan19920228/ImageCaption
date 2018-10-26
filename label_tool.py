#!/usr/bin/env python
import numpy as np
import cv2
from sklearn import svm
import os, sys
import pickle
import cPickle
import matplotlib.pyplot as plt
import random
import json
import shutil

if __name__ == "__main__":
    #dataset_bwv = {}
    #dataset_bwv['dataset'] = 'bwv'
    #dataset_bwv['images'] = []

    # Read already-labeled progress from file.
    with open('../data/dataset_bwv.json','r') as f:
        dataset_bwv = json.load(f)

    stop_point = dataset_bwv.get('stop_point', 0)
    # import pdb; pdb.set_trace()

    bwv_img_source_dir = '/home/bertozzilab/Desktop/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages_enhanced_v2'
    bwv_img_dest_dir = '/home/bertozzilab/Desktop/ImageCaptioning_pytorch_bwv/data/image_bwv'
    train_n = len(os.listdir(os.path.join(bwv_img_dest_dir,"train2014")))
    val_n = len(os.listdir(os.path.join(bwv_img_dest_dir,"val2014")))
    labeled = train_n+val_n
    all_imgs = os.listdir(bwv_img_source_dir)

    
    #random.seed(4) # Everytime we do the same shuffle
    #random.shuffle(all_imgs)

    for i in range(stop_point, len(all_imgs)):
        print("Current progress: {0} train labeled, {1} val labeled".format(train_n, val_n))
        im = cv2.imread(os.path.join(bwv_img_source_dir, all_imgs[i]))
        im = im[:,:,[2,1,0]]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')   
        plt.ion() 
        plt.show()
        # Use this image or not
        while True:
            try:
                DoThis = raw_input("Do we want to use this image? [%d/%d], y/n : " % (i, len(all_imgs)))
                if DoThis not in ['y','n']:
                    raise ValueError('Error! Please enter y or n.')
                break;
            except ValueError:
                print("Try again. ")
        if DoThis == 'n':
            plt.close('all')
            continue

        while True:
            try:
                split = input("Train(0) or Validation(1)? : ")
                if split not in [0,1]:
                    raise ValueError('Error! Please enter 0 or 1.')
                break;
            except (SyntaxError, ValueError) as e:  # need to catch syntax error too.
                print("Try again. ")
        caption = raw_input("Enter the caption for this image: ")

        dataset_bwv['images'].append({})
        dataset_bwv['images'][-1]['imgid'] = train_n + val_n
        dataset_bwv['images'][-1]['sentids'] = [train_n + val_n]
        if split == 0:
            dataset_bwv['images'][-1]['split'] = 'train'
            dataset_bwv['images'][-1]['filepath'] = 'train2014'
            train_n = train_n + 1
            shutil.copy(os.path.join(bwv_img_source_dir,all_imgs[i]), os.path.join(bwv_img_dest_dir,"train2014"))
        elif split == 1:
            dataset_bwv['images'][-1]['split'] = 'val'
            dataset_bwv['images'][-1]['filepath'] = 'val2014'
            val_n = val_n + 1
            shutil.copy(os.path.join(bwv_img_source_dir,all_imgs[i]), os.path.join(bwv_img_dest_dir,"val2014"))
        else:
            print("Error!")
        dataset_bwv['images'][-1]['filename'] = all_imgs[i]
        dataset_bwv['images'][-1]['bwvid'] = i

        dataset_bwv['images'][-1]['sentences'] = [{}]
        dataset_bwv['images'][-1]['sentences'][0]['raw'] = caption
        #check if it ends with a period.
        if caption[-1] is not '.':
            caption = caption + "."
        dataset_bwv['images'][-1]['sentences'][0]['tokens'] = caption.split()
        dataset_bwv['images'][-1]['sentences'][0]['tokens'][-1] = dataset_bwv['images'][-1]['sentences'][0]['tokens'][-1][:-1]
        dataset_bwv['images'][-1]['sentences'][0]['imgid'] = dataset_bwv['images'][-1]['imgid'] 
        dataset_bwv['images'][-1]['sentences'][0]['sentid'] = dataset_bwv['images'][-1]['imgid']

        plt.close('all')
        dataset_bwv['stop_point'] = i+1;
        with open('../data/dataset_bwv.json','w') as f:  # save for every image. 
            json.dump(dataset_bwv,f)
        GoOn = raw_input("Continue to next image? y/n : (Warning! Please type 'n' to not to lose the progress!!)")
        if GoOn == 'y':
            continue
        elif GoOn == 'n':
            break

    with open('../data/dataset_bwv.json','w') as f:
        json.dump(dataset_bwv,f)




        




