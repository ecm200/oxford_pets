#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division

import argparse

# Local modules
from cub_tools.trainer import Ignite_Trainer
from cub_tools.transforms import makeDefaultTransforms

import sys
sys.path.insert(1, '/home/edmorris/projects/image_classification/oxford_pets/tools/')
from data import create_dataloaders

parser = argparse.ArgumentParser(description='PyTorch Image Classification Trainer - Ed Morris (c) 2021')
parser.add_argument('--config', metavar="FILE", help='Path and name of configuration file for training. Should be a .yaml file.', required=False, default='/home/edmorris/projects/image_classification/oxford_pets/scripts/configs/resnet50_config.yaml')
parser.print_help()
args = parser.parse_args()
#config = 'configs/googlenet_config.yaml'

# Data transformers
data_transforms = makeDefaultTransforms()

## Get the data loaders here and pass them explicitely to the Trainer object
train_loader, val_loader = create_dataloaders(
    data_transforms=data_transforms, 
    data_dir="data/images", 
    batch_size=16, 
    num_workers=4, 
    train_file='data/annotations/trainval.txt', 
    test_file='data/annotations/test.txt', 
    shuffle={'train' : True, 'test' : False}, 
    test_batch_size=2)

trainer = Ignite_Trainer(
    config=args.config, 
    data_transforms=data_transforms,
    train_loader=train_loader,
    val_loader=val_loader)

# Setup the data transformers
#print('[INFO] Creating data transforms...')
#trainer.create_datatransforms()

# Setup the dataloaders
#print('[INFO] Creating data loaders...')
#trainer.create_dataloaders()

# Setup the model
print('[INFO] Creating the model...')
trainer.create_model()

# Setup the optimizer
print('[INFO] Creating optimizer...')
trainer.create_optimizer()

# Setup the scheduler
trainer.create_scheduler()

# Train the model
trainer.run()

## Save the best model
#trainer.save_best_model()