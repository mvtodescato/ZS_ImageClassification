# ZS_ImageClassification

## From paper "No Labels Needed: Zero-Shot Image Classification with Collaborative Self-Learning"

# Code files
-ic_noloop_paper.py
	
This is the code of our approach to perform zero-shot image classification without the self-learning loop ("Seed" on the paper). Use -h tag to see the help of the code. Remember to extract features of the dataset first using simple_features.py. 

-ic_wloop_paper.py

This is the code of our approach to perform zero-shot image classification with the complete approach ("Complete" on the paper). Use -h tag to see the help of the code. Remember to extract features of the dataset first using simple_features.py. 

-simple_features.py
	
In this code we extract features from the datasets using pre-trained models (vit_g14 by default).

-data_loaders.py

Code to load the datasets. 

#Obs

To use CLIP you need to download and install it from: https://github.com/openai/CLIP.

Cifar100 and Cifar10 are the easier datasets to test.
