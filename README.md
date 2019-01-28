# atlas-classifier
Solution for [Human Protein Atlas Image Classification - 
Classify subcellular protein patterns in human cells](https://www.kaggle.com/c/human-protein-atlas-image-classification)  
Since the competition is closed, I'll keep this solution trivial - just apply multi-label focal loss to ResNet and DenseNet.  
For classes with very few samples, I think Siamese network might be better than multi-label classifier.  
My F1 (micro) score is 0.62 on val set (seperately selected from train set with multi-label stratification).  
I guess train and test sets have different distributions, so ensembling and hyper-parameter finetuning should be helpful.  

## Features
I'll try the following features:  
1. multi-label focal loss with class weights
2. ResNet and DenseNet
3. multi-label stratification on val set selection
4. different metrics and score thresholds
5. ensembling
6. N-fold cross validation

## Data
For resolution, 512 * 512 is used.  
For augmentation:  
1. randomly flip vertically and horizontally
2. randomly rotation within +- 15 degrees
3. randomly crop within 90%-100% scale and 0.9-1.1 ratio
