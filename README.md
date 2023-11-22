# Hybrid Test-time Batch Normalization

_Classification under domain shift._  
_(Collaboration between: LBUM/CRCHUM - Université de Montréal and Mila - Quebec AI Institute)_  

This code is part of a **NeurIPS 2023** _Workshop DistShift_ submission.  

**Note  - Preliminary work**  

## Introduction
In deep learning, Batch Normalization is a commonly used technique to stabilize and accelerate training. However, in scenarios where the training and test data distributions differ significantly, traditional BN layers may not perform optimally. TTN addresses this issue by adapting BN layers during inference based on the distance between original BN statistics and those computed on target data.

## Usage
**Note  - Data not available in this repository. Code is presented for demonstration purposes.**  
To run the code, one needs the following datasets:  
- CIFAR-10 and CIFAR-10-Corruptions  
- ImageNet-1K and ImageNet-1K-Corruptions  
- Byra et al. publicly available dataset: 10.5281/zenodo.1009145  

  
Note that the trained weights and source data for the Ultrasound cannot be distributed.  
Requirements.txt created automatically with pipreqs package.

## Contact information
For any questions or comments, please contact the project authors at:

Pedro Vianna: **pedro.vianna@umontreal.ca**  

## Reference
@inproceedings{vianna2023channel,  
title={Channel Selection for Test-Time Adaptation Under Distribution Shift},  
author={Pedro Vianna and Muawiz Chaudhary and An Tang and Guy Cloutier and Guy Wolf and Michael Eickenberg and Eugene Belilovsky},  
booktitle={NeurIPS 2023 Workshop on Distribution Shifts: New Frontiers with Foundation Models},  
year={2023},  
url={ https://openreview.net/forum?id=BTOBu7y2ZD }  
}
