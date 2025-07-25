# Hybrid Test-time Batch Normalization

_Classification under domain shift._  
_(Collaboration between: LBUM/CRCHUM - Université de Montréal and Mila - Quebec AI Institute)_  

**Note  - This code is the basis for the following publications:**  

- Vianna P, Chaudhary M, Tang A, Cloutier G, Wolf G, Eickenberg M, Belilovsky E. Channel selection for test-time adaptation under distribution shift. In **NeurIPS 2023** _Workshop on Distribution Shifts (DistShift)_: New Frontiers with Foundation Models 2023 Dec 7. Available at: https://openreview.net/pdf?id=BTOBu7y2ZD  
- Vianna P, Chaudhary M, Mehrbod P, Tang A, Cloutier G, Wolf G, Eickenberg M, Belilovsky E. Channel-selective normalization for label-shift robust test-time adaptation. 2024 Feb 7. Proceedings of **The 3rd Conference on Lifelong Learning Agents (CoLLAs 2024)**, in Proceedings of Machine Learning Research 274:514-533. Available at https://proceedings.mlr.press/v274/vianna25a.html, also on arXiv:2402.04958, https://arxiv.org/abs/2402.04958  
- Vianna P, Mehrbod P, Chaudhary M, Eickenberg M, Wolf G, Belilovsky E, Tang A, Cloutier G. Unsupervised Test-Time Adaptation for Hepatic Steatosis Grading Using Ultrasound B-Mode Images. **IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control**, vol. 72, no. 5, pp. 601-611, May 2025, doi: 10.1109/TUFFC.2025.3555180. Available at: https://ieeexplore.ieee.org/document/10942471    



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

## References (BibTeX)
@article{vianna2025unsupervised,  
  title={Unsupervised Test-Time Adaptation for Hepatic Steatosis Grading Using Ultrasound B-Mode Images},  
  author={Vianna, Pedro and Mehrbod, Paria and Chaudhary, Muawiz and Eickenberg, Michael and Wolf, Guy and Belilovsky, Eugene and Tang, An and Cloutier, Guy},  
  journal={IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control},  
  year={2025},  
  publisher={IEEE}  
}

@inproceedings{vianna2025channel,
  title={Channel-Selective Normalization for Label-Shift Robust Test-Time Adaptation},  
  author={Vianna, Pedro and Chaudhary, Muawiz Sajjad and Mehrbod, Paria and Tang, An and Cloutier, Guy and Wolf, Guy and Eickenberg, Michael and Belilovsky, Eugene},  
  booktitle={Conference on Lifelong Learning Agents},  
  pages={514--533},  
  year={2025},  
  organization={PMLR}  
}  

@inproceedings{vianna2023channel,   
title={Channel Selection for Test-Time Adaptation Under Distribution Shift},  
author={Pedro Vianna and Muawiz Chaudhary and An Tang and Guy Cloutier and Guy Wolf and Michael Eickenberg and Eugene Belilovsky},  
booktitle={NeurIPS 2023 Workshop on Distribution Shifts: New Frontiers with Foundation Models},  
year={2023},  
url={ https://openreview.net/forum?id=BTOBu7y2ZD }  
}
