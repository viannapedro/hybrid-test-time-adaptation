from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import math
import random

from pprint import pprint

class ThresholdLayer(nn.Module):
    '''
    The ThresholdLayer is a PyTorch module that performs thresholding on a batch of 2D images passed as input. It has the following parameters:

    og_bn: a nn.BatchNorm2d instance, representing the original batch normalization layer.
    threshold: a threshold value used for thresholding.
    threshold_choice: a string that determines the function used to calculate the threshold. The available options are "Const", "Mean", and "Median".
    dist_choice: a string that determines the function used to calculate the distance between the original and adapted batch normalization layers. The available options are "Wasser", "Diff",         "DiffAbs", "DiffVar", and "DiffVarAbs".
    switch: a boolean flag indicating whether the threshold should be used for masking values above or below it.

    During initialization, the ThresholdLayer saves the state dict of the original batch normalization layer (og_bn) and creates a new batch normalization layer (adapt_bn) with the same number     of features and momentum 1. It also initializes the distance function and the threshold function based on the chosen options.

    During the forward pass, the original batch normalization layer is set to evaluation mode, and the adapted batch normalization layer is set to training mode. The input is passed through         both batch normalization layers, and the distance between the two is calculated using the chosen distance function. The output of the layer is a thresholded combination of the output of the     original and adapted batch normalization layers, where the combination is determined by the threshold and the switch flag.
    '''
    
    def __init__(self, og_bn, count, bn_layers, threshold=0.01, threshold_choice="Linear-Decay", dist_choice="Wasser", switch=False):
        super(ThresholdLayer, self).__init__()
        self.og_bn = nn.BatchNorm2d(og_bn.num_features)
        self.og_bn.load_state_dict(og_bn.state_dict())

        self.adapt_bn = nn.BatchNorm2d(og_bn.num_features, momentum=1)
        self.adapt_bn.load_state_dict(og_bn.state_dict())
        self.adapt_bn.reset_running_stats()

        self.threshold = threshold
        self.switch = switch

        diff =  lambda x, y: x.running_mean - y.running_mean
        diff_var =  lambda x, y: x.running_var - y.running_var

        diff_abs = lambda x, y: torch.pow(diff(x, y), 2)
        diff_var_abs = lambda x, y: torch.pow(diff_var(x, y), 2)

        wasser = lambda x, y: diff_abs(x, y) + x.running_var + y.running_var - 2*torch.sqrt(x.running_var)*torch.sqrt(y.running_var)

        
        dist_choices = {"Wasser": wasser, "Diff": diff, "DiffAbs": diff_abs,
                "DiffVar": diff_var, "DiffVarAbs": diff_var_abs}

        self.dist_choice = dist_choice
        self.dist_func = dist_choices[dist_choice]


        self.threshold_choice = threshold_choice
        threshold_choices = {"Const": lambda x: threshold, "Mean": lambda x:
                torch.mean(x), "Median": lambda x: torch.median(x), 
                "Linear-Decay": lambda x: 1}
        self.threshold_func = threshold_choices[threshold_choice]

        self.switch = switch
        self.bn_layers = bn_layers
        self.count = count

    def forward(self, x):
        self.og_bn.eval()
        self.adapt_bn.train()
        og_norm = self.og_bn(x)
        adapt_norm = self.adapt_bn(x)
        mask = self.dist_func(self.og_bn, self.adapt_bn)

        
        if self.threshold_choice == 'Linear-Decay':
            print('Layer #: %i out of %i' % (self.count+1, self.bn_layers))
            
            x_percent = []
            x_layers = np.arange(0, self.bn_layers)
            for num, layer in enumerate(x_layers):
                pct = 100 - num*(100/(self.bn_layers-1))
                x_percent.append(max(pct,0.))
            
            percentage = x_percent[self.count]/100

            sorted_mask, _ = torch.sort(mask)

            # We want to adapt 'percentage'-channels of a layer. 
            # Ex: First layer should be 100% adapted [n_keep = 0], last layer should be 0% adapted [n_keep = len(mask)].
            n_adapts = int(percentage*len(mask))
            n_keep = len(mask) - n_adapts
            print('Adapting %.1f%% of this layer (%i out of %i channels)' % (x_percent[self.count], n_adapts, n_adapts+n_keep))
            
            if n_adapts > 0:
                if n_adapts == len(mask):
                    adapt_list = [True] * len(mask)
                else:
                    # We want to adapt the channels with lowest wasserstein difference. 
                    threshold = sorted_mask[n_adapts]
                    adapt_list = [True if value < threshold else False for value in mask]
                    
                    if self.switch: #In case we want to adapt the top wasser. Switch by default is False.
                        threshold = sorted_mask[-n_adapts] 
                        adapt_list = [True if value >= threshold else False for value in mask]                    
                    
            else:
                adapt_list = [False] * len(mask)

            adapt_list = torch.tensor(adapt_list, device=torch.device('cuda:0'))
            mask_1 = (adapt_list.bool()).int().float()
            mask_0 = (~mask_1.bool()).int().float()
            adapt_norm = adapt_norm.detach().clone()
            og_norm = og_norm.detach().clone()
                          
        else:
            if self.switch:
                mask = mask > self.threshold_func(mask)
            else:
                mask = mask < self.threshold_func(mask)
            mask_1 = mask.int().float()
            mask_0 = (~mask).int().float()

        mask_0 = mask_0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        mask_1 = mask_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        output_mask = (mask_0 * og_norm) + (mask_1 * adapt_norm)

        return output_mask

def replace_layers(model, threshold, threshold_choice, dist_choice, switch, model_bn_layers, count=0):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            count = replace_layers(module, threshold, threshold_choice, dist_choice, switch, model_bn_layers, count)
            
        if isinstance(module, nn.BatchNorm2d):
            ## simple module
            new_bn = ThresholdLayer(module, count, model_bn_layers, threshold, threshold_choice, dist_choice, switch)
            setattr(model, n, new_bn)
            count += 1
    return count
 
class Threshold(nn.Module):
    """Norm adapts a model by estimating feature statistics during testing.

    Once equipped with Norm, the model normalizes its features during testing
    with batch-wise statistics, just like batch norm does during training.
    """

    def __init__(self, model, threshold, threshold_choice, dist_choice, switch):
        super(Threshold, self).__init__()
        self.model = configure_model(model, threshold, threshold_choice, dist_choice, switch)

        self.model_state = deepcopy(self.model.state_dict())

    def forward(self, x):
        return self.model(x)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)


def configure_model(model, threshold, threshold_choice, dist_choice, switch):
    """Configure model for adaptation by test-time normalization."""
    model.cpu()
    model_bn_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            model_bn_layers += 1

    replace_layers(model, threshold, threshold_choice, dist_choice, switch, model_bn_layers)
          
    model = nn.DataParallel(model)
    model.cuda()

    return model
