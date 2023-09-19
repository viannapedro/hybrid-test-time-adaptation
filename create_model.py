import torch
import torchvision.models as models
import torch.nn as nn
from cheap_resnet import ResNet, Bottleneck
import thresholding_combined as thc
import copy

def ResNet26():
    return ResNet(Bottleneck, [2, 2, 2, 2], num_classes=10)

def batch_norm_stats(model, print_stats=True):
    bn_layers = 0
    bn_stats = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers += 1
            layer_mean = module.running_mean
            layer_var = module.running_var
            layer_stats = [[layer_mean.min().item(), layer_mean.mean().item(), layer_mean.max().item()],
                           [layer_var.min().item(), layer_var.mean().item(), layer_var.max().item()]]
            bn_stats.append(layer_stats)
    if print_stats:
        print('There are %.d batch_norm layers.' % bn_layers)
        print('Stats for those layers: \nrunning_mean and running_var [min, mean, max]')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in bn_stats]))
    return bn_stats

def list_modules(model):
    module_list = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            module_list.append(copy.deepcopy(m))
    return nn.ModuleList(module_list)

def import_arch(model_arch, weights_path, device, ultrasound = False):
    print('==> Building model..')
    if model_arch == 'Resnet-26':
        base_model = ResNet26()
        trained_weights = torch.load(weights_path)
        sd = trained_weights['net']
        base_model.load_state_dict(sd)

    if model_arch == "Resnet-18":
        base_model = models.resnet18(weights="IMAGENET1K_V1").to(device)

    print('Weights loaded.')

    net = base_model.to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True

    if ultrasound: 
        base_model.fc = nn.Linear(512,2)
        net.load_state_dict(torch.load(weights_path))
    
    net.eval()
    print("Model created: %s." % model_arch)

    trained_bn_stats = batch_norm_stats(net, print_stats=False)
    modules = list_modules(net)
    
    return net, trained_bn_stats, modules

def model_adapt(net, threshold_choice = 'Linear-Decay', dist_choice = 'Wasser', switch = False, threshold = 0):
    """
        Default values set for the proposed Hybrid-TTN method at WACV 2024 submission.
        For Wasserstein distance, we don't need threshold.
    """
      
    print('We are adapting the model with:', threshold_choice)
    copynet = copy.deepcopy(net)

    norm_net = thc.Threshold(copynet, threshold, threshold_choice, dist_choice, switch)
    if switch: print('Adapting highest values (inverted logic).')
    print('Model adapted for threshold.')

    adapted_model_stats = batch_norm_stats(norm_net, print_stats=False)
    adap_modules = list_modules(norm_net)
    
    return norm_net, adapted_model_stats, adap_modules

    