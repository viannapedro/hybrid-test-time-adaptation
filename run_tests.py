import torch
import copy
import torch.nn as nn


def test(net, dataloader, device):
    net.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx ==0:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100.*correct/total
                
    print('Simple Test accuracy: %.2f%%' % acc)
    
def TTN(net, dataloader, device):
    bn_layers = 0
    for module_name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers += 1

    adp_model = copy.deepcopy(net)
    adp_model.to(device)

    for batch_layer in range(0,bn_layers+1):
        adp_model.eval()
        bn_count = 0
        correct = 0
        total = 0
        
        for module_name, module in adp_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_count += 1
                if batch_layer >= bn_count:
                    module.reset_running_stats()
                    module.train()
                    module.momentum = 1.0
                    
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx == 0:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = adp_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100.*correct/total

                if batch_layer == 0:  print('Simple Test accuracy: %.2f%%' % acc)
                if batch_layer == bn_layers: print("Adapted TTN Test Accuracy: %.2f%% (layer %i) \n -------------------------" % (acc, batch_layer))
            
def hybridTTN(net, normnet, dataloader, device):
    correct = 0
    total = 0
    adp_correct = 0
    adp_total = 0
    
    net.to(device)
    normnet.to(device)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx == 0:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                te_acc = 100.*correct/total

                adp_outputs = normnet(inputs)
                _, adp_predicted = adp_outputs.max(1)
                adp_total += targets.size(0)
                adp_correct += adp_predicted.eq(targets).sum().item()
                adp_acc = 100.*adp_correct/adp_total

        print('Simple Test accuracy: %.2f%%' % (te_acc))
        print('Hybrid-TTN Adapted accuracy: %.2f%%' % (adp_acc))
        print('Delta accuracy:  %.2f (%.2f%%)' % (adp_acc - te_acc, (adp_acc - te_acc)*100/te_acc))