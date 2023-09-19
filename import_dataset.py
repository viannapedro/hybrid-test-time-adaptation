import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import torchvision
import os
from PIL import Image


def import_cifar(n_classes, corruption, severity, batch_size=500):    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    selected_classes = random.sample(classes, n_classes)
    
    if severity > 0:
        print('Classes are:', classes)
        print('Using %i classes, %s with severity %i' % (n_classes, corruption, severity))
        corruption_str = corruption + '_' + str(severity)
        corrup_file = '/data/CIFAR-10-C/' + corruption + '.npy'
        data = np.load(corrup_file)
        labels = np.load('/data/CIFAR-10-C/labels.npy')
        images = data[(severity-1) * 10000: ((severity-1) * 10000) + 10000,...]
        images = images.transpose(0,3,1,2)
        labels = labels[(severity-1) * 10000: ((severity-1) * 10000) + 10000]

        #Get the indices of the selected classes
        samples_per_class = int(batch_size/n_classes)
        print('Selected classes:', selected_classes)
        print(samples_per_class, 'samples per class')
        indexes = [classes.index(elem) for elem in selected_classes]

        # Initialize the arrays for the selected samples
        X_selected = np.zeros((samples_per_class * n_classes, images.shape[1], images.shape[2], images.shape[3]), dtype=np.uint8)
        Y_selected = np.zeros((samples_per_class * n_classes,), dtype=np.uint8)

        # Loop through the selected classes and get the samples
        index = 0
        for i, c in enumerate(indexes):
            class_indices = np.where(labels == c)[0]
            selected_indices = random.sample(list(class_indices), samples_per_class)
            X_selected[index:index+samples_per_class] = images[selected_indices]
            Y_selected[index:index+samples_per_class] = c
            index += samples_per_class

        class CustomTensorDataset(Dataset):
            """TensorDataset with support of transforms.
            """
            def __init__(self, tensors, transform=None):
                assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
                self.tensors = tensors
                self.transform = transform

            def __getitem__(self, index):
                x = self.tensors[0][index]

                if self.transform:
                    x = self.transform(x)

                y = self.tensors[1][index]

                return x, y

            def __len__(self):
                return self.tensors[0].size(0)

        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        X_train = torch.tensor(X_selected)
        Y_train = torch.tensor(Y_selected)
        corrupt_dataset_trans = CustomTensorDataset(tensors=(X_train,Y_train), transform=transform_test)
        dataloader = torch.utils.data.DataLoader(corrupt_dataset_trans, batch_size=batch_size, shuffle=True, num_workers=2)
    
    else:
        print("Using uncorrupted images.")
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        cifarset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        samples = []
        if n_classes == 1:
            print(selected_classes)
            print(classes)
            class_index = classes.index(selected_classes[0])
            testsubset = torch.utils.data.Subset(cifarset, [idx for idx in range(len(cifarset)) if cifarset[idx][1] == class_index])
        else:
            rng = np.random.default_rng()
            class_index = rng.choice(len(classes), size=n_classes, replace=False)
            testsubset = torch.utils.data.Subset(cifarset, [idx for idx in range(len(cifarset)) if cifarset[idx][1] in class_index])
        dataloader = torch.utils.data.DataLoader(testsubset, batch_size=batch_size, shuffle=True, num_workers=2)

    return dataloader
        
def import_imagenet(n_classes, corruption, severity, transform, batch_size=500):
    classes = range(0,1000)
    selected_classes = random.sample(classes, n_classes)
    
    if severity > 0:
        print('Using %i classes, %s with severity %i' % (n_classes, corruption, severity))
        dataset = torchvision.datasets.ImageFolder("/data/Imagenet-C/{}/{}/".format(corruption, severity), transform=transform)

    else:
        print('Using %i classes, without corruptions' % (n_classes))
        path = "/data/imagenet_torchvision/val/"

        dataset = torchvision.datasets.ImageFolder(path, transform=transform)
        print('Loaded dataset without corruptions.')

    dataset.targets = np.array(dataset.targets)
    indices = np.where(np.in1d(dataset.targets, selected_classes))[0]
    if len(indices)>batch_size:
        indices = random.sample(list(indices), batch_size)

    dataset.targets = dataset.targets[indices].tolist()
    dataset.samples = [dataset.samples[index] for index in indices]
    print('Loaded dataset with %i images.' % len(dataset))

    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)
    return dataloader

def replace_labels(dataset, task_n):
    replace_dict = {0:{0.0:0, 1.0:1, 2.0:1, 3.0:1}, 1:{0.0:0, 1.0:0, 2.0:1, 3.0:1}, 2:{0.0:0, 1.0:0, 2.0:0, 3.0:1}}
    copyset = dataset.copy() 
    copyset['steatosis_binary'] = dataset['steatosis'].replace(replace_dict[task_n])
    return copyset

def counts_props(dataset_file):
    counts = pd.value_counts(dataset_file['steatosis_binary'], normalize=False, dropna=False)
    props = pd.value_counts(dataset_file['steatosis_binary'], normalize=True, dropna=False)
    return counts, props

def import_US(target_dist, sample_seed = 42, task_n=0, batch_size=500):   
    '''
        Task_n defines the classification problem.
        0 for diagnosis (Steatosis 0 vs Steatosis >=1), 1 for <= 1 vs >= 2, 2 for <=2 vs 3. 
    '''
    #Target path
    data_path_t = '/data/byra_dataset/byra_dataset.csv'
    images_path_t = '/data/byra_dataset/'

    t_dataset = pd.read_csv(data_path_t)
    target_dataset = t_dataset.dropna(subset=['filename'])
    target_dataset = target_dataset.rename(columns={'Steatosis_grade':'steatosis'})

    val_set = target_dataset
    test_set = val_set.copy()

    max_batch = batch_size

    test_0 = test_set.loc[test_set['steatosis']==0]
    test_1 = test_set.loc[test_set['steatosis']==1]
    test_2 = test_set.loc[test_set['steatosis']==2]
    test_3 = test_set.loc[test_set['steatosis']==3]
    test_bin = test_set.loc[test_set['steatosis']!=0]

    lens = [len(sets) for sets in [test_0,test_1,test_2,test_3]]
    source_props = [0.163, 0.4083, 0.21067, 0.21799] #this is hard coded. obtained from the training set on private data
    target_props = pd.value_counts(test_set['steatosis'],normalize=True)
    if min(lens) <= max_batch: max_batch = min(lens)

    test_0_s = test_0.sample(max_batch, random_state = sample_seed)
    test_1_s = test_1.sample(max_batch, random_state = sample_seed)
    test_2_s = test_2.sample(max_batch, random_state = sample_seed)
    test_3_s = test_3.sample(max_batch, random_state = sample_seed)
    test_b_s = test_bin.sample(max_batch,random_state = sample_seed)

    test_0_o = test_0.sample(round(max_batch*target_props[0]), random_state = sample_seed)
    test_1_o = test_1.sample(round(max_batch*target_props[1]), random_state = sample_seed)
    test_2_o = test_2.sample(round(max_batch*target_props[2]), random_state = sample_seed)
    test_3_o = test_3.sample(round(max_batch*target_props[3]), random_state = sample_seed)
    test_b_o = test_bin.sample(round(max_batch-len(test_0_o)), random_state = sample_seed)

    test_0_S = test_0.sample(round(max_batch*source_props[0]), random_state = sample_seed)
    test_1_S = test_1.sample(round(max_batch*source_props[1]), random_state = sample_seed)
    test_2_S = test_2.sample(round(max_batch*source_props[2]), random_state = sample_seed)
    test_3_S = test_3.sample(round(max_batch*source_props[3]), random_state = sample_seed)
    test_b_S = test_bin.sample(round(max_batch-len(test_0_S)), random_state = sample_seed)

    test_og = pd.concat([test_0_o, test_b_o])
    test_source = pd.concat([test_0_S, test_b_S]) 
    test_eq = pd.concat([test_0_s[:round(max_batch/2)], test_b_s[:round(max_batch/2)]])
    test_0  = test_0_s
    test_1 = test_b_s

    val = replace_labels(val_set, task_n)
    test_og = replace_labels(test_og, task_n)
    test_source = replace_labels(test_source, task_n)
    test_eq = replace_labels(test_eq, task_n)
    test_0 = replace_labels(test_0, task_n)
    test_1 = replace_labels(test_1, task_n)
    test_2 = replace_labels(test_2, task_n)
    test_3 = replace_labels(test_3, task_n)

    print('Finished preparing the sets.')

    print('Importing the images.')

    class CustomImageDataset(Dataset):
        def __init__(self, subset, imgs_path, x_transform=None):
            self.subset = subset
            self.imgs_path = imgs_path
            self.x_transform = x_transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            image_file = os.path.join(self.imgs_path, self.subset.iloc[idx]['filename'])
            image = Image.open(image_file).convert('RGB')

            if self.x_transform:
                image = self.x_transform(image)          
            label = self.subset.iloc[idx]['steatosis_binary']

            return image, label

    test_counts, test_prop = counts_props(test_og)

    #Train dataset mean and var
    train_mean = 0.0846  #train_data[0].mean() on the training set
    train_std = 0.1395   #train_data[0].std() on the training set

    resize_shape = (128, 128)

    x_trans = transforms.Compose(
        [transforms.Resize(resize_shape), 
         transforms.ToTensor(), 
         transforms.Normalize(torch.tensor([train_mean]*3), torch.tensor([train_std]*3))]
        )

    val_dataset = CustomImageDataset(val, images_path_t, x_trans)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    test_dataset = CustomImageDataset(test_og, images_path_t, x_trans)
    test_dataset_source = CustomImageDataset(test_source, images_path_t, x_trans)
    test_dataset_eq = CustomImageDataset(test_eq, images_path_t, x_trans)
    test_dataset_0 = CustomImageDataset(test_0, images_path_t, x_trans)
    test_dataset_1 = CustomImageDataset(test_1, images_path_t, x_trans)

    testloader = DataLoader(test_dataset, batch_size=len(test_og), shuffle=True, num_workers=4)
    testloader_source = DataLoader(test_dataset_source, batch_size=len(test_source), shuffle=True, num_workers=4)
    testloader_eq = DataLoader(test_dataset_eq, batch_size=len(test_eq), shuffle=True, num_workers=4)
    testloader_zeroes = DataLoader(test_dataset_0, batch_size=len(test_0), shuffle=True, num_workers=4)
    testloader_ones = DataLoader(test_dataset_1, batch_size=len(test_1), shuffle=True, num_workers=4)

    classes = [np.unique(d['steatosis_binary']) for d in [test_og, test_source, test_eq, test_0, test_1]]

    print('Finished preparing the dataloaders.')
    
    if target_dist == 'original':
        dataloader = testloader
        print('Using original target distribution.')
    
    elif target_dist == 'source':
        dataloader = testloader_source
        print('Using source distribution.')

    elif target_dist == 'balanced':
        dataloader = testloader_eq
        print('Using 50:50 distribution.')

    elif target_dist == 'zeroes':
        dataloader = testloader_zeroes
        print('Using only zeroes in the target.')

    elif target_dist == 'ones':
        dataloader = testloader_ones
        print('Using only ones in the target.')
    
    return dataloader