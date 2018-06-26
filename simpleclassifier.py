import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.optim import lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import copy
import time
from tqdm import tqdm

import io

import matplotlib.pyplot as plt
import matplotlib.image as mpimgs
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def save_model(model, model_path):
    print("Saving the model.")
    torch.save(model.cpu().state_dict(), model_path)

def load(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def get_transforms(tvfolders):
    image_transforms ={x: transforms.Compose([
            transforms.Resize([64,64]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])for x in tvfolders}


def get_data_sets(data_folder, tvfolders, image_transforms):
    data_set = {x: torchvision.datasets.ImageFolder(os.path.join(data_folder,x), image_transforms[x])for x in ['Train','Val']}
    return data_set

def get_data_loaders(data_sets, data_folder, tvfolders, batch_size=4, num_workers =4):
    dataloaders = {x: torch.utils.data.DataLoader(data_sets[x], batch_size = 4, shuffle = True, num_workers = 4) for x in ['Train', 'Val']}
    return dataloaders

def get_data_set_length(data_sets):
    dataset_size = {x: len(data_sets[x])for x in ['Train', 'Val']}
    return dataset_size

def get_class_names(data_sets):
    class_names = data_sets['Train'].classes
    return class_names


def train(model, dataloaders, dataset_size, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    #keeping time for training
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                inputs, labels = data

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            #if we get a better validation phase then we should update the model
            if phase == 'Val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print('Updated model')
                best_model = copy.deepcopy(model.state_dict())

            print()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            # deep copy the model
            if phase == 'Val' and epoch_acc.double() > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #load best model weights
    model.load_state_dict(best_model_wts)
    return model

def predict(model, image):
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return class_names[predicted[0]]

def imshow(imgs , title = None):
    imgs = imgs.numpy().transpose(1,2,0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array ([0.229, .224, 0.225])
    imgs = imgs * std + mean
    imgs = np.clip(imgs, 0 , 1)
    plt.imshow(imgs)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def freeze_resnet_layers(model, num_layers):
    child_counter = 0
    for child in model.children():
        if child_counter > num_layers:
            for param in child.parameters():
                param.requires_grad = False
        elif child_counter == num_layers:
            children_of_child_counter = 0
            for children_of_child in child.children():
                if children_of_child_counter < 1:
                    for param in children_of_child.parameters():
                        param.requires_grad = False
        child_counter += 1

def predict_batch(model, dataloaders):
    dataiter = iter(dataloaders['Val'])
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images), 'Batch')
    plt.show()
    print('Actl: ', ' '.join('%5s' % class_names[labels[j]] for j in range(4)))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('Pred: ', ' '.join('%5s' % class_names[predicted[j]]
                              for j in range(4)))



model = models.resnet18(pretrained=False)
data_folder = 'Marine'
trainval = ['Train', 'Val']
image_transforms= {'Train': transforms.Compose([
            transforms.Resize([64,64]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Val': transforms.Compose([
            transforms.Resize([64,64]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
dataset = get_data_sets(data_folder, trainval, image_transforms)
dataset_sizes = get_data_set_length(dataset)
dataloaders = get_data_loaders(dataset , data_folder, trainval)
class_names = dataset['Train'].classes
device = "cuda" if torch.cuda.is_available() else "cpu"

num_final_in = model.fc.in_features

# The final layer of the model is model.fc so we can basically just overwrite it
#to have the output = number of classes we need. Say, 300 classes.
NUM_CLASSES = len(class_names)
freeze_resnet_layers(model,num_final_in-4)
model.avgpool = nn.AdaptiveAvgPool2d(1)
model = model.to(device)
model.fc = nn.Linear(num_final_in, NUM_CLASSES)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.005)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train(model, dataloaders,dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs = 15 )


path = 'model.pt'
torch.save(model.cpu().state_dict(), path)
