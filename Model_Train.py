import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import csv
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data(data_dir):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                 shuffle=True, num_workers=10)
                  for x in ['train', 'val']}
    testloader=torch.utils.data.DataLoader(image_datasets['test'],batch_size=1,shuffle=False)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    return (dataloaders,testloader,class_names,num_classes,dataset_sizes)


def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

def plot(val_loss,train_loss,typ):
    plt.title("{} after epoch: {}".format(typ,len(train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel(typ)
    plt.plot(list(range(len(train_loss))),train_loss,color="r",label="Train "+typ)
    plt.plot(list(range(len(val_loss))),val_loss,color="b",label="Validation "+typ)
    plt.legend()
    plt.savefig(os.path.join(data_dir,typ+".png"))
    plt.close()

# The function used for training and validating the CNN model
def train_model(model,num_epochs=25,model_name = "kaggle"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=.1)
    val_loss_gph=[]
    train_loss_gph=[]
    val_acc_gph=[]
    train_acc_gph=[]
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs= model(inputs)
                    _, preds = torch.max(outputs, 1) #was (outputs,1) for non-inception and (outputs.data,1) for inception
                    #_, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                train_loss_gph.append(epoch_loss)
                train_acc_gph.append(epoch_acc.cpu())
            if phase == 'val':
                val_loss_gph.append(epoch_loss)
                val_acc_gph.append(epoch_acc.cpu())
            plot(val_loss_gph,train_loss_gph, model_name+"_Loss")
            plot(val_acc_gph,train_acc_gph, model_name+"_Accuracy")
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, data_dir+"/"+model_name+".h5")
                print('==>Model Saved')

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def VGG11(class_names):
    num_classes=len(class_names)
    VGG11 = models.vgg11(pretrained = True)
    num_ftrs= VGG11.classifier[0].in_features
    VGG11.classifier=nn.Linear(num_ftrs,num_classes)
    VGG11=VGG11.to(device)
    return VGG11

def Googlenet(class_names):
    num_classes=len(class_names)
    Googlenet=models.googlenet(pretrained=True)
    num_ftrs= Googlenet.fc.in_features
    Googlenet.fc=nn.Linear(num_ftrs,num_classes)
    Googlenet = Googlenet.to(device)
    return Googlenet

def MobileNet(class_names):
    num_classes=len(class_names)
    Mobile = models.mobilenet_v3_small(pretrained = True)
    num_ftrs = Mobile.classifier[3].in_features
    Mobile.classifier[3]=nn.Linear(num_ftrs,num_classes)
    return Mobile