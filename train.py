import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from nets import *
import time, os, copy, argparse
import multiprocessing
from torchsummary import summary
#from matplotlib import pyplot as plt

# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, help="Training mode: finetune/transfer/scratch")
args= vars(ap.parse_args())

# Set training mode
train_mode=args["mode"]

# github download
# https://github.com/anilsathyan7/pytorch-image-classification/tree/master

# Set the train and validation directory paths
train_directory = '/home/rob/Pictures/Tassles/train/'
valid_directory = '/home/rob/Pictures/Tassles/val/'
# Set the model save path
PATH="/home/rob/Pictures/Tassles/models/tassle.pth"

# Batch size
bs = 32
vbs = 10 # make batch size for val smaller as not many images!
# Number of epochs
num_epochs = 10
# Number of classes
num_classes = 4
# Number of workers
num_cpu = multiprocessing.cpu_count()

# Applying transforms to the data
# could apply our own transforms???
image_transforms = { 
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        #transforms.RandomRotation(degrees=45),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.RandomAffine(degrees=(0, 180), translate=(0.0, 0.1), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        #transforms.RandomAdjustSharpness(2.0, p=0.5),
        #transforms.CenterCrop(size=224),
        #transforms.Resize((224,224)),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406],
        #                     [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        #transforms.Resize(size=256),
        #transforms.CenterCrop(size=224),
        #transforms.Resize((224,224)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406],
        #                     [0.229, 0.224, 0.225])
    ])
}
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]" Using the mean and std of Imagenet is a common practice. They are calculated based on millions of images. If you want to train from scratch on your own dataset, you can calculate the new mean and std. Otherwise, using the Imagenet pretrianed model with its own mean and std is recommended.
# Whether or not to use ImageNet's mean and stddev depends on your data. Assuming your data are ordinary photos of "natural scenes"† (people, buildings, animals, varied lighting/angles/backgrounds, etc.), and assuming your dataset is biased in the same way ImageNet is (in terms of class balance), then it's ok to normalize with ImageNet's scene statistics. If the photos are "special" somehow (color filtered, contrast adjusted, uncommon lighting, etc.) or an "un-natural subject" (medical images, satellite imagery, hand drawings, etc.) then I would recommend correctly normalizing your dataset before model training!*
# Load data from folders
dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}
 
# Size of train and validation data
dataset_sizes = {
    'train':len(dataset['train']),
    'valid':len(dataset['valid'])
}

# Create iterators for data loading
dataloaders = {
    'train':data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=True),
    'valid':data.DataLoader(dataset['valid'], batch_size=vbs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=True)
}

# Class names or target labels
class_names = dataset['train'].classes
print("Classes:", class_names)
 
# Print the train and validation data sizes
print("Training-set size:",dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'])

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if train_mode=='finetune':
    # Load a pretrained model - Resnet18
    #print("\nLoading resnet18 for finetuning ...\n")
    #model_ft = models.resnet18(pretrained=True)
    #print("\nLoading resnet34 for finetuning ...\n")
    #model_ft = models.resnet34(pretrained=True)
    print("\nLoading resnet152 for finetuning ...\n")
    model_ft = models.resnet152(pretrained=True)
    #print("\nLoading Googlenet for finetuning ...\n")
    #model_ft = models.googlenet(pretrained=True)


    # Modify fc layers to match num_classes
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes )

elif train_mode=='scratch':
    # Load a custom model - VGG11
    print("\nLoading VGG11 for training from scratch ...\n")
    #model_ft = MyVGG11(in_ch=3,num_classes=11)
    model_ft = MyVGG11(in_ch=3,num_classes=4)

    # Set number of epochs to a higher value
    num_epochs=10 # default is 100

elif train_mode=='transfer':
    # Load a pretrained model - MobilenetV2
    print("\nLoading mobilenetv2 as feature extractor ...\n")
    model_ft = models.mobilenet_v2(pretrained=True)    

    # Freeze all the required layers (i.e except last conv block and fc layers)
    for params in list(model_ft.parameters())[0:-5]:
        params.requires_grad = False

    # Modify fc layers to match num_classes
    num_ftrs=model_ft.classifier[-1].in_features
    model_ft.classifier=nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
        )    

# Transfer the model to GPU
model_ft = model_ft.to(device)

# Print model summary
print('Model Summary:-\n')
for num, (name, param) in enumerate(model_ft.named_parameters()):
    print(num, name, param.requires_grad )
summary(model_ft, input_size=(3, 400, 500))
print(model_ft)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer 
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9) # RJL learning rate!

# Learning rate decay
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) # every 7 epochs lr is multiplied by gamma

# Model training routine 
print("\nTraining:-\n")
def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Tensorboard summary
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            #print("running_corrects="+running_corrects);
            #epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
# Save the entire model
print("\nSaving the model...")
torch.save(model_ft, PATH)

'''
Sample run: python train.py --mode=finetune
'''
