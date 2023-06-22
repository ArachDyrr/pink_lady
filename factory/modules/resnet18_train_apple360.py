from datetime import datetime
from modules.myFunctions import test_model_360
from modules.myTrainFunctions import train, set_device
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import modules.cosmos_functions as cf
import torch.optim as optim
import torch
import wandb


# hyperparameters
learning_rate = 0.0001
epochs = 2
betas = (0.9, 0.999)
momentum = 0.1
dropout = 0.1  # does not influence resnets dropout
amsgrad = False
optchoice = "adam"  # 'sgd' or 'adam'
early_stopping = False


# set the device
device = set_device()


# import the resnet18 model from pytorch and set output to 4 classes
model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 24)
model.to(device)
model.train()


# load the dataset, tranform and normalise it

transform = T.Compose(
    [
        T.ToTensor(),
        T.transforms.Resize((224, 224), antialias=True),
        # T.transforms.Resize((256, 256), antialias=True),
        # T.transforms.RandomCrop((224, 224)),
        T.transforms.RandomHorizontalFlip(),
        T.transforms.RandomVerticalFlip(),
        # T.transforms.RandomRotation(25),  # Randomly rotate the image by a maximum of 30 degrees
        # T.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Slightly change the image color
        # T.transforms.RandomGrayscale(p=0.1),  # Randomly convert the image to grayscale with a probability of 10%
        # T.transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),  # Randomly erase rectangular patches of the image with a probability of 10%
        # # T.transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3),  # Randomly apply a perspective transformation to the image with a probability of 10%
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



dataset_path_train = "./storage/images/apple360/Training"
dataset_path_val = "./storage/images/apple360/Validation"
train_dataset = ImageFolder(dataset_path_train, transform=transform)
val_dataset = ImageFolder(dataset_path_val, transform=transform)

# Create the DataLoader to load the dataset in batches
batch_size = 32
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)  # num_workers uitzoeken of het zin heeft met MPS
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# optimizer selector from hyperparameters
if optchoice == "adam":
    optimizer = optim.Adam(
        model.parameters(), betas=betas, amsgrad=amsgrad, lr=learning_rate
    )
    print("optimizer = Adam")
elif optchoice == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    print("optimizer = sgd")


# loss function
criterion = nn.CrossEntropyLoss()  # Define the loss function


# set epochloss to empty list
epoch_loss = []


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
userAccountID = cf.settings["userAccountID"]
saveFileName = f"{timestamp}_{userAccountID}"
# create dictionairies for the hyperparameters and model parameters
hyperparameters = {
    "saveFileName": saveFileName,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "momentum": momentum,
    "betas": betas,
    "dropout": dropout,
}
model_parameters = {
    "model": "resnet18",
    "optimizer": optchoice,
    "criterion": "CrossEntropyLoss",
}  # 'model' : model, 'optimizer' : optimizer, 'criterion' : criterion
parameters = {**hyperparameters, **model_parameters}  # merge the two dictionairies


# start a new wandb run to track this script
wandb.init(project="resnet18_224x224_apple360_test", config=parameters)


# train the model
training = train(
    model, train_loader, val_loader, criterion, optimizer, epochs, device, early_stopping, saveFileName
)


# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

# test the model.


_, _, file_data = training
loadpath = f'.{file_data["local_save_path"]}/{file_data["min_loss_file"]}'
print('#-----------------------') 

#test the model

testing_model = torch.load(loadpath)
test_data_path = './storage/images/apple360/Test'
test_model_360(testing_model, test_data_path, device, 32)