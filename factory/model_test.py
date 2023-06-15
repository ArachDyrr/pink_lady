# imports
import torch
from modules.myFunctions import set_device, test_model

# Load the test dataset
dataset_path = "./storage/images/apple_extended_unedited/Test"
# import the model state
imported_model_state_path = "/home/arachdyrr/workspace/pink_lady/storage/data/generated/20230613-231803_PinkLady_loss.pt"
# set the device
device = set_device()
# import the model
model = torch.load(imported_model_state_path)

# test the model
test_model(model, dataset_path, device)
