# imports
import torch
from modules.myFunctions import set_device, test_model

# Load the test dataset
dataset_path = "./storage/images/apple_resized_128/Test"
# import the model state
imported_model_state_path = "./storage/data/generated/20230613-153713_pinky_loss.pt"   # test to 224x224 accuracy
# set the device
device = set_device() 
# import the model
model = torch.load(imported_model_state_path)

# test the model
test_model(model, dataset_path, device)