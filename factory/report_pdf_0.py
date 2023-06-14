# Generate a pdf report of the model performance on the test dataset
# imports
import torch
from modules.myFunctions import set_device, test_model
from modules.pdf_generator import generate_pdf

# Load the test dataset
dataset_path = "./storage/images/apple_resized_128/Test"
# import the model state
imported_model_state_path = "./storage/data/generated/20230613-153713_pinky_loss.pt"   # test to 224x224 accuracy
# set the device
device = set_device() 
# import the model
model = torch.load(imported_model_state_path)

# test the model
report = test_model(model, dataset_path, device)

test_file_name = 'report_test'

generate_pdf(test_file_name, report)