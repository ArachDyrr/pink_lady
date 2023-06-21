# imports
import torch
from modules.myFunctions import set_device, test_model, test_model_360

# Load the test dataset
dataset_path = "./storage/images/apple_extended_unedited/Test"
# import the model state
imported_model_state_path = "./storage/data/generated/test_pinky.pt_loss.pt"
# set the device
device = set_device()
# import the model
model = torch.load(imported_model_state_path)

print('#------------------------------------------------------------------------')
print("Original apples")
print('#------------------------------------------------------------------------')
# test the model
test_model(model, dataset_path, device)



print('#------------------------------------------------------------------------')
print("Apples360")
print('#------------------------------------------------------------------------')
dataset_path_360 = "./storage/images/apple360/Test"
result360= test_model_360(model, dataset_path_360, device, 32)
print(result360)
print(type(result360['Confusion Matrix']))