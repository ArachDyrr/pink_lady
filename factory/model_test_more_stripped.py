# imports
import torch
from modules.myFunctions import set_device, test_model, test_model_more
import pandas as pd

# Load the test dataset

# import the model state
imported_model_state_path = "./storage/data/generated/20230616-152021_resnet18_more_pinky_loss.pt"
# set the device
device = set_device()
# import the model
model = torch.load(imported_model_state_path)

print('#------------------------------------------------------------------------')
print("test dataset: more_apple_extended_unedited")
print('#------------------------------------------------------------------------')
# test the model
dataset_path = "./storage/images/more_apples/original_test"
unedited = test_model_more(model, dataset_path, device, 32)



print('#------------------------------------------------------------------------')
print("more_test_stripped")
print('#------------------------------------------------------------------------')
dataset_path_360 = "./build/cropper/cropped/more_test/test"
result_more_stripped= test_model_more(model, dataset_path_360, device, 32)
# print(result_more)
# print(type(result_more['Confusion Matrix']))

print('#------------------------------------------------------------------------')
print("more_test")
print('#------------------------------------------------------------------------')
dataset_path_360 = "./storage/images/more_apples/test"
result_more= test_model_more(model, dataset_path_360, device, 32)
# # print(result_more)
# # print(type(result_more['Confusion Matrix']))

