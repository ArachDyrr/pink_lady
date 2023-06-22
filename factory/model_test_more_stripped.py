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
print("test dataset: original_test_fresh_rotten")
print('#------------------------------------------------------------------------')
# test the model
dataset_path = "./storage/images/more_apples/original_test"
unedited = test_model_more(model, dataset_path, device, 32)



# print('#------------------------------------------------------------------------')
# print("more_test_stripped")
# print('#------------------------------------------------------------------------')
# dataset_path_360 = "./storage/images/stripped/more_test/test"
# result_more_stripped= test_model_more(model, dataset_path_360, device, 32)
# # print(result_more)
# # print(type(result_more['Confusion Matrix']))

print('#------------------------------------------------------------------------')
print("test dataset: kagle_dataset_test")
print('#------------------------------------------------------------------------')
dataset_path_more_test = "./storage/images/more_apples/test"
result_more= test_model_more(model, dataset_path_more_test, device, 32)
# # print(result_more)
# # print(type(result_more['Confusion Matrix']))
print('#------------------------------------------------------------------------')
