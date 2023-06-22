# imports
import torch
from modules.myFunctions import set_device, test_model, test_model_more

# Load the test dataset
dataset_path = "./storage/images/apple_extended_unedited/Test"
# import the model state
imported_model_state_path = "./hall_of_fame/swin_t/20230621-002105_resnet18_more_pinky_loss.pt"
# set the device
device = set_device()
device = 'cpu'
# import the model
model = torch.load(imported_model_state_path)

# print('#------------------------------------------------------------------------')
# print("train dataset: more_apples")
# print('#------------------------------------------------------------------------')
# # test the model
# test_model(model, dataset_path, device)



print('#------------------------------------------------------------------------')
print("more_apples")
print('#------------------------------------------------------------------------')
dataset_more = "./storage/images/more_apples/original_test"
result_more= test_model_more(model, dataset_more, device, 32)
# print(result_more)
# print(type(result_more['Confusion Matrix']))