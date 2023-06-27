# imports
import torch
from modules.myFunctions import set_device, test_model, test_model_more

# Load the test dataset
dataset_path = "./storage/images/apple_extended_unedited/Test"
# import the model state
imported_model_state_path = "./storage/data/generated/20230622-111305_resnet18_more_pinky_loss.pt"
# set the device
# device = set_device()
device = 'cpu'
print(f'device is {device}')
# import the model
model = torch.load(imported_model_state_path)



print('#------------------------------------------------------------------------')
print("more_apples trained with CELoss weightbalancing")
print('#------------------------------------------------------------------------')
print(f'device is {device}')
dataset_path_360 = "./storage/images/more_apples/original_test"
result_more= test_model_more(model, dataset_path_360, device, 32)
# print(result_more)
# print(type(result_more['Confusion Matrix']))


# device = set_device()
imported_model_state_path = "./storage/data/generated/20230622-111305_resnet18_more_pinky_acc.pt"
model = torch.load(imported_model_state_path)


print('#------------------------------------------------------------------------')
print("more_apples without weightbalancing")
print('#------------------------------------------------------------------------')
print(f'device is {device}')
dataset_path_360 = "./storage/images/more_apples/original_test"
result_more= test_model_more(model, dataset_path_360, device, 32)
# print(result_more)
# print(type(result_more['Confusion Matrix']))