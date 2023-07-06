# imports
import torch
from modules.myFunctions import set_device, test_model, test_model_more

# Load the test dataset
dataset_path = "./storage/images/more_apples/original_test"
# import the model state
imported_model_state_path = "./hall_of_fame/kaggle_resnet18_weightbalanced/20230622-111305_resnet18_more_pinky_loss.pt"
# set the device
device = set_device()
# import the model
model = torch.load(imported_model_state_path)
model.to(device)    

print('#------------------------------------------------------------------------')
print("train dataset: more_apples")
print('#------------------------------------------------------------------------')
# test the model
test_model_more(model, dataset_path, device)



print('#------------------------------------------------------------------------')
print("more_apples")
print('#------------------------------------------------------------------------')
dataset_more = "./storage/images/more_apples/test"
result_more= test_model_more(model, dataset_more, device, 32)
# print(result_more)
# print(type(result_more['Confusion Matrix']))