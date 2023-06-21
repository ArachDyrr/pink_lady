import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)

# BEPAAL DE SCOPE VAN DEZE MODULE

# quick and dirty fix for import pathing.
if __name__ == "__main__":
    from MyAQLclass import MyAQLclass  # use for testing
else:
    from modules.MyAQLclass import MyAQLclass  # use for testing


# function to set device to GPU/mps if available
def set_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Device is '{device}'")
    return device


# function to test the AQL score and the model
def AQL_test_model(model, datasetPath, device):
    model.eval()

    # Load the test dataset
    dataset_path = datasetPath
    transform = T.Compose(
        [
            T.ToTensor(),
            T.transforms.Resize((256, 256), antialias=True),
            T.transforms.RandomCrop((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # pull the relevant AQL data
    AQLtest = MyAQLclass()
    lotsize = AQLtest.get_lotsize()
    test_inspection_lvl = AQLtest.get_test_inspection_lvl()
    batch_size = AQLtest.batch_size()

    dataset = ImageFolder(dataset_path, transform=transform)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    labels_dict = dataset.class_to_idx

    # Track the overall test accuracy and accuracy by each type of apple
    overall_correct = 0
    overall_total = 0
    normal_correct = 0
    normal_total = 0
    abnormal_correct = 0
    abnormal_total = 0

    # Initialize the confusion matrix
    num_classes = len(labels_dict)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Iterate over the test dataset
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Get predictions
        _, predicted = torch.max(outputs.data, 1)

        # Update accuracy counts
        overall_correct += (predicted == labels).sum().item()
        overall_total += labels.size(0)

        # Calculate accuracy for normal apples vs. abnormal apples
        normal_mask = labels == labels_dict["Normal_Apple"]
        abnormal_mask = ~normal_mask
        normal_correct += (predicted[normal_mask] == labels[normal_mask]).sum().item()
        normal_total += normal_mask.sum().item()
        abnormal_correct += (
            (predicted[abnormal_mask] == labels[abnormal_mask]).sum().item()
        )
        abnormal_total += abnormal_mask.sum().item()

        # Update the confusion matrix
        for true_label, predicted_label in zip(
            labels.cpu().numpy(), predicted.cpu().numpy()
        ):
            confusion_matrix[true_label][predicted_label] += 1

        # Break the loop after processing the first batch
        if batch_idx == 0:
            break

    # Calculate overall accuracy
    # Prevent division by zero
    if (overall_total != 0):
        overall_accuracy = overall_correct / overall_total

    # Calculate accuracy for normal apples and abnormal apples separately
    normal_accuracy = normal_correct / normal_total if normal_total != 0 else 0.0
    abnormal_accuracy = (
        abnormal_correct / abnormal_total if abnormal_total != 0 else 0.0
    )



    ## IS DIT NOG NODIG?

    # Print overall accuracy
    print(f"Overall accuracy: {overall_accuracy:.4f}")

    # Print accuracy for normal apples and abnormal apples separately
    print(f"Normal Apple accuracy: {normal_accuracy:.4f}")
    print(f"Abnormal Apple accuracy: {abnormal_accuracy:.4f}")

    # Print the confusion matrix
    print()
    print(labels_dict)
    print("Confusion Matrix:")
    print(confusion_matrix)

    # get the AQL label
    rejected_apples = np.sum(confusion_matrix) - np.sum(confusion_matrix[1])

    AQLtest.test_input = rejected_apples
    AQL_label = AQLtest.output()

    print(f"From a lot of {lotsize} in accordance quality level {test_inspection_lvl},")
    print(f"a batch of {batch_size} has been randomly drawn.")
    print(f"the number of rejected apples is: {rejected_apples}")
    print(f"The AQL label is: Class_{AQL_label}")
    # print()
    # print(len(test_dataloader))

    report_dict = {
        "Overall accuracy": overall_accuracy,
        "Normal Apple accuracy": normal_accuracy,
        "Abnormal Apple accuracy": abnormal_accuracy,
        "Labels": labels_dict,
        "Confusion Matrix": confusion_matrix,
        "lots size": lotsize,
        "test_inspection_lvl": test_inspection_lvl,
        "batch_size": batch_size,
        "rejected apples": rejected_apples,
        "AQL label": AQL_label,
    }
    return report_dict


def test_model(model, datasetPath, device, batchSize=None):
    model.eval()

    # Load the test dataset
    dataset_path = datasetPath
    transform = T.Compose(
        [
            T.ToTensor(),
            T.transforms.Resize((256, 256), antialias=True),
            T.transforms.RandomCrop((224, 224)),
            # T.transforms.RandomHorizontalFlip(),
            # T.transforms.RandomVerticalFlip(),
            # T.transforms.RandomRotation(25),  # Randomly rotate the image by a maximum of 30 degrees
            # T.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Slightly change the image color
            # T.transforms.RandomGrayscale(p=0.1),  # Randomly convert the image to grayscale with a probability of 10%
            # T.transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),  # Randomly erase rectangular patches of the image with a probability of 10%
            # # T.transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3),  # Randomly apply a perspective transformation to the image with a probability of 10%
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageFolder(dataset_path, transform=transform)

    if batchSize is None:
        batch_size = len(dataset)
    else:
        batch_size = batchSize

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    labels_dict = dataset.class_to_idx

    # Track the overall test accuracy and accuracy by each type of apple
    overall_correct = 0
    overall_total = 0
    normal_correct = 0
    normal_total = 0
    abnormal_correct = 0
    abnormal_total = 0

    # Initialize the confusion matrix
    num_classes = len(labels_dict)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Iterate over the test dataset
    # Iterate over the test dataset
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Get predictions
        _, predicted = torch.max(outputs.data, 1)

        # Update accuracy counts
        overall_correct += (predicted == labels).sum().item()
        overall_total += labels.size(0)

        # Calculate accuracy for normal apples vs. abnormal apples
        normal_mask = labels == labels_dict["Normal_Apple"]
        abnormal_mask = ~normal_mask
        normal_correct += (predicted[normal_mask] == labels[normal_mask]).sum().item()
        normal_total += normal_mask.sum().item()
        abnormal_correct += (
            (predicted[abnormal_mask] == labels[abnormal_mask]).sum().item()
        )
        abnormal_total += abnormal_mask.sum().item()

        # Update the confusion matrix
        for true_label, predicted_label in zip(
            labels.cpu().numpy(), predicted.cpu().numpy()
        ):
            confusion_matrix[true_label][predicted_label] += 1

        # Break the loop after processing the first batch
        if batchSize is None:
            if batch_idx == 0:
                break

    # Calculate overall accuracy
    overall_accuracy = overall_correct / overall_total

    # Calculate accuracy for normal apples and abnormal apples separately
    normal_accuracy = normal_correct / normal_total if normal_total != 0 else 0.0
    abnormal_accuracy = (
        abnormal_correct / abnormal_total if abnormal_total != 0 else 0.0
    )

    # Print overall accuracy
    print(f"Overall accuracy: {overall_accuracy:.4f}")

    # Print accuracy for normal apples and abnormal apples separately
    print(f"Normal Apple accuracy: {normal_accuracy:.4f}")
    print(f"Abnormal Apple accuracy: {abnormal_accuracy:.4f}")

    # Print the confusion matrix
    print()
    print(labels_dict)
    print("Confusion Matrix:")
    print(confusion_matrix)

    test_dict = {
        "Overall accuracy": overall_accuracy,
        "Normal Apple accuracy": normal_accuracy,
        "Abnormal Apple accuracy": abnormal_accuracy,
        "Labels": labels_dict,
        "Confusion Matrix": confusion_matrix,
        "batch_size": batch_size,
    }
    return test_dict


# to convert a number to a roman numeral
def number_to_roman(number):
    roman_mapping = {
        1000: "M",
        900: "CM",
        500: "D",
        400: "CD",
        100: "C",
        90: "XC",
        50: "L",
        40: "XL",
        10: "X",
        9: "IX",
        5: "V",
        4: "IV",
        1: "I",
    }

    roman_numeral = ""
    for value, symbol in roman_mapping.items():
        while number >= value:
            roman_numeral += symbol
            number -= value

    return roman_numeral


# KAN DIT WEG?
# Move all .heic files from a source folder to a destination folder
def move_files(source_folder, destination_folder, file_extension=".heic"):
    # Create the destination folder if it doesn't exist
    # os.makedirs(destination_folder, exist_ok=True)

    # Get all files in the source folder
    files = os.listdir(source_folder)

    # Filter only .heic files
    heic_files = [f for f in files if f.endswith(file_extension)]

    # Move each .heic file to the destination folder
    for file in heic_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.move(source_path, destination_path)


# move files containing "_" and "apple" to respective folders
def move_apple_files(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Get all files in the source folder
    files = os.listdir(source_folder)

    # Move files containing "apple" to respective folders
    for file in files:
        if "apple" in file:
            # Extract the prefix from the file name
            prefix = file.split("_")[0]

            # Create the destination folder for the prefix if it doesn't exist
            prefix_folder = os.path.join(destination_folder, prefix)
            os.makedirs(prefix_folder, exist_ok=True)

            # Move the file to the destination folder
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(prefix_folder, file)
            shutil.move(source_path, destination_path)



# NIET GEBRUIKEN IN RUNTIME OMGEVING
# functions to display images
def reverse_normalize(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image.clone()
    for i in range(3):
        image[i] = (image[i] * std[i]) + mean[i]
    return image


def show_batch(test_d):
    # Get the first batch of data from the DataLoader
    data_test = next(iter(test_d))

    # Retrieve the first tensor and its corresponding label
    image_test = data_test[0][0]
    label_test = data_test[1][0]

    # Reverse the normalization of the image
    image_test = reverse_normalize(image_test)

    # Convert the image tensor to a NumPy array and transpose the dimensions
    np_image_test = image_test.permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(np_image_test)
    plt.title(f"{label_test}, {image_test.shape}")
    plt.axis("off")

    # Show the plot
    plt.show()


# # test for function to move files from one folder to another
# fromf = '/Users/stephandekker/workspace/pink_lady/storage/images/heic_apples'
# tof = '/Users/stephandekker/workspace/pink_lady/storage/images/apple_extended_unedited/Test/Normal_Apple'
# move_files(fromf, tof, '.jpg')

# IS DIT NOG NODIG?

# acces the optimiser
def optimiser_state(optimizer):
    # Access the optimizer's state dictionary
    optimizer_state = optimizer.state_dict()

    # Print the parameter groups
    for i, param_group in enumerate(optimizer_state["param_groups"]):
        print(f"Parameter Group {i+1}:")
        for key, value in param_group.items():
            print(f"    {key}: {value}")
        print()

# VERPLAATSEN NAAR APARTE TEST MODULE?
def test_model_360(model, datasetPath, device, batchSize=None):
    model.eval()

    # Load the test dataset
    dataset_path = datasetPath
    transform = T.Compose(
        [
            T.ToTensor(),
            T.transforms.Resize((224, 224), antialias=True),
            # T.transforms.RandomCrop((224, 224)),
            # T.transforms.RandomHorizontalFlip(),
            # T.transforms.RandomVerticalFlip(),
            # T.transforms.RandomRotation(25),  # Randomly rotate the image by a maximum of 30 degrees
            # T.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Slightly change the image color
            # T.transforms.RandomGrayscale(p=0.1),  # Randomly convert the image to grayscale with a probability of 10%
            # T.transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),  # Randomly erase rectangular patches of the image with a probability of 10%
            # # T.transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3),  # Randomly apply a perspective transformation to the image with a probability of 10%
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageFolder(dataset_path, transform=transform)

    if batchSize is None:
        batch_size = len(dataset)
    else:
        batch_size = batchSize

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    labels_dict = dataset.class_to_idx

    # Track the overall test accuracy and accuracy by each type of apple
    overall_correct = 0
    overall_total = 0
    normal_correct = 0
    normal_total = 0
    abnormal_correct = 0
    abnormal_total = 0

    # Initialize the confusion matrix
    num_classes = len(labels_dict)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Iterate over the test dataset
    # Iterate over the test dataset
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Get predictions
        _, predicted = torch.max(outputs.data, 1)

        # Update accuracy counts
        overall_correct += (predicted == labels).sum().item()
        overall_total += labels.size(0)


        # Update the confusion matrix
        for true_label, predicted_label in zip(
            labels.cpu().numpy(), predicted.cpu().numpy()
        ):
            confusion_matrix[true_label][predicted_label] += 1

        # Break the loop after processing the first batch
        if batchSize is None:
            if batch_idx == 0:
                break

    # Calculate overall accuracy
    overall_accuracy = overall_correct / overall_total


    test_dict = {
        "Overall accuracy": overall_accuracy,
        "Labels": labels_dict,
        "Confusion Matrix": confusion_matrix,
        "batch_size": batch_size,
    }
    return test_dict

def test_model(model, datasetPath, device, batch_size=None):
    model.eval()

    # Load the test dataset
    dataset_path = datasetPath
    transform = T.Compose(
        [
            T.ToTensor(),
            T.transforms.Resize((256, 256), antialias=True),
            T.transforms.RandomCrop((224, 224)),
            # T.transforms.RandomHorizontalFlip(),
            # T.transforms.RandomVerticalFlip(),
            # T.transforms.RandomRotation(25),  # Randomly rotate the image by a maximum of 30 degrees
            # T.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Slightly change the image color
            # T.transforms.RandomGrayscale(p=0.1),  # Randomly convert the image to grayscale with a probability of 10%
            # T.transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),  # Randomly erase rectangular patches of the image with a probability of 10%
            # # T.transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3),  # Randomly apply a perspective transformation to the image with a probability of 10%
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageFolder(dataset_path, transform=transform)

    if batch_size is None:
        batch_size = len(dataset)
    else:
        batch_size = batch_size

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    labels_dict = dataset.class_to_idx

    # Track the overall test accuracy and accuracy by each type of apple
    overall_correct = 0
    overall_total = 0
    normal_correct = 0
    normal_total = 0
    abnormal_correct = 0
    abnormal_total = 0

    # Initialize the confusion matrix
    num_classes = len(labels_dict)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Iterate over the test dataset
    # Iterate over the test dataset
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Get predictions
        _, predicted = torch.max(outputs.data, 1)

        # Update accuracy counts
        overall_correct += (predicted == labels).sum().item()
        overall_total += labels.size(0)

        # Calculate accuracy for normal apples vs. abnormal apples
        normal_mask = labels == labels_dict["Normal_Apple"]
        abnormal_mask = ~normal_mask
        normal_correct += (predicted[normal_mask] == labels[normal_mask]).sum().item()
        normal_total += normal_mask.sum().item()
        abnormal_correct += (
            (predicted[abnormal_mask] == labels[abnormal_mask]).sum().item()
        )
        abnormal_total += abnormal_mask.sum().item()

        # Update the confusion matrix
        for true_label, predicted_label in zip(
            labels.cpu().numpy(), predicted.cpu().numpy()
        ):
            confusion_matrix[true_label][predicted_label] += 1

        # Break the loop after processing the first batch
        if batch_idx == 0:
            break

    # Calculate overall accuracy
    overall_accuracy = overall_correct / overall_total

    # Calculate accuracy for normal apples and abnormal apples separately
    normal_accuracy = normal_correct / normal_total if normal_total != 0 else 0.0
    abnormal_accuracy = (
        abnormal_correct / abnormal_total if abnormal_total != 0 else 0.0
    )

    # Print overall accuracy
    print(f"Overall accuracy: {overall_accuracy:.4f}")

    # Print accuracy for normal apples and abnormal apples separately
    print(f"Normal Apple accuracy: {normal_accuracy:.4f}")
    print(f"Abnormal Apple accuracy: {abnormal_accuracy:.4f}")

    # Print the confusion matrix
    print()
    print(labels_dict)
    print("Confusion Matrix:")
    print(confusion_matrix)

    test_dict = {
        "Overall accuracy": overall_accuracy,
        "Normal Apple accuracy": normal_accuracy,
        "Abnormal Apple accuracy": abnormal_accuracy,
        "Labels": labels_dict,
        "Confusion Matrix": confusion_matrix,
        "batch_size": batch_size,
    }
    return test_dict

# NAAR TEST MODULE
def test_model_more(model, datasetPath, device, batchSize=None):
    model.eval()

    # Load the test dataset
    dataset_path = datasetPath
    transform = T.Compose(
        [
            T.ToTensor(),
            T.transforms.Resize((224, 224), antialias=True),
            # T.transforms.Resize((256, 256), antialias=True),
            T.transforms.RandomCrop((224, 224)),
            # T.transforms.RandomHorizontalFlip(),
            # T.transforms.RandomVerticalFlip(),
            # T.transforms.RandomRotation(25),  # Randomly rotate the image by a maximum of 30 degrees
            # T.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Slightly change the image color
            # T.transforms.RandomGrayscale(p=0.1),  # Randomly convert the image to grayscale with a probability of 10%
            # T.transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),  # Randomly erase rectangular patches of the image with a probability of 10%
            # # T.transforms.RandomPerspective(distortion_scale=0.2, p=0.1, interpolation=3),  # Randomly apply a perspective transformation to the image with a probability of 10%
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageFolder(dataset_path, transform=transform)

    if batchSize is None:
        batch_size = len(dataset)
    else:
        batch_size = batchSize

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    labels_dict = dataset.class_to_idx

    # Track the overall test accuracy and accuracy by each type of apple
    overall_correct = 0
    overall_total = 0
    normal_correct = 0
    normal_total = 0
    abnormal_correct = 0
    abnormal_total = 0

    # Initialize the confusion matrix
    num_classes = len(labels_dict)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Iterate over the test dataset
    # Iterate over the test dataset
    for batch_idx, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Get predictions
        _, predicted = torch.max(outputs.data, 1)

        # Update accuracy counts
        overall_correct += (predicted == labels).sum().item()
        overall_total += labels.size(0)

        # Calculate accuracy for normal apples vs. abnormal apples
        normal_mask = labels == labels_dict["freshapples"]
        abnormal_mask = ~normal_mask
        normal_correct += (predicted[normal_mask] == labels[normal_mask]).sum().item()
        normal_total += normal_mask.sum().item()
        abnormal_correct += (
            (predicted[abnormal_mask] == labels[abnormal_mask]).sum().item()
        )
        abnormal_total += abnormal_mask.sum().item()

        # Update the confusion matrix
        for true_label, predicted_label in zip(
            labels.cpu().numpy(), predicted.cpu().numpy()
        ):
            confusion_matrix[true_label][predicted_label] += 1

        # Break the loop after processing the first batch
        if batchSize is None:
            if batch_idx == 0:
                break

    # Calculate overall accuracy
    overall_accuracy = overall_correct / overall_total

    # Calculate accuracy for normal apples and abnormal apples separately
    normal_accuracy = normal_correct / normal_total if normal_total != 0 else 0.0
    abnormal_accuracy = (
        abnormal_correct / abnormal_total if abnormal_total != 0 else 0.0
    )

    # Print overall accuracy
    print(f"Overall accuracy: {overall_accuracy:.4f}")

    # Print accuracy for normal apples and abnormal apples separately
    print(f"Normal Apple accuracy: {normal_accuracy:.4f}")
    print(f"Abnormal Apple accuracy: {abnormal_accuracy:.4f}")

    # Print the confusion matrix
    print()
    print(labels_dict)
    print("Confusion Matrix:")
    print(confusion_matrix)

    test_dict = {
        "Overall accuracy": overall_accuracy,
        "Normal Apple accuracy": normal_accuracy,
        "Abnormal Apple accuracy": abnormal_accuracy,
        "Labels": labels_dict,
        "Confusion Matrix": confusion_matrix,
        "batch_size": batch_size,
    }
    return test_dict