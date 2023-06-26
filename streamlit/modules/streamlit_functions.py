import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import io

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

    # print(f"Device is '{device}'")
    return device

class StreamlitDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)
    
def create_dataset(uploaded_files, image_size=(224, 224)):
    class CustomDataset(Dataset):
        def __init__(self, images, transform=None):
            self.images = images
            self.transform = transform

        # def __getitem__(self, index):
        #     image = Image.open(io.BytesIO(self.images[index].read()))
        #     if self.transform:
        #         image = self.transform(image)
        #     return image
        
        def __getitem__(self, index):
            image = Image.open(io.BytesIO(self.images[index].read()))
            tensor_image = T.ToTensor()(image)
            if self.transform:
                tensor_image = self.transform(tensor_image)
            return tensor_image

        def __len__(self):
            return len(self.images)

    # Define the transformation to apply to the images
    transform = T.Compose([
        T.Resize(image_size),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.ToTensor()
    ])

    # Create a custom dataset using the uploaded files
    dataset = CustomDataset(uploaded_files, transform=transform)


    return dataset

def streamlit_batch_check(model, dataloader_test, device):
    model.eval()
    model.to(device)

    result_list = []

    for batch in dataloader_test:
        batch = batch.to(device)
        with torch.no_grad():
            prediction = model(batch)
            prediction = prediction.argmax(dim=1, keepdim=True)
            result_list.append(prediction)
    
    return result_list
    