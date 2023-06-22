import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
if __name__ == "__main__":
    from myTrainFunctions import set_device
else:
    from modules.myTrainFunctions import set_device

device = set_device()

class MyCNN_97531_max(nn.Module):
    def __init__(self, dropout_rate):
        super(MyCNN_97531_max, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        # Calculate the input size for the fully connected layer
        self.fc_input_size = self._calculate_fc_input_size()

        # Define the fully connected layer
        self.fc = nn.Linear(self.fc_input_size, 4)

        # Define the activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Set the dropout rate

    def forward(self, x):
        # Apply the convolutional layers with dropout and relu
        x = self.relu(self.conv1(x))
        x = self.dropout(x)

        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        x = self.relu(self.conv3(x))
        x = self.dropout(x)

        x = self.relu(self.conv4(x))
        x = self.dropout(x)

        x = self.relu(self.conv5(x))
        x = self.dropout(x)

        # Flatten the tensor before the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer with relu and dropout
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _calculate_fc_input_size(self):
        # Generate a random input tensor to calculate the output shape after convolutions
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
        # Flatten the tensor before the fully connected layer
        flattened_size = x.view(1, -1).size(1)
        return flattened_size
    
    def get_flattensize(self):
        return self.fc_input_size

# # Create an instance of the network with a specific dropout rate
# dropout_rate = 0.2  # Set your desired dropout rate
# model = MyCNN_97531(dropout_rate)
# # Create an instance of the network


# # Create a random input tensor with shape [batch_size, 3, 224, 224]
# input_tensor = torch.randn(10, 3, 224, 224)

# # Forward pass
# output = model(input_tensor)
# print(output.shape)  # Should be [batch_size, 4]

import torch
import torch.nn as nn

class MyCNN_97531_checkpoint(nn.Module):
    def __init__(self, dropout_rate):
        super(MyCNN_97531_checkpoint, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        # Calculate the input size for the fully connected layer
        self.fc_input_size = self._calculate_fc_input_size()

        # Define the fully connected layer
        self.fc = nn.Linear(self.fc_input_size, 4)

        # Define the activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Set the dropout rate


    def forward(self, x):
        # Apply the convolutional layers with dropout and relu
        x = checkpoint(self._conv1_forward, x)
        x = self.dropout(x)

        x = checkpoint(self._conv2_forward, x)
        x = self.dropout(x)

        x = checkpoint(self._conv3_forward, x)
        x = self.dropout(x)

        x = checkpoint(self._conv4_forward, x)
        x = self.dropout(x)

        x = checkpoint(self._conv5_forward, x)
        x = self.dropout(x)

        # Flatten the tensor before the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer with relu and dropout
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _calculate_fc_input_size(self):
        # Generate a random input tensor to calculate the output shape after convolutions
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
        # Flatten the tensor before the fully connected layer
        flattened_size = x.view(1, -1).size(1)
        return flattened_size

    def _conv1_forward(self, x):
        return self.relu(self.conv1(x))

    def _conv2_forward(self, x):
        return self.relu(self.conv2(x))

    def _conv3_forward(self, x):
        return self.relu(self.conv3(x))

    def _conv4_forward(self, x):
        return self.relu(self.conv4(x))

    def _conv5_forward(self, x):
        return self.relu(self.conv5(x))


class MyCNN_97531half(nn.Module):
    def __init__(self, dropout_rate):
        super(MyCNN_97531half, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Calculate the input size for the fully connected layer
        self.fc_input_size = self._calculate_fc_input_size()

        # Define the fully connected layer
        self.fc = nn.Linear(self.fc_input_size, 4)

        # Define the activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Set the dropout rate

    def forward(self, x):
        # Apply the convolutional layers with dropout and relu
        x = self.relu(self.conv1(x))
        x = self.dropout(x)

        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        x = self.relu(self.conv3(x))
        x = self.dropout(x)

        x = self.relu(self.conv4(x))
        x = self.dropout(x)

        x = self.relu(self.conv5(x))
        x = self.dropout(x)

        # Flatten the tensor before the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer with relu and dropout
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _calculate_fc_input_size(self):
        # Generate a random input tensor to calculate the output shape after convolutions
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
        # Flatten the tensor before the fully connected layer
        flattened_size = x.view(1, -1).size(1)
        return flattened_size

class MyCNN_97531quart(nn.Module):
    def __init__(self, dropout_rate):
        super(MyCNN_97531quart, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

        # Calculate the input size for the fully connected layer
        self.fc_input_size = self._calculate_fc_input_size()

        # Define the fully connected layer
        self.fc = nn.Linear(self.fc_input_size, 4)

        # Define the activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Set the dropout rate

    def forward(self, x):
        # Apply the convolutional layers with dropout and relu
        x = self.relu(self.conv1(x))
        x = self.dropout(x)

        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        x = self.relu(self.conv3(x))
        x = self.dropout(x)

        x = self.relu(self.conv4(x))
        x = self.dropout(x)

        x = self.relu(self.conv5(x))
        x = self.dropout(x)

        # Flatten the tensor before the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer with relu and dropout
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _calculate_fc_input_size(self):
        # Generate a random input tensor to calculate the output shape after convolutions
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
        # Flatten the tensor before the fully connected layer
        flattened_size = x.view(1, -1).size(1)
        return flattened_size

class MyCNN_975318th(nn.Module):
    def __init__(self, dropout_rate):
        super(MyCNN_975318th, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        # Calculate the input size for the fully connected layer
        self.fc_input_size = self._calculate_fc_input_size()

        # Define the fully connected layer
        self.fc = nn.Linear(self.fc_input_size, 4)

        # Define the activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Set the dropout rate

    def forward(self, x):
        # Apply the convolutional layers with dropout and relu
        x = self.relu(self.conv1(x))
        x = self.dropout(x)

        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        x = self.relu(self.conv3(x))
        x = self.dropout(x)

        x = self.relu(self.conv4(x))
        x = self.dropout(x)

        x = self.relu(self.conv5(x))
        x = self.dropout(x)

        # Flatten the tensor before the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer with relu and dropout
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _calculate_fc_input_size(self):
        # Generate a random input tensor to calculate the output shape after convolutions
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
        # Flatten the tensor before the fully connected layer
        flattened_size = x.view(1, -1).size(1)
        return flattened_size


class MyCNN_7531(nn.Module):
    def __init__(self, dropout_rate):
        super(MyCNN_7531, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 4, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(8, 64, kernel_size=1, stride=1, padding=0)

        # Calculate the input size for the fully connected layer
        self.fc_input_size = self._calculate_fc_input_size()

        # Define the fully connected layer
        self.fc = nn.Linear(self.fc_input_size, 4)

        # Define the activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Set the dropout rate

    def forward(self, x):
        # Apply the convolutional layers with dropout and relu
        x = self.relu(self.conv1(x))
        x = self.dropout(x)

        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        x = self.relu(self.conv3(x))
        x = self.dropout(x)

        x = self.relu(self.conv4(x))
        x = self.dropout(x)

        x = self.relu(self.conv5(x))
        x = self.dropout(x)

        # Flatten the tensor before the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer with relu and dropout
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _calculate_fc_input_size(self):
        # Generate a random input tensor to calculate the output shape after convolutions
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
        # Flatten the tensor before the fully connected layer
        flattened_size = x.view(1, -1).size(1)
        return flattened_size


class MyCNN_7531(nn.Module):
    def __init__(self, dropout_rate):
        super(MyCNN_7531, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(8, 64, kernel_size=1, stride=1, padding=0)

        # Calculate the input size for the fully connected layer
        self.fc_input_size = self._calculate_fc_input_size()

        # Define the fully connected layer
        self.fc = nn.Linear(self.fc_input_size, 4)

        # Define the activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Set the dropout rate

    def forward(self, x):
        # Apply the convolutional layers with dropout and relu
        x = self.relu(self.conv1(x))
        x = self.dropout(x)

        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        x = self.relu(self.conv3(x))
        x = self.dropout(x)

        x = self.relu(self.conv4(x))
        x = self.dropout(x)

        # Flatten the tensor before the fully connected layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer with relu and dropout
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _calculate_fc_input_size(self):
        # Generate a random input tensor to calculate the output shape after convolutions
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
        # Flatten the tensor before the fully connected layer
        flattened_size = x.view(1, -1).size(1)
        return flattened_size


def test():
    # Create an instance of the network with a specific dropout rate
    dropout_rate = 0.2  # Set your desired dropout rate
    model = MyCNN_97531_checkpoint(dropout_rate)
    model.to(device)
    # # Create an instance of the network


    # Create a random input tensor with shape [batch_size, 3, 224, 224]
    input_tensor = torch.randn(10, 3, 224, 224, requires_grad=True)
    input_tensor = input_tensor.to(device)
    # Forward pass
    output = model(input_tensor)
    print(output.shape)  # Should be [batch_size, 4]


if __name__ == "__main__":
    test()
