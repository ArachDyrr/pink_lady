import azure.cosmos.documents as documents
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
import datetime

import config

# ----------------------------------------------------------------------------------------------------------
# Prerequistes -
#
# 1. An Azure Cosmos account -
#    https://docs.microsoft.com/azure/cosmos-db/create-cosmosdb-resources-portal#create-an-azure-cosmos-db-account
#   In the Azure portal, configure the following environment variables under Settings->Configuration->Application Settings
#   with your own values:
#   "ACCOUNT_HOST": "<URL of your Cosmos account>",
#   "ACCOUNT_KEY": "<your Cosmos account key>",
#   "COSMOS_DATABASE": "<database name>",
#   "COSMOS_CONTAINER": "<container name>"
#
# 2. Microsoft Azure Cosmos PyPi package -
#    https://pypi.python.org/pypi/azure-cosmos/
# ----------------------------------------------------------------------------------------------------------
# Sample - demonstrates the basic CRUD operations on a Item resource for Azure Cosmos
# ----------------------------------------------------------------------------------------------------------

HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']
DATABASE_ID = config.settings['database_id']
CONTAINER_ID = config.settings['container_id']


# create a test json file to store the NN's hyper parameters, epochs, learnrate, etc.

test_json = { NeuralNet = 'cnn',
                Epochs = 10,
                LearnRate = 0.001,
                BatchSize = 32,
                Optimizer = 'Adam',
                Loss = 'categorical_crossentropy',
                Metrics = ['accuracy'],
                Dropout = 0.2,
                Activation = 'relu',
                Padding = 'same',
                Pooling = 'max',
                Conv2D = 32,
                KernelSize = (3,3),
                Strides = (1,1),
                InputShape = (28,28,1),
                Dense = 128,
                OutputDense = 10,
                OutputActivation = 'softmax',
                OutputShape = 10
            }





# create the function to store the test_json in the database


