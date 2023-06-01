from azure.cosmos import CosmosClient
import base64
from config import settings
import torch

# Retrieve settings from the config file
endpoint_uri = settings['host']
auth_key = settings['master_key']
database_name = settings['database_id']
container_name = settings['container_id']

# Create the Cosmos DB client
client = CosmosClient(endpoint_uri, auth_key)

# Get the database and container
database = client.get_database_client(database_name)
container = database.get_container_client(container_name)

# Retrieve the document containing the .pth data (assuming you know the unique_id)
unique_id = "20230531_2310_dyrr"
item = container.read_item(item=unique_id, partition_key=unique_id)

# Retrieve the pth_data from the item
pth_bytes = item['pth_data']

# Specify the path to save the .pth file
pth_file_path = f'/Users/stephandekker/workspace/pink_lady/noSQL/pth/imported/{unique_id}.pth'

# Decode and save the .pth data to a file
decoded_pth_data = base64.b64decode(pth_bytes)
with open(pth_file_path, 'wb') as file:
    file.write(decoded_pth_data)

# Retrieve the hyperparameters
hyperparameters = item['hyperparameters']

# Save the variable as a .pth file
torch.save(decoded_pth_data, pth_file_path)

# # print the hyperparameters
# print(hyperparameters)



