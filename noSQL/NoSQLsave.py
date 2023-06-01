from azure.cosmos import CosmosClient
import base64
from config import settings, userAccountID
from datetime import datetime

from modelhyperparameters import runx   # import the model hyperparameters

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

# create a timestamp and a unique ID
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
unique_id = f'{timestamp}_{userAccountID}'

# get the .pth file and encode it to base64
pth_file_path = '/Users/stephandekker/workspace/pink_lady/noSQL/pth/generated/use_this_Sonar_NN_35rl_53S_1_b15_e400_lr-3.pth'
with open(pth_file_path, 'rb') as file:
    pth_data = file.read()
pth_bytes = base64.b64encode(pth_data).decode('utf-8')



# create the item to be saved on Cosmos DB
# document = {'id': unique_id, 'pth_data': pth_bytes, 'runtime': timestamp, 'run_id': runx['run_id'], 'model': runx['model'], 'learn_rate': runx['learn_rate'], 'epochs': runx['epochs'], 'test_loss': runx['test_loss'], 'test_loss_epoch': runx['test_loss_epoch'], 'train_loss': runx['train_loss'], 'test_acc': runx['test_acc'], 'train_acc': runx['train_acc'], 'test_acc_epoch': runx['test_acc_epoch'], 'optimizer': runx['optimizer'], 'dropout': runx['dropout'], 'momentum': runx['momentum'], 'batch_size': runx['batch_size'], 'num_workers': runx['num_workers'], 'seed': runx['seed']}
document = {'id': unique_id, 'pth_data': pth_bytes, 'hyperparameters': runx}

container.create_item(document)
print('Item with id \'{0}\' created'.format(unique_id))
