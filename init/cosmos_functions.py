from azure.cosmos import CosmosClient
import base64
from config import *
import torch
from datetime import datetime



# function to load the hyperparameters from Cosmos NoSQL DB
def cosmos_load_parameters(unique_id ,parameter, settings=settings):
    
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
    item = container.read_item(item=unique_id, partition_key=unique_id)


    # Retrieve the hyperparameters
    parameter = item[f'{parameter}']
   
    return parameter

# function to load the .pth file from Cosmos NoSQL DB
def cosmos_load_pth(unique_id, local_save_path=None, settings=settings): 
        
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
        item = container.read_item(item=unique_id, partition_key=unique_id)
    
        # Retrieve the pth_data from the item
        pth_bytes = item['pth_data']
    
        # Decode and save the .pth data to a file
        decoded_pth_data = base64.b64decode(pth_bytes)

        # Save the variable as a .pth file
        if local_save_path != None:
            pth_file_path = f'{local_save_path}/{unique_id}.pth'
            torch.save(decoded_pth_data, pth_file_path)
    
        output = decoded_pth_data
        return output

# function to save the .pth file to Cosmos NoSQL DB
def cosmos_save_data(pth_data=None, parameters=None, run_result=None, settings=settings):   
    
    # Retrieve settings from the config file
    endpoint_uri = settings['host']
    auth_key = settings['master_key']
    database_name = settings['database_id']
    container_name = settings['container_id']
    userAccountID = settings['userAccountID']

    # Create the Cosmos DB client
    client = CosmosClient(endpoint_uri, auth_key)

    # Get the database and container
    database = client.get_database_client(database_name)
    container = database.get_container_client(container_name)

    # create a timestamp and a unique ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    unique_id = f'{timestamp}_{userAccountID}'

    # get the .pth file and encode it to base64
    if pth_data != None:
        pth_file_path = '/Users/stephandekker/workspace/pink_lady/noSQL/pth/generated/use_this_Sonar_NN_35rl_53S_1_b15_e400_lr-3.pth'
        with open(pth_file_path, 'rb') as file:
            pth_data = file.read()
        pth_bytes = base64.b64encode(pth_data).decode('utf-8')



    # create the item to be saved on Cosmos DB
    # document = {'id': unique_id, 'pth_data': pth_bytes, 'runtime': timestamp, 'run_id': runx['run_id'], 'model': runx['model'], 'learn_rate': runx['learn_rate'], 'epochs': runx['epochs'], 'test_loss': runx['test_loss'], 'test_loss_epoch': runx['test_loss_epoch'], 'train_loss': runx['train_loss'], 'test_acc': runx['test_acc'], 'train_acc': runx['train_acc'], 'test_acc_epoch': runx['test_acc_epoch'], 'optimizer': runx['optimizer'], 'dropout': runx['dropout'], 'momentum': runx['momentum'], 'batch_size': runx['batch_size'], 'num_workers': runx['num_workers'], 'seed': runx['seed']}
    document = {'id': unique_id, 'pth_data': pth_bytes, 'parameters': parameters, 'run_result': run_result}
    container.create_item(document)
    print('Item with id \'{0}\' created'.format(unique_id))

# test function of all functions in this file. 
def run_tests():
    # test the load_parameters function
    unique_id = "20230602_1600_pinky"
    id = 'id'
    parameter = 'parameters'
    run_result = 'run_result'  
    id = cosmos_load_parameters(unique_id, id) 
    run_result = cosmos_load_parameters(unique_id, run_result)
    parameter = cosmos_load_parameters(unique_id, parameter)
    print(f'''
    id = {id}
    run_result = {run_result}
    parameter = {parameter}

    ''')
    
    # test the load_pth function
    local_save_path = '/Users/stephandekker/workspace/pink_lady/noSQL/pth/imported'
    unique_id = "20230602_1600_pinky"
    pth = cosmos_load_pth(unique_id, local_save_path)
    print(pth)
    print()
    

    
    # test the save_data function
    parameters = {'learning_rate': 0.01, 'epochs': 1, 'momentum': 0.9, 'dropout': 0.2, 'model': 'cnn', 'optimizer': 'SGD', 'criterion': 'CrossEntropyLoss'}
    history = {'n_epochs': 1, 'loss': {'train': [2.1098746801053405], 'val': [1.8478429771352698]}, 'acc': {'train': [19.64021164545937], 'val': [23.580246907928842]}}
    pth_file_path = '20230602-143630_pinky.pt'
    cosmos_save_data(pth_file_path, parameters, history)


    
# command to only run the tests if this file is run directly
if __name__ == '__main__':
    run_tests()