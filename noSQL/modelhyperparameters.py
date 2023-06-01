from datetime import datetime

# create a timestamp and a unique ID
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# model parameters
x = 1
run_nr = f'run_nr_{x}'
new_run_nr = f'run_nr_{x+1}'
model = 'CNN'
learn_rate = 0.001
epochs = 10
test_loss = 0.1
test_loss_epoch = 8
train_loss = 0.2
test_acc = 0.3
train_acc = 0.4
test_acc_epoch =9
run_id = 'run1'
optimizer = 'Adam'
batch_size = 32
dropout = 0.5
momentum = 0.9
weight_decay = 0.0005
num_workers = 4
seed = 42


runx = {'id' : run_nr,
        'runtime' : timestamp,
        'run_id' : run_id,
        'model' : model,
        'learn_rate' : learn_rate,
        'epochs' : epochs,
        'test_loss' : test_loss,
        'test_loss_epoch' : test_loss_epoch,
        'train_loss' : train_loss,
        'test_acc' : test_acc,
        'train_acc' : train_acc,
        'test_acc_epoch' : test_acc_epoch,
        'optimizer' : optimizer,
        'dropout' : dropout,
        'momentum' : momentum,
        'batch_size' : batch_size,
        'num_workers' : 4,
        'seed' : seed
        }
