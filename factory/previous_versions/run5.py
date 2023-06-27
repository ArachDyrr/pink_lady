import subprocess

for _ in range(5):
    subprocess.run(['python', 'factory/resnet18_train_more_criterion_weights.py'])