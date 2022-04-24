import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pylab as plt
import dataset
import model
import pickle
import trainers
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 100
data_dir = '/scratch/gilbreth/ahmedb/CelebA'
results_dir = '/scratch/gilbreth/ahmedb/vae/saved_results'

train_data = dataset.CelebA(data_dir, download=False, split='train')
train_loader = train_data.load_data(batch_size=64, shuffle=True)
test_data = dataset.CelebA(data_dir, download=False, split='test')
test_loader = train_data.load_data(batch_size=64, shuffle=True)
vae = model.CAE()

trainer = trainers.VAE_trainer(vae,results_dir)

train_loss, test_loss = trainer.train(train_loader,num_epochs)
trainer.save_checkpoint(num_epochs)

loss = {'train_loss': train_loss, 'test_loss': test_loss}
filename='autencoder_loss_history.pickle'
with open(os.path.join(results_dir, filename),'wb') as f:
    pickle.dump(loss, f)   

print("Successful run...!")
