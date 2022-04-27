import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset
import model
import trainers

print("nlb_run started...!")
num_epochs = 500
results_dir = '/scratch/gilbreth/ahmedb/vae/lstm_saved_results/small'

data = dataset.nlb_data()
loader = DataLoader(data, batch_size=512, shuffle=True)
criterion = nn.PoissonNLLLoss(log_input=False)


inp, label = next(iter(loader))
print(f"Shape of inputs: {inp.shape}")
print(f"Shape of labels: {label.shape}")

in_units = inp.shape[2]
out_units = label.shape[2]
t_out = label.shape[1]

vae = model.lstm_ae(in_units, out_units, t_out, drop=0.46)
trainer = trainers.VAE_trainer(vae,results_dir)
train_loss, test_loss = trainer.train(loader, num_epochs, criterion, save_checkpoint=True)


print("Successful run...!")


