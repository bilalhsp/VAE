import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors
import yaml
import dataset
import model
import trainers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("nlb_run started...!")
num_epochs = 500
with open('./config.yaml', 'r') as f:
    manifest = yaml.load(f)#, Loader=yaml.FullLoader)

hyper_param = manifest['hyper_param']
model_param = manifest['small']
results_dir = model_param['results_dir']

dataset = NWBDataset(fpath=model_param['data_dir'])
dataset.resample(5)
train_dict = make_train_input_tensors(dataset=dataset, 
            dataset_name=model_param['dataset_name'], 
            # trial_split='train', 
            trial_split=['train', 'val'], #for Test phase
            save_file=False, 
            include_forward_pred=True)

eval_dict = make_eval_input_tensors(dataset=dataset,
            dataset_name=model_param['dataset_name'],
            # trial_split='val', 
            trial_split='test',# for Test phase
            save_file=False)

training_input = torch.tensor(train_dict['train_spikes_heldin'], device=device, dtype=torch.float32)
eval_input = torch.tensor(eval_dict['eval_spikes_heldin'], device=device, dtype=torch.float32)


training_input = torch.tensor(
    np.concatenate([
        train_dict['train_spikes_heldin'], 
        np.zeros(train_dict['train_spikes_heldin_forward'].shape), # zeroed inputs for forecasting
    ], axis=1), device=device, dtype=torch.float32)

training_output = torch.tensor(
        np.concatenate([
            np.concatenate([
                train_dict['train_spikes_heldin'],
                train_dict['train_spikes_heldin_forward'],
            ], axis=1),
            np.concatenate([
                train_dict['train_spikes_heldout'],
                train_dict['train_spikes_heldout_forward'],
            ], axis=1),
        ], axis=2), dtype=torch.float32)



# eval_input = torch.tensor(
#     np.concatenate([
#         eval_dict['eval_spikes_heldin'],
#         np.zeros((
#             eval_dict['eval_spikes_heldin'].shape[0],
#             train_dict['train_spikes_heldin_forward'].shape[1],
#             eval_dict['eval_spikes_heldin'].shape[2]
#         )),
#     ], axis=1), device=device, dtype=torch.float32)

# data = dataset.nlb_data()
# loader = DataLoader(data, batch_size=512, shuffle=True)



# inp, label = next(iter(loader))
# print(f"Shape of inputs: {inp.shape}")
# print(f"Shape of labels: {label.shape}")

# in_units = inp.shape[2]
# out_units = label.shape[2]
# t_out = label.shape[1]
criterion = nn.PoissonNLLLoss(log_input=False)
vae = model.lstm_ae(hyper_param, model_param)
trainer = trainers.VAE_trainer(vae,results_dir, hyper_param)
train_loss, test_loss = trainer.train(training_input,training_output , num_epochs, criterion, save_checkpoint=True)


print("Successful run...!")


