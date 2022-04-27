import os
import torch
from nlb_tools.evaluation import evaluate
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors
from nlb_tools.make_tensors import make_eval_target_tensors
from nlb_tools.make_tensors import save_to_h5

import numpy as np
import model
import trainers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################
# Results for MC_Maze_Large
############################
# Loading the dataset...!
print("-------Large Dataset---------")
print("Loading the dataset...!")
curr_path = os.getcwd()
fpath = curr_path + '/000138/sub-Jenkins/'
dataset = NWBDataset(fpath=fpath)
dataset.resample(5)
train_dict = make_train_input_tensors(dataset=dataset, 
            dataset_name='mc_maze_large', 
            # trial_split='train', 
            trial_split=['train', 'val'], #for Test phase
            save_file=False, 
            include_forward_pred=True)

eval_dict = make_eval_input_tensors(dataset=dataset,
            dataset_name='mc_maze_large',
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

eval_input = torch.tensor(
    np.concatenate([
        eval_dict['eval_spikes_heldin'],
        np.zeros((
            eval_dict['eval_spikes_heldin'].shape[0],
            train_dict['train_spikes_heldin_forward'].shape[1],
            eval_dict['eval_spikes_heldin'].shape[2]
        )),
    ], axis=1), device=device, dtype=torch.float32)

# training_output = torch.tensor(
#     np.concatenate([
#         np.concatenate([
#             train_dict['train_spikes_heldin'],
#             train_dict['train_spikes_heldin_forward'],
#         ], axis=1),
#         np.concatenate([
#             train_dict['train_spikes_heldout'],
#             train_dict['train_spikes_heldout_forward'],
#         ], axis=1),
#     ], axis=2), device=device, dtype=torch.float32)




# print("Shapes of input and output")
# print(training_input.shape)
# print(training_output.shape)
# print("Eval input shape")
# print(eval_input.shape)

#Loading the pre-trained model...!
results_dir = '/scratch/gilbreth/ahmedb/vae/lstm_saved_results/large'
vae = model.lstm_ae(122, 162, 180, drop=0.25)
trainer = trainers.VAE_trainer(vae,results_dir)

epoch = trainer.load_checkpoint()
print(f"Loaded model was trained for {epoch} epochs")

# Get predictions from the model...!
trainer.model.eval()
training_predictions_large = trainer.model.predict(training_input).cpu().detach().numpy()
eval_predictions_large = trainer.model.predict(eval_input).cpu().detach().numpy()

#Prepare the submission dict...!
tlen_large = train_dict['train_spikes_heldin'].shape[1]
num_heldin_large = train_dict['train_spikes_heldin'].shape[2]

########################
# Results for MC_Maze_Medium
############################
# Loading the dataset...!
print("-------Medium Dataset---------")
print("Loading the dataset...!")
curr_path = os.getcwd()
fpath = curr_path + '/000139/sub-Jenkins/'
dataset = NWBDataset(fpath=fpath)
dataset.resample(5)
train_dict = make_train_input_tensors(dataset=dataset, 
            dataset_name='mc_maze_medium', 
            # trial_split='train', 
            trial_split=['train', 'val'], #for Test phase
            save_file=False, 
            include_forward_pred=True)

eval_dict = make_eval_input_tensors(dataset=dataset,
            dataset_name='mc_maze_medium',
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

eval_input = torch.tensor(
    np.concatenate([
        eval_dict['eval_spikes_heldin'],
        np.zeros((
            eval_dict['eval_spikes_heldin'].shape[0],
            train_dict['train_spikes_heldin_forward'].shape[1],
            eval_dict['eval_spikes_heldin'].shape[2]
        )),
    ], axis=1), device=device, dtype=torch.float32)



#Loading the pre-trained model...!
results_dir = '/scratch/gilbreth/ahmedb/vae/lstm_saved_results/medium'
vae = model.lstm_ae(114, 152, 180, drop=0.25)
trainer = trainers.VAE_trainer(vae,results_dir)

epoch = trainer.load_checkpoint()
print(f"Loaded model was trained for {epoch} epochs")

# Get predictions from the model...!
trainer.model.eval()
training_predictions_medium = trainer.model.predict(training_input).cpu().detach().numpy()
eval_predictions_medium = trainer.model.predict(eval_input).cpu().detach().numpy()

#Prepare the submission dict...!
tlen_medium = train_dict['train_spikes_heldin'].shape[1]
num_heldin_medium = train_dict['train_spikes_heldin'].shape[2]



########################
# Results for MC_Maze_Small
############################
# Loading the dataset...!
print("-------Small Dataset---------")

print("Loading the dataset...!")
curr_path = os.getcwd()
fpath = curr_path + '/000140/sub-Jenkins/'
dataset = NWBDataset(fpath=fpath)
dataset.resample(5)
train_dict = make_train_input_tensors(dataset=dataset, 
            dataset_name='mc_maze_small', 
            # trial_split='train', 
            trial_split=['train', 'val'], #for Test phase
            save_file=False, 
            include_forward_pred=True)

eval_dict = make_eval_input_tensors(dataset=dataset,
            dataset_name='mc_maze_small',
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

eval_input = torch.tensor(
    np.concatenate([
        eval_dict['eval_spikes_heldin'],
        np.zeros((
            eval_dict['eval_spikes_heldin'].shape[0],
            train_dict['train_spikes_heldin_forward'].shape[1],
            eval_dict['eval_spikes_heldin'].shape[2]
        )),
    ], axis=1), device=device, dtype=torch.float32)


#Loading the pre-trained model...!
results_dir = '/scratch/gilbreth/ahmedb/vae/lstm_saved_results/small'
vae = model.lstm_ae(107, 142, 180, drop=0.25)
trainer = trainers.VAE_trainer(vae,results_dir)

epoch = trainer.load_checkpoint()
print(f"Loaded model was trained for {epoch} epochs")

# Get predictions from the model...!
trainer.model.eval()
training_predictions_small = trainer.model.predict(training_input).cpu().detach().numpy()
eval_predictions_small = trainer.model.predict(eval_input).cpu().detach().numpy()

#Prepare the submission dict...!
tlen_small = train_dict['train_spikes_heldin'].shape[1]
num_heldin_small = train_dict['train_spikes_heldin'].shape[2]



# Combining results...!
print("Combining all results...!")

submission = {
    'mc_maze_large': {
        'train_rates_heldin': training_predictions_large[:, :tlen_large, :num_heldin_large],
        'train_rates_heldout': training_predictions_large[:, :tlen_large, num_heldin_large:],
        'eval_rates_heldin': eval_predictions_large[:, :tlen_large, :num_heldin_large],
        'eval_rates_heldout': eval_predictions_large[:, :tlen_large, num_heldin_large:],
        'eval_rates_heldin_forward': eval_predictions_large[:, tlen_large:, :num_heldin_large],
        'eval_rates_heldout_forward': eval_predictions_large[:, tlen_large:, num_heldin_large:]
    },
    'mc_maze_medium': {
        'train_rates_heldin': training_predictions_medium[:, :tlen_medium, :num_heldin_medium],
        'train_rates_heldout': training_predictions_medium[:, :tlen_medium, num_heldin_medium:],
        'eval_rates_heldin': eval_predictions_medium[:, :tlen_medium, :num_heldin_medium],
        'eval_rates_heldout': eval_predictions_medium[:, :tlen_medium, num_heldin_medium:],
        'eval_rates_heldin_forward': eval_predictions_medium[:, tlen_medium:, :num_heldin_medium],
        'eval_rates_heldout_forward': eval_predictions_medium[:, tlen_medium:, num_heldin_medium:]
    },
    'mc_maze_small': {
        'train_rates_heldin': training_predictions_small[:, :tlen_small, :num_heldin_small],
        'train_rates_heldout': training_predictions_small[:, :tlen_small, num_heldin_small:],
        'eval_rates_heldin': eval_predictions_small[:, :tlen_small, :num_heldin_small],
        'eval_rates_heldout': eval_predictions_small[:, :tlen_small, num_heldin_small:],
        'eval_rates_heldin_forward': eval_predictions_small[:, tlen_small:, :num_heldin_small],
        'eval_rates_heldout_forward': eval_predictions_small[:, tlen_small:, num_heldin_small:]
    }
}


# Saving the submission results... use 'putput.out'
print("Saving the submission file...!")
save_to_h5(submission, './submissions/submission.h5')
print("Done...!")


# print(f"Submission dict ready, loading target dict...!")
# #Evaluating the predictions...!
# target_dict = make_eval_target_tensors(dataset=dataset, 
#                                        dataset_name='mc_maze',
#                                        train_trial_split='train',
#                                        eval_trial_split='val',
#                                        include_psth=True,
#                                        save_file=False)

# print("Evaluating scores...!")
# evaluate(target_dict, submission)