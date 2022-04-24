import os
import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import fnmatch
import time
#import pickle
#
import utils

class VAE_trainer():
    def __init__(self, model, results_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0)
        self.results_dir = results_dir
        self.cd_ratio = 0.27
        print(f"Trainer object created with model: {self.model.name}")

    def train(self, train_data, num_epochs, criterion, test_data=None, load_weights = False, save_checkpoint=True): 
        print(f"Training for {num_epochs} epoch has started...!") 
        prev_epochs = 0
        if load_weights:
            prev_epochs = self.load_checkpoint()  
            if not os.path.exists(os.path.join(self.results_dir, "results.txt")):
                with open(os.path.join(self.results_dir, "results.txt"), 'w') as f:
                    f.write(f"Epoch, Exe_time, Loss \n")  
        else:
            prev_epochs = 0
            with open(os.path.join(self.results_dir, "results.txt"), 'w') as f:
                    f.write(f"Epoch, Exe_time, Loss \n")      
        loss_history = []
        test_loss_history = []
        for i in range(1, num_epochs+1):
            epoch = prev_epochs + i
            start_time = time.time()
            self.model.train()
            error = []
            count=0
            for train_input,train_output in train_data:
                train_input = train_input.to(self.device)
                train_output = train_output.to(self.device)

                input_mask, output_mask = self.make_mask(train_input, train_output)
                masked_train_input = train_input.clone()
                masked_train_input[input_mask] = 0.0 
                      
                count+=1
                
                self.opt.zero_grad()
                train_predictions = self.model(masked_train_input)
                #loss = utils.VAE_loss_fn(out, x)
                # loss = criterion(out, y)
                loss = torch.nn.functional.poisson_nll_loss(train_predictions[output_mask], train_output[output_mask], log_input=False)
                loss.backward()
                self.opt.step()

                error.append(loss.item())
            epoch_loss = sum(error)/count
            
            loss_history.append(epoch_loss)
            if test_data != None:
                test_loss_history.append(self.test(test_data))
                epoch_loss = test_loss_history[-1]
            end_time = time.time()
            exe_time = (end_time-start_time)/60 
            with open(os.path.join(self.results_dir, "results.txt"), 'a') as f:
                f.write(f"{epoch}, {exe_time:.2f}, {epoch_loss:.3f} \n")
            
            if save_checkpoint and epoch %100==0: 
                self.save_checkpoint(epoch)

        print(f"Model trained for {num_epochs} epochs")
        return loss_history, test_loss_history
        
    def test(self, test_data, criterion):  
        self.model.eval()
        error = []
        count=0
        for x,y in test_data:
            x = x.to(self.device)
            y = y.to(self.device)        
            count+=1
            
            with torch.no_grad():
                out = self.model(x)
            #loss = utils.VAE_loss_fn(out, x)
            loss = criterion(out, y)    
            error.append(loss.item())
        loss = sum(error)/count

        return loss

    def save_checkpoint(self, epoch):
        checkpoint = {'epoch': epoch, 'model_state': self.model.state_dict(),
         'opt_state': self.opt.state_dict()}
        torch.save(checkpoint, os.path.join(self.results_dir, f"{self.model.name}_checkpoint_epochs_{epoch:03d}.pt")) 
        print(f"Checkpoint saved with epochs:{epoch}")       

    def load_checkpoint(self, filename=None):
        if filename==None:
            files = []
            for f in os.listdir(self.results_dir):
                if fnmatch.fnmatch(f, f"{self.model.name}_checkpoint*.pt"):
                    files.append(f)
            files.sort()
            chkpoint = files[-1]
            filename = os.path.join(self.results_dir, chkpoint)
        
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state'])
        self.opt.load_state_dict(checkpoint['opt_state'])

        return checkpoint['epoch']

    def make_mask(self, train_input, train_output):
        cd_ratio = self.cd_ratio
        input_mask = torch.zeros((train_input.shape[0] * train_input.shape[1] * train_input.shape[2]), dtype=torch.bool)
        idxs = torch.randperm(input_mask.shape[0])[:int(round(cd_ratio * input_mask.shape[0]))]
        input_mask[idxs] = True
        input_mask = input_mask.view((train_input.shape[0], train_input.shape[1], train_input.shape[2]))
        output_mask = torch.ones(train_output.shape, dtype=torch.bool)
        output_mask[:, :, :input_mask.shape[2]] = input_mask
        return input_mask, output_mask

    def score(self, input, output, prefix='val'):
        """Evaluates model performance on given data"""
        self.model.eval()
        predictions = self.model(input)
        self.model.train()
        loss = torch.nn.functional.poisson_nll_loss(predictions, output, log_input=False)
        num_heldout = output.shape[2] - input.shape[2]
        cosmooth_loss = torch.nn.functional.poisson_nll_loss(
            predictions[:, :, -num_heldout:], output[:, :, -num_heldout:], log_input=False)
        return {f'{prefix}_nll': loss.item(), f'{prefix}_cosmooth_nll': cosmooth_loss.item()}, predictions  
        
