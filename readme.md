### Using Variational Autoencoders to generate Images and Neural Data
- Trained (Convolution) VAE on 'CelebA' dataset
    - worked out the 'loss' function with the help of class notes
    - implemented 'model.CAE(.)' and 'trainers.VAE_trainer(..)' classes (refered to online available implementations for some help but written my own code)
    - Use trained model to reconstruct images and randomly generate images
- NLB challenge:
    - Trained 'model.lstm_ae(..)' on 3 datasets namely 'MC_MAZE_LARGE', 'MC_MAZE_MEDIUM' and 'MC_MAZE_SMALL'
    - Refered to official documentation of NLB challenge and used their data loading/processing pipeline.
    - Implemented model on my own, tried 2 variations:
        - one where 'latent space' uses only the last time step of encoded features
        - second wher 'latent space' used all time steps of the encoded features.
    - Optimized hyper-parameters (latent_dim's, learning_rate, weight_decay, number of layers) 
        - was able to get good (decreasing) training loss for all 3 datasets after trying some hyper parameters
    - For evaluation, I uploaded results on the 'evalai' website and the results were not very impressive and did not improve beyond a certain point even with different combinations of hyper-parameters.
 

###Task 1:
Training VAE on CelebA dataset for generating images:
This notebook can be used to go through the 'test.ipynb', which has step-by-step instructions on how to load pre-trained model and generate random images.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bilalhsp/VAE/blob/main/test.ipynb)

Note: Reonstruction part needs to load 'CelebA' data but the link provided by pytorch is not reliable and often fails to load data. I manually downloaded data from some other source and then used that for training. The dataset is huge and cannot be made part of the submission. You can try to load data but this check may cause errors. Generating random images does not need any data (only pretrained weights) so it works anyway.


###Task 2:
Training LSTM_Autoencoder for Neural Latent Benchmarking:
This notebook allows to use pretrained weights (separately) for 3 datasets.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bilalhsp/VAE/blob/main/nlb_data.ipynb)
 It will create a submission file that can be uploaded to 'evalai' using the script provided at the end of the notebook but my login will be needed in order to look at results of the submission.
