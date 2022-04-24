import torch


def VAE_loss_fn(out, true):
    pred, mu, log_var = out
    # Regularization Loss...!
    kld = -0.5*torch.sum(1 + log_var - mu**2 - log_var.exp())
    # Reconstruction Loss...!
    bce = torch.nn.functional.binary_cross_entropy(pred, true, reduction='sum')

    return kld + bce