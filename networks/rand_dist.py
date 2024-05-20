import torch
import numpy as np
from copy import deepcopy

# Utility file to seed rngs
def seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)


# A highly simplified convenience class for sampling from distributions
# One could also use PyTorch's inbuilt distributions package.
# Note that this class requires initialization to proceed as
# x = Distribution(torch.randn(size))
# x.init_distribution(dist_type, **dist_kwargs)
# x = x.to(device,dtype)
# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)
  
    def __init__(self, *args, **kwargs):
        super().__init__()


    def init_distribution(self, dist_type, **kwargs):
        seed_rng(kwargs.get('seed', 42))  # Default seed if not provided
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif dist_type == 'uniform':
            self.low, self.high = kwargs['low'], kwargs['high']
        elif dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']
        elif dist_type == 'poisson':
            self.lam = kwargs['var']
        elif dist_type == 'gamma':
            self.scale = kwargs['var']


    def sample_(self):
        new_instance = self.new_empty(self.size())
        if self.dist_type == 'normal':
            new_instance.normal_(self.mean, self.var)
        elif self.dist_type == 'uniform':
            new_instance.uniform_(self.low, self.high)
        elif self.dist_type == 'categorical':
            new_instance.random_(0, self.num_categories)
        elif self.dist_type == 'poisson':
            data = np.random.poisson(self.lam, self.size())
            new_instance = torch.from_numpy(data).type(self.type()).to(self.device)
        elif self.dist_type == 'gamma':
            data = np.random.gamma(shape=1, scale=self.scale, size=self.size())
            new_instance = torch.from_numpy(data).type(self.type()).to(self.device)
        # Ensure the new instance has the same distribution settings
        new_instance.init_distribution(self.dist_type, **self.dist_kwargs)
        return new_instance.detach()


    def to(self, *args, **kwargs):
        new_obj = super().to(*args, **kwargs)
        # Ensure new_obj is wrapped in Distribution and has the same distribution properties
        if not isinstance(new_obj, Distribution):
            # This assumes there's a way to directly create a Distribution instance from a Tensor.
            # Might need a custom method or modification to __new__ or __init__ to support this.
            new_obj = Distribution(new_obj)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        return new_obj

# Convenience function to prepare a z vector
def prepare_z_dist(G_batch_size, dim_z, device='cuda', seed=0):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=1.0, seed=seed)
    z_ = z_.to(device)
    return z_

# Convenience function to prepare a z vector
def prepare_y_dist(G_batch_size, nclasses, device='cuda', seed=0):
    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses, seed=seed)
    y_ = y_.to(device, torch.int64)
    return y_
