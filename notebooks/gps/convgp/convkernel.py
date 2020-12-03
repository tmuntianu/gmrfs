import torch, gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval
from torch.nn import Parameter
from torch.nn.functional import unfold
import warnings

class ConvKernel(Kernel):
    '''
    Computes a covariance matrix from image patches, based on a base kernel
    Includes weights for each patch, as described in van der Wilk's paper https://arxiv.org/abs/1709.01894

    Implementation largely based on the GPFlow implementation, with obvious
    differences due to torch vs tf

    DOES NOT USE THE GPyTorch NATIVE RBF KERNEL! that kernel takes too much memory,
    instead here we implement the RBF ourselves

    see gpytorch_conv.py for the native RBF kernel implementation

    args:
        base_kernel: an instantiated nn.Module that contains the base kernel to use
        image_shape: a tuple or list of the image shape being passed
        patch_shape: a tuple or list of the patch shape being used
        color_channels: the number of channels in the images
        kwargs: args to be passed to the super constructor
    '''

    def __init__(self, image_shape, patch_shape, color_channels=1, **kwargs):
        super(ConvKernel, self).__init__(**kwargs)
        # self.base_kernel = base_kernel
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.color_channels = color_channels

        self.register_parameter(name='rbf_lengthscale', parameter=Parameter(torch.tensor(1.0)))
        self.register_parameter(
            name='patch_weights',
            parameter=Parameter(torch.ones(self.get_num_patches()))
        )
        self.register_constraint('patch_weights', Interval(-1,1))

    def get_num_patches(self):
        return (
            (self.image_shape[0] - self.patch_shape[0] + 1)
            * (self.image_shape[1] - self.patch_shape[1] + 1)
            * self.color_channels
        )

    def sum_broadcasted(self, x1, x2):
        flat = torch.add(x1.reshape(-1,1), x2.reshape(1,-1))
        return flat.reshape(torch.Size(torch.cat([torch.tensor(x1.shape), torch.tensor(x2.shape)],0)))


    def rbf(self, x1, x2):
        if x2 is None:
            x1s = torch.sum(torch.square(x1),-1,keepdim=True)
            dist = -2 * torch.matmul(x1, x1.permute(0,2,1))
            dist += x1s + torch.conj(x1s.permute(0,2,1))
            dist /= torch.square(self.rbf_lengthscale)
            covar = torch.exp(-0.5 * dist)
            return covar
        x1s = torch.sum(torch.square(x1), -1)
        x2s = torch.sum(torch.square(x2), -1)
        dist = -2 * torch.tensordot(x1, x2, ([-1], [-1]))
        dist += self.sum_broadcasted(x1s, x2s)

        dist /= torch.square(self.rbf_lengthscale)
        covar = torch.exp(-0.5 * dist)
        return covar

    def get_patches(self, X):
        Xp = unfold(X, self.patch_shape)
        Xp = Xp.permute(0,2,1)
        Xp = Xp.reshape([Xp.shape[0],Xp.shape[1]*self.color_channels,-1])
        return Xp

    def forward(self, x1, x2=None, **kwargs):
        x1p = self.get_patches(x1)
        x2p = x2 if x2 is None else self.get_patches(x2)

        # K = self.base_kernel(x1p, x2p, **kwargs)
        K = self.rbf(x1p, x2p)
        w = self.patch_weights[:,None] * self.patch_weights[None,:]
        Kw = K.mul(w[None,:,None,:])
        return torch.sum(Kw, (1,3)).mul(self.get_num_patches() ** -2.0)

# with gpytorch.settings.lazily_evaluate_kernels(False):
#     a = torch.randn(5,3,32,32).cuda()
#     b = torch.randn(5,3,32,32).cuda()
#     covar = ConvKernel((32,32),(10,10),color_channels=3).cuda()
#     c = covar(a,b).numpy()
#     print(c.shape)
