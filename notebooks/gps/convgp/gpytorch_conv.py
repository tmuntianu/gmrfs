import torch, gpytorch
from gpytorch.kernels import Kernel, RBFKernel
from gpytorch.constraints import Interval
from gpytorch.lazy import lazify, LazyEvaluatedKernelTensor
from gpytorch.utils.memoize import cached
from torch.nn import Parameter
from torch.nn.functional import unfold
import warnings

class LazyEvaluatedConvKernelTensor(LazyEvaluatedKernelTensor):
    '''
    Overriding the _size method for LazyEvaluatedKernelTensor to correct the
    expected size for lazy evaluations of ConvKernel
    '''
    def __init__(self, x1, x2, kernel, **kwargs):
        super(LazyEvaluatedConvKernelTensor, self).__init__(x1, x2, kernel, **kwargs)
        if x1.shape[0] != x2.shape[0]:
            raise RuntimeError('x1 and x2 should have the same shape')
        self.num_img = x1.shape[0]

    @cached(name='size')
    def _size(self):
        return torch.Size([self.num_img, self.num_img])

class ConvKernel(Kernel):
    '''
    Computes a covariance matrix from image patches, based on a base kernel
    Includes weights for each patch, as described in van der Wilk's paper https://arxiv.org/abs/1709.01894

    Implementation largely based on the GPFlow implementation, with obvious
    differences due to torch vs tf

    THIS IMPLEMENTATION USES THE NATIVE GPyTorch RBFKernel, and eats a lot of
    memory, it is recommended to use the other kernel in convkernel.py

    args:
        num_img: the number of images being passed to the kernel, in the shape
                 [N x C x W x H] where N is the # of images, C is the # of channels
        base_kernel: an instantiated nn.Module that contains the base kernel to use
        image_shape: a tuple or list of the image shape being passed
        patch_shape: a tuple or list of the patch shape being used
        color_channels: the number of channels in the images
        kwargs: args to be passed to the super constructor
    '''

    def __init__(self, image_shape, patch_shape, color_channels=1, **kwargs):
        super(ConvKernel, self).__init__(**kwargs)
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.color_channels = color_channels
        self.base_kernels = {}

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

    def extract_patches(self, X):
        Xp = unfold(X, self.patch_shape)
        Xp = Xp.permute(0,2,1)
        Xp = Xp.reshape([Xp.shape[0],Xp.shape[1]*self.color_channels,-1])
        return Xp

    def get_patches(self, X1, X2):
        X1p = self.extract_patches(X1)

        if X2 is None:
            X2p = X1p
        else:
            X2p = self.extract_patches(X2)

        X1p = X1p[:,:,None,None,:]
        X2p = X2p[None,None,:]
        return X1p, X2p

    def forward(self, x1, x2=None, **kwargs):
        x1p, x2p = self.get_patches(x1, x2)
        # x1p = x1p.cuda(); x2p = x2p.cuda()

        num_img = x1p.shape[0]
        if num_img not in self.base_kernels:
            rbf = RBFKernel(batch_shape=torch.Size([num_img, self.get_num_patches(), 1]))
            self.base_kernels[num_img] = rbf.cuda()
        else:
            rbf = self.base_kernels[num_img]

        K = rbf(x1p, x2p, **kwargs)
        K = K.squeeze(3)
        w = self.patch_weights[:,None] * self.patch_weights[None,:]
        Kw = K.mul(w[None,:,None,:])
        return Kw.sum((1,3)).mul(self.get_num_patches() ** -2.0)

with gpytorch.settings.lazily_evaluate_kernels(False):
    a = torch.randn(4,3,10,10).cuda()
    b = torch.randn(4,3,10,10).cuda()
    covar = ConvKernel((10,10),(3,3), color_channels=3).cuda()
    # c = LazyEvaluatedConvKernelTensor(a, b, covar).numpy()
    c = covar(a,b).numpy()
    print(c.shape)
