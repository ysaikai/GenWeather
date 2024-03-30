# %%
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

EPS = torch.tensor(1e-3) # for stability
softplus = nn.Softplus()

"""
The output is a list of length 4. Each element consists of (p1,p2):
  - for Radn ~ gamma(p1,p2)
  - for MinT ~ N(p1,p2)
  - for MaxT-MinT ~ gamma(p1,p2)
  - for Rain ~ gamma(p1,p2)
in this order.
(For gamma, assumed PyTorch Gamma, specified by shape & rate. If NumPy, it's
specified by shape & scale and the likelihood must be modified accordingly.)

The network process input x as follows.
  - Masked conv (usign a channel-specific 2-D filter)
  - Dilated convs
  - 1x1 convs
The masked conv filter is made small to emphasise the masking effect.

Given size_filter and n_dilate, n_receptive = size_filter**n_dilate.

n_padding=n_predic-1, which is equal to the number of times a filter of size
n_receptive is slided so that prediction for the first time slot is based
only on the conditioning part and prediction for the last time slot is based
on no padded 0.

After the conv of w=size_filter and d=dilation factor, the output size
decreases by (w-1)*d=((w-1)*d+1)-1, where (w-1)*d+1 is the effective
filter size. For example, in the training,
  len(x)=2401=n_rcpt
  n_padding = 364
  size_filter=7
  n_dilate=4,
gives the output of size 2401+364-6*(1+7+49+343)=365=n_pred. In the testing,
  len(x)=2037=n_rcpt-n_pred+1
  n_padding = 364
  size_filter=7
  n_dilate=4,
gives the output of size 2037+364-6*(1+7+49+343)=1. Then, we repeatedly
append the recent output to the input and reduce a padded 0 until we get
n_pred number of outputs.
"""
class Net(nn.Module):
  def __init__(self, n_pred=365, height=4, device_ids=[0]):
    super().__init__()
    self.device_ids = device_ids
    if n_pred < 800: # a rough threshold
      size_filter = 7
    else:
      size_filter = 5
    size_mask = 2
    self.size_mask = size_mask
    """
    By design, 1st element of in_channels is 1. The elements of out_channels
    are equal to the entries of in_channels shifted by 1 and the element is
    2 by design (i.e. 2 parameters for all conditional distributions).
    For example, `self.in_channels = [1] + [8,16,32,16] + [2]` implies
      - 1 for the masked 2D conv
      - [8,16,32,16] for 1D dilated conv with an increasing dilation
      - 2 for 1x1 conv
    n.b. in- & out-channel for 1x1 need not match.
    """
    if size_filter==7:
      self.in_channels = [1] + [8,16,32,64] + [64,32,16,8]
      self.n_dilate = 4
    elif size_filter==5:
      self.in_channels = [1] + [8,8,16,32,64] + [64,32,16,8]
      self.n_dilate = 5
    self.out_channels = [_ for _ in self.in_channels[1:]] + [2]
    self.height = height
    self.n_pred = n_pred
    self.n_rcpt = size_filter**self.n_dilate # |receptive field|
    self.n_cond = self.n_rcpt - self.n_pred + 1

    # Masked convolution (without bias)
    self.masked = nn.Conv2d(self.in_channels[0],
                            self.out_channels[0],
                            (height,size_mask), bias=False)
    params_mask = []
    for _ in range(height):
      params_mask.append(nn.Parameter(deepcopy(self.masked.weight)))
    del self.masked.weight
    self.params_mask = nn.ParameterList(params_mask)
    # """DEBUG
    # Used to check the masked part is never updated and remains 0
    # throughout the training.
    # """
    for i in range(height):
      with torch.no_grad():
        self.params_mask[i][:,:,i:,-1] = 0

    modules_mid = []

    # Causal dilated convolutions
    """
    First, as usual, conv of a filter of size l shrinks the output length
    by l-1, which is reflected by `size_mask-1` in `size_mask-1-1`. Plus,
    as noted in train.py, the mask conv requires len(x)=n_rcpt+1 which is
    reflected by the last `-1` in `size_mask-1-1`.
    """
    modules_mid.append(nn.ZeroPad1d(((self.n_pred-1)+(size_mask-1-1), 0)))
    for i in range(self.n_dilate):
      modules_mid.append(nn.Conv1d(self.in_channels[i+1],
                                   self.out_channels[i+1],
                                   size_filter,
                                   dilation=size_filter**i))
      modules_mid.append(nn.Tanh())

    # 1x1 convolutions
    modules_mid.append(nn.Conv1d(self.in_channels[-4],
                                 self.out_channels[-4], 1))
    modules_mid.append(nn.ReLU())
    modules_mid.append(nn.Conv1d(self.in_channels[-3],
                                 self.out_channels[-3], 1))
    modules_mid.append(nn.ReLU())
    modules_mid.append(nn.Conv1d(self.in_channels[-2],
                                 self.out_channels[-2], 1))
    modules_mid.append(nn.ReLU())
    modules_mid.append(nn.Conv1d(self.in_channels[-1],
                                 self.out_channels[-1], 1))
    self.modules_mid = nn.ModuleList(modules_mid)


  def forward(self, x, channel=0):
    if self.training:
      y = []

      # Masked convolution
      for i in range(self.height):
        """Masks
        Masks with an increasing number of 1 at the last column

        Creating masks at this point is handy because use of
        torch.ones_like(self.modules_mask[i].weight[0,0,:,:]) copies
        not only the shape but also its memory location (on CPU or GPU)
        so that we need not take care of `mask.to(device)`.
        """
        mask = torch.ones_like(self.params_mask[i][0,0,:,:])
        mask[i:,-1] = 0
        self.masked.weight = self.params_mask[i]*mask
        y.append(self.masked(x).squeeze(2))

      # Causal dilated & first 1x1 convolutions
      for f in self.modules_mid:
        for i in range(self.height):
          y[i] = f(y[i])
      
      # Swap the last two dims for ease of loss_fn computation
      for i in range(self.height):
        y[i] = y[i].transpose(-1,-2)
      
      # Positivity constraint on p2 for Gaussian
      y[1] = torch.cat((y[1][:,:,:1],
                        torch.maximum(EPS,softplus(y[1][:,:,1:]))), 2)
      # Positivity constraint on p1 & p2 for gamma
      for i in [j for j in range(self.height) if j != 1]:
        y[i] = torch.maximum(EPS,softplus(y[i]))

      return y
    else:
      i = channel
      
      # Masked convolution
      mask = torch.ones_like(self.params_mask[i][0,0,:,:])
      mask[i:,-1] = 0
      self.masked.weight = self.params_mask[i]*mask
      y = self.masked(x).squeeze(2)

      # Causal dilated & first 1x1 convolutions (the padding omitted)
      for f in self.modules_mid:
        if isinstance(f, nn.ZeroPad1d)==False:
          y = f(y)
      
      # Swap the last two dims for ease of loss_fn computation
      y = y.transpose(-1,-2)
      
      # Positivity constraint for all but the mean of Gaussian
      if i == 1:
        y = torch.cat((y[:,:,:1],
                       torch.maximum(EPS,softplus(y[:,:,1:]))), 2)
      else:
        y = torch.maximum(EPS,softplus(y))

      return y


# %%
