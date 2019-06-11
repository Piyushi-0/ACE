"""model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BetaVAE_mnist_mod(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=1):
        super(BetaVAE_mnist_mod, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.temperature = 1
        # self.softmax = nn.Softmax(dim = -1)
        self.encoder = nn.Sequential(
            nn.Linear(794, 512),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Linear(512, 256),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Linear(256, 128),          # B,  64,  8,  8
            nn.ReLU(True),                # B, 256
            nn.Linear(128, z_dim*2),             # B, z_dim*2
        )
        self.last_layer=nn.Sequential(nn.Linear(z_dim+10, 10))#####################
	#self.last_layer=nn.Linear(z_dim+10, 10)        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim+10, 128),               # B, 256
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.Linear(512, 784), # B,  64,  8,  8
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, y):
        distributions = self._encode(x, y)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        #print('z length'+str(len(z)))
        logits =self._classify(z, y)#self.last_layer(z2)##################
	x_recon = self._decode(z, y)

        return x_recon, mu, logvar, logits################

    def _encode(self, x, y):
        return self.encoder(torch.cat([x, y], 1))

    def _decode(self, z, y):
        return self.decoder(torch.cat([z, y], 1))  
        
    def _classify(self, z, y):
        z2=torch.cat([z, y], 1)
    	return self.last_layer(z2)     

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    pass
