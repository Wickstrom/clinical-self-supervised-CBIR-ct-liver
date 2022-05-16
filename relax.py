import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Resize

class RELAX(nn.Module):
    def __init__(self, x, f, num_batches=30, batch_size=100):
        super().__init__()

        self.device = x.device
        self.batch_size = batch_size
        self.shape = tuple(x.shape[2:])
        self.num_batches = num_batches
        self.pdist = nn.CosineSimilarity(dim=1)
        
        self.upsample = Resize(self.shape, antialias=True)

        self.x = x
        self.encoder = f
        self.h_star = f(x).expand(batch_size, -1)

        self.R = torch.zeros(self.shape, device=self.device)
        self.U = torch.zeros(self.shape, device=self.device)

        self.sum_of_weights = (1e-10)*torch.ones(self.shape, device=self.device)

        self.data_shape = torch.Size([self.batch_size, 3, self.shape[0], self.shape[1]])
        self.data_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda')[None, :, None, None]
        self.data_std = torch.tensor([0.229, 0.224, 0.225], device='cuda')[None, :, None, None]

    def forward(self, fill_vals_zeros=True):

        for batch in range(self.num_batches):
            for masks in self.mask_generator():
                
                if fill_vals_zeros:
                    x_mask = self.x * masks
                else:
                    noise = torch.randn(self.data_shape, device='cuda')*self.data_std+self.data_mean
                    x_mask = self.x * masks + (1-masks)*noise

                h = self.encoder(x_mask)
                sims = self.pdist(self.h_star, h)

                for si, mi in zip(sims, masks.squeeze()):

                    self.sum_of_weights += mi

                    R_prev = self.R.clone()
                    self.R = self.R + mi*(si-self.R) / self.sum_of_weights
                    self.U = self.U + (si-self.R) * (si-R_prev) * mi

        return None

    def importance(self):
        return self.R

    def uncertainty(self):
        return self.U / (self.sum_of_weights - 1)

    def U_RELAX(self, gamma=1.0, average='mean'):
        if average == 'mean':
          return self.importance()*(self.uncertainty() < gamma*self.uncertainty().mean())
        elif average == 'median':
          return self.importance()*(self.uncertainty() < gamma*self.uncertainty().median())
        else:
          raise NotImplementedError("Only mean and median implemented.")

    def mask_generator(self, num_cells=7, p=0.5, nsd=2):

        pad_size = (num_cells // 2, num_cells // 2, num_cells // 2, num_cells // 2)

        grid = (torch.rand(self.batch_size, 1, *((num_cells,) * nsd), device=self.device) < p).float()
        #grid_up = F.interpolate(grid, size=(self.shape), mode='bilinear', align_corners=False)
        grid_up = self.upsample(grid)
        grid_up = F.pad(grid_up, pad_size, mode='reflect')

        shift_x = torch.randint(0, num_cells, (self.batch_size,), device=self.device)
        shift_y = torch.randint(0, num_cells, (self.batch_size,), device=self.device)

        masks = torch.empty((self.batch_size, 1, self.shape[-2], self.shape[-1]), device=self.device)

        for bi in range(self.batch_size):
            masks[bi] = grid_up[bi, :,
                                shift_x[bi]:shift_x[bi] + self.shape[-2],
                                shift_y[bi]:shift_y[bi] + self.shape[-1]]

        yield masks
