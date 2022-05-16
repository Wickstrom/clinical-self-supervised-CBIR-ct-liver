import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


from PIL import Image
from relax import RELAX
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from torchvision.transforms.functional import pil_to_tensor


def to_np(x):
    return x.cpu().detach().numpy()


def upsample(x, up_shape):
    return to_np(F.interpolate(input=x, size=(up_shape), mode='bilinear'))


@torch.no_grad()
def map_eval(encoder, k_val, train_loader, test_loader):

    for idx, (x_tr_i, _, y_tr_i, _) in enumerate(train_loader):
        if idx == 0:
          e_tr = to_np(encoder(x_tr_i.to('cuda')))
          y_tr = to_np(y_tr_i)
        else:
          e_tr = np.concatenate((e_tr, to_np(encoder(x_tr_i.to('cuda')))))
          y_tr = np.concatenate((y_tr, y_tr_i))

    for idx, (x_te_i, _, y_te_i, _) in enumerate(test_loader):
        if idx == 0:
          e_te = to_np(encoder(x_te_i.to('cuda')))
          y_te = to_np(y_te_i)
        else:
          e_te = np.concatenate((e_te, to_np(encoder(x_te_i.to('cuda')))))
          y_te = np.concatenate((y_te, y_te_i))

    d_mat = cdist(e_te, e_tr)

    scores = []
    for idx, d_mat_i in enumerate(d_mat):
      temp_scores = []
      for k in range(1, k_val):
        idx_closest_samples = d_mat_i.argsort()[:k]
        temp_scores.append(np.sum(y_te[idx] == y_tr[idx_closest_samples]) / k)
      scores.append(np.mean(temp_scores))

    return np.mean(scores)


@torch.no_grad()
def knn_eval(encoder, k_val, train_loader, test_loader):

    knn_clf = KNeighborsClassifier(n_neighbors=k_val)

    for idx, (x_tr_i, _, y_tr_i, _) in enumerate(train_loader):
        if idx == 0:
          e_tr = to_np(encoder(x_tr_i.to('cuda')))
          y_tr = to_np(y_tr_i)
        else:
          e_tr = np.concatenate((e_tr, to_np(encoder(x_tr_i.to('cuda')))))
          y_tr = np.concatenate((y_tr, y_tr_i))

    for idx, (x_te_i, _, y_te_i, _) in enumerate(test_loader):
        if idx == 0:
          e_te = to_np(encoder(x_te_i.to('cuda')))
          y_te = to_np(y_te_i)
        else:
          e_te = np.concatenate((e_te, to_np(encoder(x_te_i.to('cuda')))))
          y_te = np.concatenate((y_te, y_te_i))

    knn_clf.fit(e_tr, y_tr)

    return knn_clf.score(e_te, y_te)


@torch.no_grad()
def xai_eval(encoder, data_loader, metrics):

    for x_i, _, y_i, y_mask_i in data_loader:
          if y_i == 0: continue

          up_size = y_mask_i.squeeze().shape
          y_mask_i = np.clip(to_np(y_mask_i), 0, 1)

          relax = RELAX(x_i.to('cuda'), encoder, 30, 100)
          relax.forward()

          for metric in metrics:
              _ = metric(_, upsample(x_i, up_size), [y_i.item()],
                            upsample(relax.importance()[None, None, :, :], up_size),
                            y_mask_i[None, None, :, :])


    return [np.mean(metric.all_results) for metric in metrics]



def train_supervised(train_loader, model, criterion, optimizer):

    loss_list = []
    # switch to train mode
    model.train()

    for x_tr, _, y_tr, _ in train_loader:
        x_tr = x_tr.cuda()
        y_tr = y_tr.cuda()

        # compute output and loss
        out = model(x_tr)
        loss = criterion(out, y_tr)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    return np.mean(loss_list)


def train_simsiam(train_loader, model, criterion, optimizer):

    loss_list = []
    # switch to train mode
    model.train()

    for x1, x2, _, _ in train_loader:
        x1 = x1.cuda(non_blocking=True)
        x2 = x2.cuda(non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1, x2)
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append((loss.item()+1)/2)


    return np.mean(loss_list)

@torch.no_grad()
def get_wandb_examples(encoder, dataset):

    n_rows = n_cols = 4
    #data_indices = [6, 52, 90, 105]  ###DEC###
    data_indices = [0, 5, 55, 77]   ###UNN##
    encoder.eval()
    fig = plt.figure(1, figsize=(12, 12))
    for idx, data_idx in enumerate(data_indices):
        x, _, _, y_mask = dataset.__getitem__(data_idx)
        x = x.unsqueeze(0).to('cuda')
        relax = RELAX(x, encoder)
        relax.forward()

        plt.subplot(n_rows, n_cols, n_cols*idx+1)
        plt.imshow(imsc(x[0]))
        plt.axis('off')
        plt.subplot(n_rows, n_cols, n_cols*idx+2)
        plt.imshow(y_mask)
        plt.axis('off')
        plt.subplot(n_rows, n_cols, n_cols*idx+3)
        plt.imshow(to_np(relax.importance()))
        plt.axis('off')
        plt.subplot(n_rows, n_cols, n_cols*idx+4)
        plt.imshow(to_np(relax.uncertainty()))
        plt.axis('off')
    plt.tight_layout()

    return fig


@torch.no_grad()
def imsc(img, *args, quiet=False, lim=None, interpolation='lanczos', **kwargs):
    r"""Rescale and displays an image represented as a img.
    The function scales the img :attr:`im` to the [0 ,1] range.
    The img is assumed to have shape :math:`3\times H\times W` (RGB)
    :math:`1\times H\times W` (grayscale).
    Args:
        img (:class:`torch.Tensor` or :class:`PIL.Image`): image.
        quiet (bool, optional): if False, do not display image.
            Default: ``False``.
        lim (list, optional): maximum and minimum intensity value for
            rescaling. Default: ``None``.
        interpolation (str, optional): The interpolation mode to use with
            :func:`matplotlib.pyplot.imshow` (e.g. ``'lanczos'`` or
            ``'nearest'``). Default: ``'lanczos'``.
    Returns:
        :class:`torch.Tensor`: Rescaled image img.
    """
    if isinstance(img, Image.Image):
        img = pil_to_tensor(img)
    handle = None
    with torch.no_grad():
        if not lim:
            lim = [img.min(), img.max()]
        img = img - lim[0]  # also makes a copy
        img.mul_(1 / (lim[1] - lim[0]))
        img = torch.clamp(img, min=0, max=1)
        if not quiet:
            bitmap = img.expand(3,*img.shape[1:]).permute(1, 2, 0).cpu().numpy()
    return bitmap
