import numpy as np
import pandas as pd
import torch.nn as nn


from datasets import LiverDEC
from torchvision import models
from torch.utils.data import DataLoader
from helpers import knn_eval, xai_eval, map_eval
from Quantus import PointingGame, TopKIntersection, RelevanceRankAccuracy

save_path = '/storage/results/LiverDec/'

M, dummy_label = 10, [0]
runs = [f"run{m}" for m in range(M)]

encoders = ['imagenet']
for run in runs:
    encoders.append(f"random ({run})")

metric_names = ['pointing game', 'top k', 'relevance rank']

metrics = [PointingGame(), TopKIntersection(), RelevanceRankAccuracy()]

k_vals = [5]

for k_val in k_vals:
    metric_names.append(f"knn accuracy ({k_val})")

for k_val in k_vals:
    metric_names.append(f"map score ({k_val})")

scores = pd.DataFrame(data=np.zeros((len(encoders), len(metric_names))), index=encoders, columns=metric_names)

model = models.resnet50(pretrained=True)
modules = list(model.children())[:-1]
encoder = nn.Sequential(*modules, nn.Flatten()).to('cuda')
encoder.eval()

NUM_WORKERS, BATCH_SIZE = 2, 32

tr_data = LiverDEC(train=True, no_aug=False)
tr_loader = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True,
                       drop_last=True, num_workers=NUM_WORKERS)

tr_data_no_aug = LiverDEC(train=True, no_aug=True)
tr_loader_no_aug = DataLoader(tr_data_no_aug, num_workers=4)

te_data = LiverDEC(train=False, no_aug=True)
te_loader = DataLoader(te_data, num_workers=4)

print("-"*65)
print('Calculating scores for feature extractor pretrained on imagenet.')
print("-"*65)

relax_scores = xai_eval(encoder, te_loader, metrics)

for relax_score, metric_name in zip(relax_scores, metric_names):
    scores.at['imagenet', metric_name] = relax_score

for k_val in k_vals:
    metric_name = f"knn accuracy ({k_val})"
    scores.at['imagenet', metric_name] = knn_eval(encoder, k_val, tr_loader_no_aug, te_loader)

for k_val in k_vals:
    metric_name = f"map score ({k_val})"
    scores.at['imagenet', metric_name] = map_eval(encoder, k_val, tr_loader_no_aug, te_loader)


scores.to_csv(''.join([save_path, 'imagenet', '.csv']))
"""
for run in runs:

    print(f'Calculating scores for random feature extractor ({run})')
    print("-"*65)
    metrics = [PointingGame(), TopKIntersection(), RelevanceRankAccuracy()]

    model = models.resnet50(pretrained=False)
    modules = list(model.children())[:-1]
    encoder = nn.Sequential(*modules, nn.Flatten()).to('cuda')
    encoder.eval()

    relax_scores = xai_eval(encoder, te_loader, metrics)

    for relax_score, metric_name in zip(relax_scores, metric_names) :
        scores.at[f"random ({run})", metric_name] = relax_score

    for k_val in k_vals:
        metric_name = f"knn accuracy ({k_val})"
        scores.at[f"random ({run})", metric_name] = knn_eval(encoder, k_val, tr_loader_no_aug, te_loader)

scores.to_csv(''.join([save_path, 'imagenet', '.csv']))
"""