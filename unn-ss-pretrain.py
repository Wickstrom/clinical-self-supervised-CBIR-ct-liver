import wandb
import torch
import torch.nn as nn


from simsiam import SimSiam
from torchvision import models
from datasets import LiverDEC, LiverUnn
from torch.utils.data import DataLoader
from Quantus import PointingGame, TopKIntersection, RelevanceRankAccuracy
from helpers import knn_eval, xai_eval, train_simsiam, get_wandb_examples, map_eval

NUM_RUNS, BATCH_SIZE, K_VAL, DEVICE = 5, 32, 5, 'cuda'
IN_PRETRAIN = True

if IN_PRETRAIN:
    wandb_note = 'with imagenet pretrain'
    EPOCHS = 250
else:
    wandb_note = 'train from scratch'
    EPOCHS = 2000

metric_names = ['pointing game', 'top k', 'relevance rank']

tr_data = LiverDEC(train=True, no_aug=False)
tr_loader = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

tr_data_no_aug = LiverDEC(train=True, no_aug=True)
tr_loader_no_aug = DataLoader(tr_data_no_aug, num_workers=4)

te_data = LiverUnn()
te_loader = DataLoader(te_data, num_workers=4)

lr = 0.05*BATCH_SIZE / 256
criterion = nn.CosineSimilarity(dim=1).to(DEVICE)

for run in range(NUM_RUNS):

    wandb.init(project="ctliver", entity="wickstrom", reinit=True, tags=[wandb_note])
    wandb.run.name = wandb_note+f" (run {run}, with k = 5, with wide/narrow)"
    model = SimSiam(models.resnet50, pretrained=IN_PRETRAIN).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    for epoch in range(EPOCHS):

        print(f"Run number {run}, epoch number {epoch}")

        loss_val = train_simsiam(tr_loader, model, criterion, optimizer)

        if epoch % 5 == 0:

            metrics = [PointingGame(), TopKIntersection(), RelevanceRankAccuracy()]

            modules = list(model.encoder.children())[:-1]
            encoder = nn.Sequential(*modules, nn.Flatten()).to(DEVICE)
            encoder.eval()

            relax_scores = xai_eval(encoder, te_loader, metrics)
            knn_score = knn_eval(encoder, K_VAL, tr_loader_no_aug, te_loader)
            map_score = map_eval(encoder, K_VAL, tr_loader_no_aug, te_loader)

            relax_examples = get_wandb_examples(encoder, te_data)

            wandb.log({"loss": loss_val,
                        "pointing game": relax_scores[0],
                        "top k": relax_scores[1],
                        "relevance rank": relax_scores[2],
                        "knn accuracy": knn_score,
                        "map score": map_score,
                        "relax examples": relax_examples})
