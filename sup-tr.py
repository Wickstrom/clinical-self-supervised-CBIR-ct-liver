import wandb
import torch
import torch.nn as nn

from datasets import LiverDEC
from torchvision import models
from torch.utils.data import DataLoader
from Quantus import PointingGame, TopKIntersection, RelevanceRankAccuracy
from helpers import knn_eval, xai_eval, train_supervised, get_wandb_examples


NUM_RUNS, BATCH_SIZE, EPOCHS, K_VAL, DEVICE = 3, 32, 100, 3, 'cuda'
metric_names = ['pointing game', 'top k', 'relevance rank']

tr_data = LiverDEC(train=True, no_aug=True)
tr_loader = DataLoader(tr_data, batch_size=BATCH_SIZE, shuffle=True)

te_data = LiverDEC(train=False, no_aug=True)
te_loader = DataLoader(te_data, num_workers=4)

criterion = nn.CrossEntropyLoss().to(DEVICE)

for run in range(NUM_RUNS):

    wandb.init(project="ctliver", entity="wickstrom", reinit=True, tags=[f"run number {run}"])

    model = models.resnet50(pretrained=True).to(DEVICE)
    model.fc = nn.Linear(model.fc.in_features, 2).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)

    for epoch in range(EPOCHS):

        print(f"Run number {run}, epoch number {epoch}")
        loss_val = train_supervised(tr_loader, model, criterion, optimizer)

        if epoch % 5 == 0:

            metrics = [PointingGame(), TopKIntersection(), RelevanceRankAccuracy()]

            modules = list(model.children())[:-1]
            encoder = nn.Sequential(*modules, nn.Flatten()).to(DEVICE)
            encoder.eval()

            relax_scores = xai_eval(encoder, te_loader, metrics)
            knn_score = knn_eval(encoder, 3, tr_loader, te_loader)

            relax_examples = get_wandb_examples(encoder, te_data)

            wandb.log({"loss": loss_val,
                        "pointing game": relax_scores[0],
                        "top k": relax_scores[1],
                        "relevance rank": relax_scores[2],
                        "knn accuracy": knn_score,
                        "relax examples": relax_examples})
