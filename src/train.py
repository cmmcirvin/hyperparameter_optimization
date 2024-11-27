import torch
import torch.nn as nn
import tqdm
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback

from models.cnn import CNN
from utils.config import parse_args
from utils.datasets import Dataset_MNIST1D


def run_epoch(config, model, dataset, mode="train"):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    model.train() if mode == "train" else model.eval()
    total_loss = 0
    total_correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(mode == "train"):
        for x, y in dataloader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            total_correct += (y_pred.argmax(1) == y).sum().item()

            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    train.report({
        f"{mode}_loss": total_loss,
        f"{mode}_acc": total_correct / len(dataset),
    })


def objective(config):

    model = CNN(10, p_dropout=config["p_dropout"], width=config["width"], height=config["height"], fully_connected=config["fully_connected"])

    for _ in tqdm.trange(config["num_epochs"]):

        run_epoch(config, model, train_dataset, mode="train")
        run_epoch(config, model, test_dataset, mode="test")


if __name__ == "__main__":

    train_dataset = Dataset_MNIST1D(mode="train")
    test_dataset = Dataset_MNIST1D(mode="test")

    args = parse_args()

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="test_loss",
            mode="min",
            num_samples=args.num_samples,
        ),
        run_config=train.RunConfig(
            callbacks=[WandbLoggerCallback(project=args.project)],
        ),
        param_space=vars(args),
    )
    tuner.fit()
