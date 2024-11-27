import argparse
from dataclasses import dataclass

from ray import tune
from ray.tune.search.sample import Categorical, Domain


@dataclass
class Config:
    batch_size: Domain = tune.choice([16, 32, 64, 128])
    lr: Domain = tune.uniform(1e-5, 1e-2)
    p_dropout: Domain = tune.uniform(0, 0.5)
    num_epochs: Domain = tune.randint(5, 10)
    width: Domain = tune.choice([32, 64, 128])
    height: Domain = tune.choice([2, 4, 8, 16])
    fully_connected: Domain = tune.choice([1, 2, 3])
    project: str = "hyp_opt"
    num_samples: int = 1


def parse_args():
    config = Config()

    parser = argparse.ArgumentParser()
    for field in vars(config):
        parser.add_argument(f"--{field}")
    args = parser.parse_args()

    for arg, value in vars(args).items():
        if value is None:
            continue
        elif type(getattr(config, arg)) == Domain:
            if value.isdigit():
                setattr(config, arg, Categorical([int(value)]))
            else:
                try:
                    setattr(config, arg, Categorical([float(value)]))
                except ValueError:
                    setattr(config, arg, Categorical([value]))
        else:
            setattr(config, arg, type(getattr(config, arg))(value))

    return config
