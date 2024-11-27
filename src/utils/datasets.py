import einops
import torch
from mnist1d.data import get_dataset_args, make_dataset
from torch.utils.data import Dataset


class Dataset_MNIST1D(Dataset):
    """MNIST1D dataset"""

    def __init__(self, mode="train"):
        Dataset.__init__(self)

        _defaults = get_dataset_args()
        _data = make_dataset(_defaults)

        if mode == "train":
            self.x = torch.tensor(_data["x"], dtype=torch.float32)
            self.y = torch.tensor(_data["y"], dtype=torch.int64)
        else:
            self.x = torch.tensor(_data["x_test"], dtype=torch.float32)
            self.y = torch.tensor(_data["y_test"], dtype=torch.int64)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return einops.rearrange(self.x[idx], 'd -> 1 d'), self.y[idx]
