from typing import Tuple

import torch
from matplotlib import pyplot as plt


def main():
    pass


if __name__ == '__main__':
    main()


def prepare_data(N: int, w_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.cat([torch.ones(N, 1), torch.randn(N, 2)], dim=1)
    y = torch.mv(X, w_true) + torch.randn(N) * 0.5

    return X, y


def plot_loss(loss_list: torch.float) -> None:
    # plt.plot(y)
    # plt.plot(torch.mv(X, w).detach().numpy())
    plt.plot(range(1, len(loss_list) + 1), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('MSE_Loss')
    plt.show()