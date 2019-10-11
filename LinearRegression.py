from typing import Tuple
import matplotlib.pyplot as plt

import torch


# Type Hint
def prepare_data(N: int, w_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.cat([torch.ones(N, 1), torch.randn(N, 2)], dim=1)
    y = torch.mv(X, w_true) + torch.randn(N) * 0.5

    return X, y


def main():
    torch.manual_seed(0)

    # データ生成
    w_true = torch.tensor([1, 2, 3], dtype=torch.float)
    N = 100
    X, y = prepare_data(N, w_true)

    # 重みの初期化 requires_grad=Trueにすると計算グラフ保持→逆伝播の計算ができる
    w = torch.randn(w_true.size(0), requires_grad=True)

    # 学習におけるハイパーパラメータ
    learning_rate = 0.1
    num_epochs = 20
    loss_list = []

    for epoch in range(1, num_epochs + 1):
        w.grad = None

        y_pred = torch.mv(X, w)
        loss = torch.mean((y_pred - y) ** 2)
        loss.backward()
        loss_list.append(loss)

        # print(w.grad)
        w.data = w - learning_rate * w.grad.data
        print(f'Epoch{epoch}: loss={loss.item():.4f} w={w.data} dL/dw={w.grad.data}')

    plt.plot(y)
    plt.plot(torch.mv(X, w).detach().numpy())
    # plt.plot(range(1, num_epochs+1), loss_list)
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE_Loss')
    plt.show()


if __name__ == '__main__':
    main()
