import torch
from util import prepare_data, plot_loss
from torch import nn, optim
import matplotlib.pyplot as plt


def main():
    torch.manual_seed(0)

    # データ生成
    w_true = torch.tensor([1, 2, 3], dtype=torch.float)
    N = 100
    X, y = prepare_data(N, w_true)

    # モデル構築
    model = nn.Linear(in_features=3, out_features=1, bias=False)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # print(list(model.parameters()))   # 重みは勝手に設定してくれる
    num_epochs = 10
    loss_list = []

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred.view_as(y), y)
        loss.backward()
        loss_list.append(loss.item())

        optimizer.step()

    # plt.plot(y)
    # plt.plot(model(X).detach().numpy())
    plot_loss(loss_list)


if __name__ == '__main__':
    main()
