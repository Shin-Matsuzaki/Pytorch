import torch


def main():
    t1 = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.float)
    print(t1)
    # print(t1.dtype)
    # print(t1.size())
    # print(t1.size(0))
    # print(t1.size(1))
    # print(t1.shape)

    t2 = torch.ones(size=(2, 3))
    print(t2)
    print(t2.dtype)
    print(t2.size())

    # intとfloatの加法はError
    t1+t2


if __name__ == '__main__':
    main()
