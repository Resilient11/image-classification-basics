from torch import nn

class FashionMNISTModel(nn.Module):
    def __init__(self, in_shape, hidden_unit, out_shape) -> None:
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_shape, out_features=hidden_unit ),
            nn.Linear(in_features=hidden_unit, out_features=out_shape )
        )
    def forward(self, x):
        return self.layer_stack(x)




if __name__ == '__main__':
    model = FashionMNISTModel(784, 10, 10)
    print(model)
    