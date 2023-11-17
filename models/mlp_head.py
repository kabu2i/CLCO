from torch import nn

class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(args.in_channels, args.mlp_hidden_size),
            nn.BatchNorm1d(args.mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(args.mlp_hidden_size, args.projection_size)
        )

    def forward(self, x):
        return self.net(x)
