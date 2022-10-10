from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.debug = False
        self.conv1 = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=64, stride=1, padding=32),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(  # input: 64,32,31
                nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.flatten = Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.flatten(x)


class MLDSONetwork(nn.Module):
    def __init__(self):
        super(MLDSONetwork, self).__init__()
        self.feature_extractor = Feature()

    def forward(self, x):
        return self.feature_extractor(x)
