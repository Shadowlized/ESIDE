import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, device, network='L'):
        super(BinaryClassifier, self).__init__()
        self.device = device
        self.mlp = None

        if network == 'XL':
            self.mlp = nn.Sequential(
                nn.Linear(768, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        if network == 'LW':
            self.mlp = nn.Sequential(
                nn.Linear(768, 2048),
                nn.BatchNorm1d(2048),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        if network == 'L':
            self.mlp = nn.Sequential(
                nn.Linear(768, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        if network == 'M':
            self.mlp = nn.Sequential(
                nn.Linear(768, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5),

                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        if network == 'S':
            self.mlp = nn.Sequential(
                nn.Linear(768, 512),
                nn.LeakyReLU(0.1),

                nn.Linear(512, 256),
                nn.LeakyReLU(0.1),

                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        if network == 'XS':
            self.mlp = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

    def forward(self, features, labels=None):
        logits = self.mlp(features)
        # If labels are provided, compute the loss
        if labels is not None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(logits.squeeze(), labels.float().to(self.device))
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
