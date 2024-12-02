import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from sklearn import linear_model


class GaussianCDE(L.LightningModule):  # Updated class definition
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_var_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        return mean, log_var

    def training_step(self, batch, batch_idx):
        x, y = batch
        mean, log_var = self(x)
        loss = self.loss_function(mean, log_var, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def loss_function(self, mean, log_var, y):
        var = torch.exp(log_var)
        return torch.mean(0.5 * torch.log(var) + (y - mean) ** 2 / (2 * var))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def predict_step(self, X):
        mu, logvar = self(X)
        mu = mu * self.Y_std + self.Y_mean
        logvar = logvar + torch.log(2.0*self.Y_std)
        out_mu =  mu.detach().cpu().numpy()
        out_logvar = logvar.detach().cpu().numpy()
        return out_mu, out_logvar


def create_dataloader(X, y, batch_size=32):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def fit_gaussian_cde(X_train, Y_train):
    """Fit a Gaussian conditional density estimator."""
    trainer = L.Trainer(max_epochs=100)
    model = GaussianCDE(
        input_dim=X_train.shape[1], hidden_dim=32, output_dim=Y_train.shape[1])
    X_train = torch.tensor(X_train)
    Y_train = torch.tensor(Y_train)
    model.Y_mean = Y_train.mean(dim=0)
    model.Y_std = Y_train.std(dim=0)
    Y_train = (Y_train - model.Y_mean) / model.Y_std
    train_loader = create_dataloader(X_train, Y_train, batch_size=8)
    trainer.fit(model, train_loader)
    trainer.fit_model = model
    return trainer


def fit_multitask_lasso(X_train, Y_train):
    """Fit a multi-task Lasso model."""
    underlying_model = linear_model.MultiTaskLasso(alpha=0.1)
    underlying_model.fit(X_train, Y_train)
    return underlying_model


if __name__ == "__main__":
    # Random data
    # X_train = torch.randn(20, 10)
    # Y_train = torch.randn(20, 3)
    # target_dim = Y_train.shape[1]
    # X_test = torch.randn(100, 10)
    # test_input = torch.utils.data.TensorDataset(X_test)  # do not run

    # Penicillin
    from data_utils import load_penicillin, get_indices
    X, Y, _ = load_penicillin(seed=0)
    input_dim = X.shape[1]
    target_dim = Y.shape[1]
    indices = get_indices(X.shape[0], seed=0, frac_train=0.3, frac_test=0.8, frac_val2=0.0)
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    X_train, Y_train = X[indices["train"]], Y[indices["train"]]
    X_test, Y_test = X[indices["test"]], Y[indices["test"]]
    train_loader = create_dataloader(X_train, Y_train, batch_size=8)

    # Define the model
    model = GaussianCDE(input_dim=input_dim, hidden_dim=32, output_dim=target_dim)

    # Train the model
    trainer = L.Trainer(max_epochs=100)
    # trainer.fit(model, torch.utils.data.TensorDataset(X_train, Y_train))
    trainer.fit(model, train_loader)

    # Predict
    pred = trainer.predict(model, X_test)
    # pred ~ list of 2-tuples
    mu, log_var = tuple(map(torch.stack, zip(*pred)))
    breakpoint()
