import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn

class GCNEncoder2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=False, normalize=False, bias = False)
        self.conv2 = GCNConv(hidden_dim, output_dim, add_self_loops=False, normalize=False, bias = False)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_weight)
        z = self.conv2(x, data.edge_index, data.edge_weight)
        return z  # bottleneck embedding


class GraphAE2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = GCNEncoder2(input_dim, hidden_dim, output_dim)
        self.regressor = nn.Linear(output_dim, 1)

    def forward(self, data):
        z = self.encoder(data)
        pred = self.regressor(z).squeeze(-1)
        return pred, z


def train_GCNAE2(parameters, data, device):
    input_dim = data.x.size(-1)
    lr = parameters['lr']
    epochs = parameters['epochs']
    hidden_dim = parameters['hidden_dim']
    output_dim = 1

    model = GraphAE2(input_dim, hidden_dim, output_dim).to(device).to(torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    y = data.y.to(torch.float32).squeeze()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(data)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        #if epoch % 50 == 0:
            #print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, z = model(data)
    return z.detach().cpu().numpy()