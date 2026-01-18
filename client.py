import flwr as fl
import torch
import torch.optim as optim
from model import SimpleNN
from utils import train, test

NUM_CLIENTS = 5
MALICIOUS_CLIENTS = [3, 4]
'''
Receives the global model

Trains locally (normal/poisoned)

Sends updated weights back

Evaluates on its local test data
'''

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader, input_dim, num_classes, poisoned):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SimpleNN(input_dim, num_classes).to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.poisoned = poisoned
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader, self.optimizer,
              self.loss_fn, self.device, self.poisoned)
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = test(self.model, self.test_loader, self.device)
        return float(1 - acc), len(self.test_loader.dataset), {"accuracy": acc}
