import torch.nn


# send state -> return q value of each next action
class QModel(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QModel, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_size, out_features=hidden_size),
            torch.nn.LayerNorm(normalized_shape=hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=action_size)
        )

    def forward(self, s_t0):
        return self.layers.forward(s_t0)
