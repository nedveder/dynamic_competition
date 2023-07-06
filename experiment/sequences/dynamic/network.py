import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()

    def forward(self, x, mask):
        masked_x = torch.where(mask == 1, x, torch.full_like(x, float('-inf')).to(DEVICE))
        out = torch.softmax(masked_x, dim=1)
        return out


class ForwardNet(nn.Module):
    def __init__(self, input_size: int, hidden_layers: int, hidden_size: int, output_size: int, critic=False):
        """
        Initialize the neural network.

        Args:
            input_size (int): The number of input features.
            hidden_layers (int): The number of hidden layers.
            hidden_size (int): The size of the hidden layers.
            output_size (int): The number of output features.
        """
        super(ForwardNet, self).__init__()
        self.critic = critic
        self.input_layer = nn.Linear(input_size, hidden_size, device=DEVICE)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size, device=DEVICE)
                                            for _ in range(hidden_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size, device=DEVICE)
        self.activation = nn.ReLU()
        self.masked_softmax = MaskedSoftmax()

    @staticmethod
    def _apply_constraints(state: torch.Tensor, is_test) -> torch.Tensor:
        # Create masks for valid actions
        mask = torch.ones(4).repeat(state.shape[0], 1).to(DEVICE)

        # Apply constraints
        if is_test:
            assignments, trial_numbers = torch.round(state[:, 9:11] * 25), torch.round(state[:, 13] * 100)
            mask[(assignments[:, 0] <= trial_numbers - 75) | (assignments[:, 1] <= trial_numbers - 75), 0] = 0
            # mask[(assignments[:, 0] <= trial_numbers - 75), 0] = 0
            mask[(assignments[:, 0] >= 25) | (assignments[:, 1] <= trial_numbers - 75), 1] = 0
            # mask[(assignments[:, 0] >= 25), 1] = 0
            mask[(assignments[:, 1] >= 25) | (assignments[:, 0] <= trial_numbers - 75), 2] = 0
            mask[(assignments[:, 0] >= 25) | (assignments[:, 1] >= 25), 3] = 0

        return mask

    def forward(self, x: torch.Tensor, is_test=False) -> torch.Tensor:
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor.
            is_test (bool): Whether to apply constraints or not.

        Returns:
            torch.Tensor: The output tensor.
        """
        s = False
        # Convert to tensor types
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if isinstance(x, tuple):
            x = torch.tensor([*x], dtype=torch.float)
        x = x.to(DEVICE)
        # Forward pass through hidden linear layers with activation
        out = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            out = self.activation(hidden_layer(out) + out)  # Add skip connection

        # Ensure the input tensor has a batch dimension
        if len(x.shape) == 1:
            s = True
            x = x.unsqueeze(0)

        # Mask layer
        if not self.critic:
            mask = self._apply_constraints(x, is_test)
            out = self.masked_softmax(self.output_layer(out), mask)
        else:
            out = self.output_layer(out)

        return out[0] if s else out
