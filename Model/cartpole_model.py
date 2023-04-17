import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 3), nn.PReLU(), # 4 input for the four states of the cart pole
            nn.Linear(3, 3), nn.PReLU(),
            nn.Linear(3, 2), nn.Softmax(dim=0)  # 2 output for the control, 0 for left and 1 for right. 
                                                # Softmax gives a predicted probability distribution.
        )

    # Predict. Output is the predicted probability distribution for actions.
    def forward(self, state):
        out = self.fc(torch.FloatTensor(state))
        # out = self.fc(state)
        return out
