from tm import TMDefinition
from torch import nn

class TMRNN(nn.Module):
    def __init__(self, definition: TMDefinition, hidden_size:int=16, num_layers:int=8):
        super(TMRNN, self).__init__()
        self.tm_definition = definition

        self.rnn = nn.RNN(definition.symbol_count, hidden_size, num_layers, batch_first=True)
        self.write_linear = nn.Linear(hidden_size, definition.symbol_count)
        self.write_softmax = nn.Softmax()
        self.move_linear = nn.Linear(hidden_size, 1)
    
    def forward(self, read_sequence):
        
