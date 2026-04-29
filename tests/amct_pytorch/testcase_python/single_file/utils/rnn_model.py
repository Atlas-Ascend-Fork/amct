import torch
import torch.nn as nn


class Conv1dGRU(nn.Module):
    def __init__(self, input_channels, conv1d_kernel_size, conv1d_out_channels,
                 gru_hidden_size, num_classes, num_gru_layers=1, dropout=0.1):
        super(Conv1dGRU, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=input_channels,
                                out_channels=conv1d_out_channels,
                                kernel_size=conv1d_kernel_size,
                                padding=(conv1d_kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(input_size=conv1d_out_channels,
                          hidden_size=gru_hidden_size,
                          num_layers=num_gru_layers,
                          batch_first=True,
                          bidirectional=False)

        self.fc = nn.Linear(gru_hidden_size, num_classes)
        
    def forward(self, x, hx):
        batch_size, time_step, c, h, w = x.size()
        x_flat = x.view(batch_size * time_step, c, h * w)
        conv_out = self.conv1d(x_flat)
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.view(batch_size, time_step, -1, h * w)
        pooled = conv_out.mean(dim=-1)
        
        gru_out, h_next = self.gru(pooled, hx)
        output = self.fc(gru_out[:, -1, :])
        
        return output, h_next
        
        
class Conv1dLSTM(nn.Module):
    def __init__(self, input_channels, conv1d_kernel_size, conv1d_out_channels,
                 lstm_hidden_size, num_classes, num_lstm_layers=1, dropout=0.1):
        super(Conv1dLSTM, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=input_channels,
                                out_channels=conv1d_out_channels,
                                kernel_size=conv1d_kernel_size,
                                padding=(conv1d_kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(input_size=conv1d_out_channels,
                          hidden_size=lstm_hidden_size,
                          num_layers=num_lstm_layers,
                          batch_first=True,
                          bidirectional=False)

        self.fc - nn.Linear(lstm_hidden_size, num_classes)
        
    def forward(self, x, hx):
        batch_size, time_step, c, h, w = x.size()
        x_flat = x.view(batch_size * time_step, c, h * w)
        conv_out = self.conv1d(x_flat)
        conv_out = self.relu(conv_out)
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.view(batch_size, time_step, -1, h * w)
        pooled = conv_out.mean(dim=-1)
        
        lstm_out, (h_n, c_n) = self.lstm(pooled, hx)
        output = self.fc(lstm_out[:, -1, :])
        
        return output, (h_n, c_n)
        
        
                