from torch import nn
from collections import OrderedDict
import torch.nn.functional as F

class SequenceWise(nn.Module):
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        x, batch_size_len = x.data, x.batch_sizes
        x = self.module(x)
        x = nn.utils.rnn.PackedSequence(x, batch_size_len)
        return x

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM,
                 bidirectional=False, layer_norm=True, batch_norm=False,
                 dropout=0.9):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.rnn = rnn_type(input_size=input_size,
                            hidden_size=hidden_size,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            bias=False,
                            batch_first=True)

    def forward(self, x):
        if self.layer_norm:
            # x = self.layer_norm()
            x, input_size_list = nn.utils.rnn.pad_packed_sequence(x)
            x = x.transpose(0, 1)
            ln = nn.LayerNorm(normalized_shape=x.shape[1:]).to("cuda:6")
            x = ln(x)
            x = nn.utils.rnn.pack_padded_sequence(x, input_size_list, batch_first=True)
        if self.batch_norm:
            x_batch_size = x.batch_sizes
            bn = nn.BatchNorm1d(x.data.size(-1)).to("cuda:6")
            x = bn(x)
            x = nn.utils.rnn.PackedSequence(x, x_batch_size)
        x, _ = self.rnn(x)
        return x


class CTC_Model(nn.Module):
    def __init__(self, rnn_layer, num_class):
        super(CTC_Model, self).__init__()
        rnns = []
        rnn = BatchRNN(input_size=32,
                       hidden_size=512,
                       bidirectional=True,
                       dropout=0.2,
                       layer_norm=True,
                       batch_norm=False)
        rnns.append(("0", rnn))
        for i in range(1, rnn_layer - 1):
            rnn = BatchRNN(input_size=1024,
                           hidden_size=512,
                           bidirectional=True,
                           dropout=0.2,
                           layer_norm=True,
                           batch_norm=False)
            rnns.append((f"{i}", rnn))

        self.rnns = nn.Sequential(OrderedDict(rnns))

        fc = nn.Linear(1024, num_class)

        self.fc = SequenceWise(fc)

    def forward(self, x):
        x = self.rnns(x)
        x = self.fc(x)
        x, batch_seq = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        prob = F.softmax(x, dim=-1)

        out = F.log_softmax(x, dim=-1)

        return prob, out

    @staticmethod
    def save_package(model, optimizer=None, decoder=None, epoch=None, loss_results=None, dev_loss_results=None,
                     dev_wer_results=None):
        package = {
            'rnn_param': model.rnn_param,
            'num_class': model.num_class,
            '_drop_out': model.drop_out,
            'state_dict': model.state_dict()
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()

        if decoder is not None:
            package['decoder'] = decoder

        if epoch is not None:
            package['epoch'] = epoch

        if loss_results is not None:
            package['loss_results'] = loss_results
            package['dev_loss_results'] = dev_loss_results
            package['dev_wer_results'] = dev_wer_results
        return package



