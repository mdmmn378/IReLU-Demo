from rnn.models import IRNN
import torch


def test_irnn():
    hidden_dim = 2
    input_dim = 2
    layer = 1
    model = IRNN(input_dim, hidden_dim, layer)
    assert (model.weight_hh_l0 == torch.eye(hidden_dim) * 0.01).all()
