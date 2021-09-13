from typing import Dict, Type, Union

from s2s.model._gru import GRUDecModel, GRUEncModel, GRUModel
from s2s.model._lstm import LSTMDecModel, LSTMEncModel, LSTMModel
from s2s.model._rnn import RNNDecModel, RNNEncModel, RNNModel
from s2s.model._gru_attention import AttnGRUDecModel, AttnGRUEncModel, AttnGRUModel, AttnGRUBlock
from s2s.model._lstm_attention import AttnLSTMDecModel, AttnLSTMEncModel, AttnLSTMModel, AttnLSTMBlock
from s2s.model._rnn_attention import AttnRNNDecModel, AttnRNNEncModel, AttnRNNModel, AttnRNNBlock
from s2s.model._transformer import TransformerModel


Model = Union[
    AttnGRUModel,
    AttnLSTMModel,
    AttnRNNModel,
    GRUModel,
    LSTMModel,
    RNNModel,
    TransformerModel,
]

MODEL_OPTS: Dict[str, Type[Model]] = {
    AttnGRUModel.model_name: AttnGRUModel,
    AttnLSTMModel.model_name: AttnLSTMModel,
    AttnRNNModel.model_name: AttnRNNModel,
    GRUModel.model_name: GRUModel,
    LSTMModel.model_name: LSTMModel,
    RNNModel.model_name: RNNModel,
    TransformerModel.model_name: TransformerModel,
}
