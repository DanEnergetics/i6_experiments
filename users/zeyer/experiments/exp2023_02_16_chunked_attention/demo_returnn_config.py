#!rnn.py

"""
Demo config for the RETURNN training of the chunked attention model.

This is assumed to be run within RETURNN.
"""


import copy

from returnn.config import get_global_config
from returnn.tf.util.data import SpatialDim

from i6_experiments.users.zeineldeen.models.asr.encoder.conformer_encoder import (
    ConformerEncoder,
)
from .model import RNNDecoder


config = get_global_config()

# These options can be configured via command line.
task = config.value("task", "train")
chunk_size = config.int("chunk_size", 20)
chunk_step = config.int("chunk_step", chunk_size * 3 // 4)


use_tensorflow = True


train = {
    "class": "Task12AXDataset",
    "num_seqs": 1000,
}
dev = copy.deepcopy(train)
dev.update(
    {
        "num_seqs": 100,
        "fixed_random_seed": 42,
    }
)
target = "classes"
extern_data = {
    "data": {"dim": 9},
    "classes": {"dim": 2, "sparse": True},
}


batch_size = 1000
optimizer = {"class": "adam", "epsilon": 1e-8}
learning_rate = 0.01
num_epochs = 100


conformer_encoder = ConformerEncoder(
    target=target,
    input_layer=None,
    output_layer_name="encoder_full_seq",
    num_blocks=1,
    specaug=False,
    ff_dim=64,
    enc_key_dim=32,
    att_num_heads=1,
    dropout=0.0,
    att_dropout=0.0,
    dropout_in=0.0,
)
conformer_encoder.create_network()
if chunk_size > 0:
    chunk_size_dim = SpatialDim("chunk-size", chunk_size)
    chunked_time_dim = SpatialDim("chunked-time")
    conformer_encoder.network["encoder"] = {
        "class": "window",
        "from": "encoder_full_seq",
        "window_dim": chunk_size_dim,
        "stride": chunk_step,
        "out_spatial_dim": chunked_time_dim,
    }
else:
    chunk_size_dim = None
    chunked_time_dim = None
    conformer_encoder.network.add_copy_layer("encoder", "encoder_full_seq")


transformer_decoder = RNNDecoder(
    base_model=conformer_encoder,
    target=target,
    embed_dim=32,
    lstm_num_units=32,
    output_num_units=64,
    enc_key_dim=32,
    lstm_lm_proj_dim=32,
    dropout=0.0,
    att_dropout=0.0,
    att_num_heads=1,
    embed_dropout=0.0,
    softmax_dropout=0.0,
    rec_weight_dropout=0.0,
    label_smoothing=0.0,
    enc_chunks_dim=chunked_time_dim,
    enc_time_dim=chunk_size_dim,
)
transformer_decoder.create_network()

search_output_layer = transformer_decoder.decision_layer_name

# add full network
network = conformer_encoder.network.get_net()  # type: dict
network.update(transformer_decoder.network.get_net())
