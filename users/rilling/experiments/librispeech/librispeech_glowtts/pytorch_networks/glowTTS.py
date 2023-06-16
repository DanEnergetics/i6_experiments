from dataclasses import dataclass
import torch
import numpy
from torch import nn
import multiprocessing
import math

from IPython import embed

# from returnn.datasets.hdf import SimpleHDFWriter

from . import modules
from . import commons
from . import attentions
from .monotonic_align import maximum_path

class DurationPredictor(nn.Module): 
    """
    Duration Predictor module, trained using calculated durations coming from monotonic alignment search
    """
    def __init__(self, in_channels, filter_channels, filter_size, p_dropout):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.filter_size = filter_size
        self.p_dropout = p_dropout

        self.convs = nn.Sequential(
            modules.Conv1DBlock(in_size=self.in_channels, out_size=self.filter_channels, filter_size=self.filter_size, p_dropout=p_dropout),
            modules.Conv1DBlock(in_size=self.filter_channels, out_size=self.filter_channels, filter_size=self.filter_size, p_dropout=p_dropout),
        )
        self.proj = nn.Conv1d(in_channels=self.filter_channels, out_channels=1, kernel_size=1)

    def forward(self, x, x_mask):

        x_with_mask = (x, x_mask)
        (x, x_mask) = self.convs(x_with_mask)
        x = self.proj(x * x_mask)
        return x


class TextEncoder(nn.Module):
    """
    Text Encoder model
    """
    def __init__(self,
            n_vocab,
            out_channels,
            hidden_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=None,
            block_length=None,
            mean_only=False,
            prenet=False,
            gin_channels=0):
        """Text Encoder Model based on Multi-Head Self-Attention combined with FF-CCNs

        Args:
            n_vocab (int): Size of vocabulary for embeddings
            out_channels (int): Number of output channels
            hidden_channels (int): Number of hidden channels
            filter_channels (int): Number of filter channels
            filter_channels_dp (int): Number of filter channels for duration predictor
            n_heads (int): Number of heads in encoder's Multi-Head Attention
            n_layers (int): Number of layers consisting of Multi-Head Attention and CNNs in encoder 
            kernel_size (int): Kernel Size for CNNs in encoder layers
            p_dropout (float): Dropout probability for both encoder and duration predictor
            window_size (int, optional): Window size  in Multi-Head Self-Attention for encoder. Defaults to None.
            block_length (_type_, optional): Block length for optional block masking in Multi-Head Attention for encoder. Defaults to None.
            mean_only (bool, optional): Boolean to only project text encodings to mean values instead of mean and std. Defaults to False.
            prenet (bool, optional): Boolean to add ConvReluNorm prenet before encoder . Defaults to False.
            gin_channels (int, optional): Number of channels for speaker condition. Defaults to 0.
        """
        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.prenet = prenet
        self.gin_channels = gin_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if prenet:
            self.pre = modules.ConvReluNorm(hidden_channels, hidden_channels, hidden_channels, kernel_size=5, n_layers=3, p_dropout=0.5)
        self.encoder = attentions.Encoder(
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        window_size=window_size,
        block_length=block_length,
        )

        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj_w = DurationPredictor(hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
        x = torch.transpose(x, 1, -1) # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder(x, x_mask)

        if g is not None:
            g_exp = g.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)

        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        logw = self.proj_w(x_dp, x_mask)
        return x_m, x_logs, logw, x_mask
    
class FlowDecoder(nn.Module):
    def __init__(self, 
        in_channels, 
        hidden_channels,
        kernel_size, 
        dilation_rate, 
        n_blocks, 
        n_layers, 
        p_dropout=0., 
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        gin_channels=0):
        """Flow-based decoder model

        Args:
            in_channels (int): Number of incoming channels
            hidden_channels (int): Number of hidden channels
            kernel_size (int): Kernel Size for convolutions in coupling blocks
            dilation_rate (float): Dilation Rate to define dilation in convolutions of coupling block
            n_blocks (int): Number of coupling blocks
            n_layers (int): Number of layers in CNN of the coupling blocks
            p_dropout (float, optional): Dropout probability for CNN in coupling blocks. Defaults to 0..
            n_split (int, optional): Number of splits for the 1x1 convolution for flows in the decoder. Defaults to 4.
            n_sqz (int, optional): Squeeze. Defaults to 1.
            sigmoid_scale (bool, optional): Boolean to define if log probs in coupling layers should be rescaled using sigmoid. Defaults to False.
            gin_channels (int, optional): Number of speaker embedding channels. Defaults to 0.
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        for b in range(n_blocks):
            self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
            self.flows.append(modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
            self.flows.append(
                attentions.CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale
                )
            )
    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.n_sqz > 1:
            x, x_mask = commons.channel_squeeze(x, x_mask, self.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = commons.channel_unsqueeze(x, x_mask, self.n_sqz)
        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()

class Model(nn.Module):
    """
    Flow-based TTS model based on GlowTTS Structure
    Following the definition from https://arxiv.org/abs/2005.11129
    and code from https://github.com/jaywalnut310/glow-tts
    """
    def __init__(self,
                 n_vocab: int, 
                 hidden_channels: int, 
                 filter_channels: int, 
                 filter_channels_dp: int,
                 out_channels: int,
                 kernel_size: int = 3, 
                 n_heads: int = 2,
                 n_layers_enc:int = 6, 
                 p_dropout:float = 0.,
                 n_blocks_dec: int = 12,
                 kernel_size_dec:int = 5,
                 dilation_rate:int = 5,
                 n_block_layers:int = 4,
                 p_dropout_dec:float = 0.,
                 n_speakers:int = 0,
                 gin_channels:int = 0,
                 n_split:int = 4,
                 n_sqz:int = 1,
                 sigmoid_scale:bool = False,
                 window_size:int = None,
                 block_length:int = None,
                 mean_only:bool = False,
                 hidden_channels_enc:int = None,
                 hidden_channels_dec:int = None,
                 prenet:bool = False,
                 **kwargs
                ):
        """_summary_

        Args:
            n_vocab (int): vocabulary size
            hidden_channels (int): Number of hidden channels in encoder
            filter_channels (int): Number of filter channels in encoder
            filter_channels_dp (int): Number of filter channels in decoder
            out_channels (int): Number of channels in the output
            kernel_size (int, optional): Size of kernels in the encoder. Defaults to 3.
            n_heads (int, optional): Number of heads in the Multi-Head Attention of the encoder. Defaults to 2.
            n_layers_enc (int, optional): Number of layers in the encoder. Defaults to 6.
            p_dropout (_type_, optional): Dropout probability in the encoder. Defaults to 0..
            n_blocks_dec (int, optional): Number of coupling blocks in the decoder. Defaults to 12.
            kernel_size_dec (int, optional): Kernel size in the decoder. Defaults to 5.
            dilation_rate (int, optional): Dilation rate for CNNs of coupling blocks in decoder. Defaults to 5.
            n_block_layers (int, optional): Number of layers in the CNN of the coupling blocks in decoder. Defaults to 4.
            p_dropout_dec (_type_, optional): Dropout probability in the decoder. Defaults to 0..
            n_speakers (int, optional): Number of speakers. Defaults to 0.
            gin_channels (int, optional): Number of speaker embedding channels. Defaults to 0.
            n_split (int, optional): Number of splits for the 1x1 convolution for flows in the decoder. Defaults to 4.
            n_sqz (int, optional): Squeeze. Defaults to 1.
            sigmoid_scale (bool, optional): Boolean to define if log probs in coupling layers should be rescaled using sigmoid. Defaults to False.
            window_size (int, optional): Window size  in Multi-Head Self-Attention for encoder. Defaults to None.
            block_length (_type_, optional): Block length for optional block masking in Multi-Head Attention for encoder. Defaults to None.
            mean_only (bool, optional): Boolean to only project text encodings to mean values instead of mean and std. Defaults to False.
            hidden_channels_enc (int, optional): Number of hidden channels in encoder. Defaults to hidden_channels.
            hidden_channels_dec (_type_, optional): Number of hidden channels in decodder. Defaults to hidden_channels.
            prenet (bool, optional): Boolean to add ConvReluNorm prenet before encoder . Defaults to False.
        """
        super().__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_heads = n_heads
        self.n_layers_enc = n_layers_enc
        self.p_dropout = p_dropout
        self.n_blocks_dec = n_blocks_dec
        self.kernel_size_dec = kernel_size_dec
        self.dilation_rate = dilation_rate
        self.n_block_layers = n_block_layers
        self.p_dropout_dec = p_dropout_dec
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.hidden_channels_enc = hidden_channels_enc
        self.hidden_channels_dec = hidden_channels_dec
        self.prenet = prenet

        self.encoder = TextEncoder(
            n_vocab, 
            out_channels, 
            hidden_channels_enc or hidden_channels, 
            filter_channels, 
            filter_channels_dp, 
            n_heads, 
            n_layers_enc, 
            kernel_size, 
            p_dropout, 
            window_size=window_size,
            block_length=block_length,
            mean_only=mean_only,
            prenet=prenet,
            gin_channels=gin_channels)

        self.decoder = FlowDecoder(
            out_channels, 
            hidden_channels_dec or hidden_channels, 
            kernel_size_dec, 
            dilation_rate, 
            n_blocks_dec, 
            n_block_layers, 
            p_dropout=p_dropout_dec, 
            n_split=n_split,
            n_sqz=n_sqz,
            sigmoid_scale=sigmoid_scale,
            gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)
            
    def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, gen=False, noise_scale=1., length_scale=1.):
        if g is not None:
            g = nn.functional.normalize(self.emb_g(g.squeeze(-1))).unsqueeze(-1)
        x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g) # mean, std logs, duration logs, mask

        if gen: # durations from dp only used during generation
            w = torch.exp(logw) * x_mask * length_scale # durations
            w_ceil = torch.ceil(w) # durations ceiled
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1,2]), 1).long()
            y_max_length = None
        else:
            y_max_length = y.size(2)

        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
        z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        if gen:
            attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
            z_m = torch.matmul(attn.squeeze(1).transpose(1,2), x_m.transpose(1,2)).transpose(1,2)
            z_logs = torch.matmul(attn.squeeze(1).transpose(1,2), x_logs.transpose(1,2)).transpose(1,2)
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

            z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
            y, logdet = self.decoder(z, z_mask, g=g, reverse = True)
            return (y, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
        else:
            z, logdet = self.decoder(y, z_mask, g=g, reverse=False)
            with torch.no_grad():
                x_s_sq_r = torch.exp(-2 * x_logs)
                logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1) # [b, t, 1]
                logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (z ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
                logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), z) # [b, t, d] x [b, d, t'] = [b, t, t']
                logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
                logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']

                attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
                embed()

            z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
            z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
            return (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)

    def preprocess(self, y, y_lengths, y_max_length):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:,:,:y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        self.decoder.store_inverse()




# class Model(nn.Module):
#     """
#     Default TTS aligner with 5 convolution blocks of size 5 followed by a BLSTM
#     """

#     def __init__(
#         self,
#         conv_hidden_size: int,
#         lstm_size: int,
#         speaker_embedding_size: int,
#         dropout: float,
#         target_size: int,
#         **kwargs,
#     ):
#         super().__init__()
#         self.audio_embedding = nn.Linear(80, conv_hidden_size)
#         self.speaker_embedding = nn.Embedding(251, speaker_embedding_size)
#         self.convs = nn.Sequential(
#             Conv1DBlock(conv_hidden_size + speaker_embedding_size, conv_hidden_size, filter_size=5, dropout=dropout),
#             Conv1DBlock(conv_hidden_size, conv_hidden_size, filter_size=5, dropout=dropout),
#             Conv1DBlock(conv_hidden_size, conv_hidden_size, filter_size=5, dropout=dropout),
#             Conv1DBlock(conv_hidden_size, conv_hidden_size, filter_size=5, dropout=dropout),
#             Conv1DBlock(conv_hidden_size, conv_hidden_size, filter_size=5, dropout=dropout),
#         )
#         self.blstm = nn.LSTM(input_size=conv_hidden_size, hidden_size=lstm_size, bidirectional=True, batch_first=True)
#         self.final_linear = nn.Linear(2 * lstm_size, target_size)
#         self.lstm_size = lstm_size
#         self.target_size = target_size
#         self.dropout = dropout

#         # initialize weights
#         self.apply(self._weight_init)

#     @staticmethod
#     def _weight_init(module: torch.nn.Module):
#         if isinstance(module, torch.nn.Conv1d):
#             nn.init.xavier_normal_(module.weight)

#     def forward(
#         self,
#         audio_features: torch.Tensor,
#         speaker_labels: torch.Tensor,
#         audio_features_len: torch.Tensor,
#     ):
#         """
#         :param audio_features: [B, T, F]
#         :param speaker_labels: [B, 1]
#         :param audio_features_len: length of T as [B]
#         :return: logprobs as [B, T, #PHONES]
#         """
#         speaker_embeddings: torch.Tensor = self.speaker_embedding(torch.squeeze(speaker_labels, dim=1))
#         # manually broadcast speaker embeddings to each time step
#         speaker_embeddings = torch.repeat_interleave(
#             torch.unsqueeze(speaker_embeddings, 1), audio_features.size()[1], dim=1
#         )  # [B, T, #SPK_EMB_SIZE]
#         audio_embedding = self.audio_embedding(audio_features)  # [B, T, F]

#         conv_in = torch.concat([speaker_embeddings, audio_embedding], dim=2)  # [B, T, F]
#         conv_in = torch.swapaxes(conv_in, 1, 2)  # [B, F, T]
#         conv_out = self.convs(conv_in)
#         blstm_in = torch.permute(conv_out, dims=(0, 2, 1))  # [B, F, T] -> [B, T, F]

#         blstm_packed_in = nn.utils.rnn.pack_padded_sequence(blstm_in, audio_features_len.to("cpu"), batch_first=True)
#         blstm_packed_out, _ = self.blstm(blstm_packed_in)
#         blstm_out, _ = nn.utils.rnn.pad_packed_sequence(
#             blstm_packed_out, padding_value=0.0, batch_first=True
#         )  # [B, T, F]
#         blstm_out = nn.functional.dropout(blstm_out, p=self.dropout, training=self.training)
#         logits = self.final_linear(blstm_out)  # [B, T, #PHONES]
#         log_probs = torch.log_softmax(logits, dim=2)  # [B, T, #PHONES]

#         return log_probs


def train_step(*, model: Model, data, run_ctx, **kwargs):
    audio_features = data["audio_features"]  # [B, T, F]
    audio_features = audio_features.transpose(1, 2) # [B, F, T] necessary because glowTTS expects the channels to be in the 2nd dimension
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    phonemes = data["phonemes"][indices, :]  # [B, T] (sparse)
    phonemes_len = data["phonemes:size1"][indices]  # [B, T]
    speaker_labels = data["speaker_labels"][indices, :]  # [B, 1] (sparse)

    # print(f"phoneme shape: {phonemes.shape}")
    # print(f"phoneme length: {phonemes_len}")
    # print(f"audio_feature shape: {audio_features.shape}")
    # print(f"audio_feature length: {audio_features_len}")


    (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = model(phonemes, phonemes_len, audio_features, audio_features_len, speaker_labels)

    l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
    l_dp = commons.duration_loss(logw, logw_, phonemes_len)

    run_ctx.mark_as_loss(name="mle", loss=l_mle)
    run_ctx.mark_as_loss(name="dp", loss=l_dp)

    # logprobs = model(
    #     audio_features=audio_features,
    #     audio_features_len=audio_features_len,
    #     speaker_labels=speaker_labels,
    # )

    # transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    # ctc_loss = nn.functional.ctc_loss(
    #     transposed_logprobs,
    #     phonemes,
    #     input_lengths=audio_features_len,
    #     target_lengths=phonemes_len,
    #     blank=model.target_size - 1,
    #     reduction="sum",
    # )
    # num_frames = torch.sum(phonemes_len)
    # run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_frames)


############# FORWARD STUFF ################
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


# Duration extraction helpers are taken from
# https://github.com/as-ideas/DeepForcedAligner/blob/main/dfa/duration_extraction.py
# with commit id d1f565604bba25d4c56e3e12b289ab335980e069
# MIT license
def to_node_index(i, j, cols):
    return cols * i + j


def from_node_index(node_index, cols):
    return node_index // cols, node_index % cols


def to_adj_matrix(mat):
    """
    :param mat: [T x N] matrix where for each time frame we have the N scores of our target phoneme labels
    :return: A sparse CTC-style adjacent lattice matrix where the connection weight is the score of the "target" node
        of each connection.
    """
    rows = mat.shape[0]
    cols = mat.shape[1]

    row_ind = []
    col_ind = []
    data = []

    for i in range(rows):
        for j in range(cols):

            node = to_node_index(i, j, cols)

            if j < cols - 1:
                right_node = to_node_index(i, j + 1, cols)
                weight_right = mat[i, j + 1]
                row_ind.append(node)
                col_ind.append(right_node)
                data.append(weight_right)

            if i < rows - 1 and j < cols:
                bottom_node = to_node_index(i + 1, j, cols)
                weight_bottom = mat[i + 1, j]
                row_ind.append(node)
                col_ind.append(bottom_node)
                data.append(weight_bottom)

            if i < rows - 1 and j < cols - 1:
                bottom_right_node = to_node_index(i + 1, j + 1, cols)
                weight_bottom_right = mat[i + 1, j + 1]
                row_ind.append(node)
                col_ind.append(bottom_right_node)
                data.append(weight_bottom_right)

    adj_mat = coo_matrix((data, (row_ind, col_ind)), shape=(rows * cols, rows * cols))
    return adj_mat.tocsr()


@dataclass()
class AlignSequence:
    """
    :param logprobs: [T x F] log probabilities
    :param phonemes: [N] indexed tokens with indices in the range [0, F-1]
    """
    logprobs: np.ndarray
    phonemes: np.ndarray


def extract_durations_with_dijkstra(sequence: AlignSequence) -> np.array:
    """
    Extracts durations from the attention matrix by finding the shortest monotonic path from
    top left to bottom right.

    :return durations: [N] durations which sum to T
    """

    neg_log_weights = -sequence.logprobs[:, sequence.phonemes]
    adj_matrix = to_adj_matrix(neg_log_weights)
    dist_matrix, predecessors = dijkstra(csgraph=adj_matrix, directed=True, indices=0, return_predecessors=True)
    path = []
    pr_index = predecessors[-1]
    while pr_index != 0:
        path.append(pr_index)
        pr_index = predecessors[pr_index]
    path.reverse()

    # append first and last node
    path = [0] + path + [dist_matrix.size - 1]
    cols = neg_log_weights.shape[1]
    mel_text = {}
    durations = np.zeros(sequence.phonemes.shape[0], dtype=np.int32)

    # collect indices (mel, text) along the path
    for node_index in path:
        i, j = from_node_index(node_index, cols)
        mel_text[i] = j

    for j in mel_text.values():
        durations[j] += 1

    return durations


def forward_init_hook(run_ctx, **kwargs):
    run_ctx.hdf_writer = SimpleHDFWriter("durations.hdf", dim=None, ndim=1)
    run_ctx.pool = multiprocessing.Pool(8)


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.hdf_writer.close()


def forward_step(*, model: Model, data, run_ctx, **kwargs):
    tags = data["seq_tag"]
    audio_features = data["audio_features"]  # [B, T, F]
    audio_features_len = data["audio_features:size1"]  # [B]

    # perform local length sorting for more efficient packing
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)

    audio_features = audio_features[indices, :, :]
    phonemes = data["phonemes"][indices, :]  # [B, T] (sparse)
    phonemes_len = data["phonemes:size1"][indices]  # [B, T]
    speaker_labels = data["speaker_labels"][indices, :]  # [B, 1] (sparse)

    logprobs = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len,
        speaker_labels=speaker_labels,
    )

    numpy_logprobs = logprobs.detach().cpu().numpy()
    numpy_phonemes = phonemes.detach().cpu().numpy()

    align_sequences = []

    for single_logprobs, single_phonemes, feat_len, phon_len in zip(
        numpy_logprobs, numpy_phonemes, audio_features_len, phonemes_len
    ):
        align_sequences.append(AlignSequence(single_logprobs[:feat_len], single_phonemes[:phon_len]))

    durations = run_ctx.pool.map(extract_durations_with_dijkstra, align_sequences)
    for tag, duration, feat_len, phon_len in zip(tags, durations, audio_features_len, phonemes_len):
        total_sum = numpy.sum(duration)
        assert total_sum == feat_len
        assert len(duration) == phon_len
        run_ctx.hdf_writer.insert_batch(numpy.asarray([duration]), [len(duration)], [tag])
