from .network import *
from .attention import *
from i6_experiments.users.schmitt.experiments.swb.dataset import *
from i6_experiments.users.schmitt.recombination import *
from i6_experiments.users.schmitt.rna import *
from i6_experiments.users.schmitt.specaugment import *
from i6_experiments.users.schmitt.specaugment import _mask
from i6_experiments.users.schmitt.vocab import *
from i6_experiments.users.schmitt.switchout import *
from i6_experiments.users.schmitt.targetb import *

from recipe.i6_core.returnn.config import ReturnnConfig, CodeWrapper

import numpy as np

class TransducerSWBBaseConfig:
  def __init__(self, vocab, target="orth_classes", target_num_labels=1030, targetb_blank_idx=0, data_dim=40,
               epoch_split=6, rasr_config="/u/schmitt/experiments/transducer/config/rasr-configs/merged.config",
               _attention_type=0, post_config={}, task="train", search_data_key=None, num_epochs=150,
               label_type="bpe"):

    self.post_config = post_config

    # data
    self.target = target
    self.target_num_labels = target_num_labels
    self.targetb_blank_idx = targetb_blank_idx
    self.task = task
    self.vocab = vocab
    self.rasr_config = rasr_config
    self.epoch_split = epoch_split
    self.targetb_num_labels = target_num_labels + 1
    self._cf_cache = {}
    self._alignment = None
    self.label_type = label_type
    self.search_data_key = search_data_key

    self.extern_data = {
      "data": {
        "dim": data_dim,
        "same_dim_tags_as": {"t": CodeWrapper("DimensionTag(kind=DimensionTag.Types.Spatial, description='time')")}},
      "alignment": {
        "dim": self.targetb_num_labels, "sparse": True,
        "same_dim_tags_as": {
          "t": CodeWrapper("DimensionTag(kind=DimensionTag.Types.Spatial, description='output-len')")}}}
    if task == "search":
      self.extern_data["targetb"] = {"dim": self.targetb_num_labels, "sparse": True,
                                              "available_for_inference": False}

    # time_tag_str = "DimensionTag(kind=DimensionTag.Types.Spatial, description='time')"
    # output_len_tag_str = "DimensionTag(kind=DimensionTag.Types.Spatial, description='output-len')"  # downsampled time
    # self.extern_data_epilog = [
    #   "extern_data['data']['same_dim_tags_as'] = {'t': %s}" % time_tag_str,
    #   "extern_data['alignment']['same_dim_tags_as'] = {'t': %s}" % output_len_tag_str if alignment_same_len else "None"]

    # other options
    self.network = {}
    self.use_tensorflow = True
    if self.task == "train":
        self.beam_size = 4
    else:
        self.num_epochs = num_epochs
        self.beam_size = 12
    self.learning_rate = 0.001
    self.min_learning_rate = self.learning_rate / 50.
    self.search_output_layer = "decision"
    self.debug_print_layer_output_template = True
    self.batching = "random"
    self.log_batch_size = True
    self.batch_size = 4000
    self.max_seqs = 200
    self.max_seq_length = {target: 75}
    self.truncation = -1
    self.cleanup_old_models = {"keep_last_n": 1, "keep_best_n": 1, "keep": [150]}
    self.gradient_clip = 0
    self.adam = True
    self.optimizer_epsilon = 1e-8
    self.accum_grad_multiple_step = 3
    self.stop_on_nonfinite_train_score = False
    self.tf_log_memory_usage = True
    self.gradient_noise = 0.0
    self.learning_rate_control = "newbob_multi_epoch"
    self.learning_rate_control_error_measure = "dev_error_output/output_prob"
    self.learning_rate_control_relative_error_relative_lr = True
    self.learning_rate_control_min_num_epochs_per_new_lr = 3
    self.use_learning_rate_control_always = True
    self.newbob_multi_num_epochs = 6
    self.newbob_multi_update_interval = 1
    self.newbob_learning_rate_decay = 0.7

    # prolog
    self.import_prolog = ["from returnn.tf.util.data import DimensionTag", "import os", "import numpy as np",
                          "from subprocess import check_output, CalledProcessError"]

    self.function_prolog = [_mask, random_mask, transform]
    if task == "search":
      assert label_type != "phonemes"
      self.function_prolog += [
        get_vocab_tf,
        get_vocab_sym,
        out_str,
        targetb_recomb_recog,
        get_filtered_score_op,
        get_filtered_score_cpp,
      ]

  def get_config(self):
    config_dict = {k: v for k, v in self.__dict__.items() if
                   not (k.endswith("_prolog") or k.endswith("_epilog") or k == "post_config")}
    prolog = [prolog_item for k, prolog_list in self.__dict__.items() if k.endswith("_prolog") for prolog_item in
              prolog_list]
    epilog = [epilog_item for k, epilog_list in self.__dict__.items() if k.endswith("_epilog") for epilog_item in
              epilog_list]
    # print(epilog)
    post_config = self.post_config

    return ReturnnConfig(config=config_dict, post_config=post_config, python_prolog=prolog, python_epilog=epilog)

  def set_for_search(self, dataset_key):
    self.extern_data["targetb"] = {"dim": self.targetb_num_labels, "sparse": True, "available_for_inference": False}
    self.dataset_epilog += ["search_data = get_dataset_dict('%s')" % dataset_key]
    self.batch_size = 4000
    self.beam_size = 12

  def set_config_for_search(self, config: ReturnnConfig, dataset_key):
    config.config["extern_data"]["targetb"] = {"dim": self.targetb_num_labels, "sparse": True,
                                              "available_for_inference": False}
    # index = config.python_epilog.index("eval_datasets = {'devtrain': get_dataset_dict('devtrain')}")
    config.python_epilog += ["search_data = get_dataset_dict('%s')" % dataset_key]
    # config.python_epilog.insert(index+1, "search_data = get_dataset_dict('%s')" % dataset_key)
    config.config.update({
      "batch_size": 4000,
      "beam_size": 12
    })

  def update(self, **kwargs):
    self.__dict__.update(kwargs)

    if "EncKeyTotalDim" in kwargs:
      self.EncKeyPerHeadDim = self.EncKeyTotalDim // self.AttNumHeads
    if "AttNumHeads" in kwargs:
      self.EncKeyPerHeadDim = self.EncKeyTotalDim // self.AttNumHeads
      self.EncValuePerHeadDim = self.EncValueTotalDim // self.AttNumHeads


class TransducerSWBAlignmentConfig(TransducerSWBBaseConfig):
  def __init__(self, *args, **kwargs):

    super().__init__(*args, **kwargs)

    self.extern_data["align_score"] = {"shape": (1,), "dtype": "float32"}
    self.extern_data[self.target] = {"dim": self.target_num_labels, "sparse": True}

    self.function_prolog += [
      rna_loss,
      rna_alignment,
      rna_alignment_out,
      rna_loss_out,
      get_alignment_net_dict,
      custom_construction_algo_alignment
    ]

    self.network_prolog = [
      "get_net_dict = get_alignment_net_dict",
      "custom_construction_algo = custom_construction_algo_alignment"]

    self.network_epilog = [
      "network = get_net_dict(pretrain_idx=None)",
      "pretrain = {'copy_param_mode': 'subset', 'construction_algo': custom_construction_algo}"]


class TransducerSWBExtendedConfig(TransducerSWBBaseConfig):
  def __init__(
    self, *args, att_seg_emb_size, att_seg_use_emb, att_win_size, lstm_dim,
    att_weight_feedback, att_type, att_seg_clamp_size, att_seg_left_size, att_seg_right_size, att_area,
    att_num_heads, length_model_inputs, label_smoothing, prev_att_in_state, fast_rec_full,
    scheduled_sampling, use_attention, emit_extra_loss, efficient_loss, time_red, ctx_size="full",
    fast_rec=False, pretrain=True, with_silence=False, sep_sil_model=False, sil_idx=None, sos_idx=0,
    train_data_opts=None, dev_data_opts=None, devtrain_data_opts=None, search_data_opts=None, **kwargs):

    super().__init__(*args, **kwargs)

    self.batch_size = 10000 if self.task == "train" else 4000
    chunk_size = 60
    self.chunking = ({
      "data": chunk_size * int(np.prod(time_red)), "alignment": chunk_size}, {
      "data": chunk_size * int(np.prod(time_red)) // 2, "alignment": chunk_size // 2})
    self.accum_grad_multiple_step = 2

    self.function_prolog += [custom_construction_algo]
    # if self.task == "train":
    #   self.function_prolog += [
    #     switchout_target,
    #   ]
    # TODO check, whether all current combinations of hyperparameters are working for training and search
    self.network = get_extended_net_dict(pretrain_idx=None, learning_rate=self.learning_rate, num_epochs=150,
      enc_val_dec_factor=1, target_num_labels=self.target_num_labels, target=self.target, task=self.task,
      targetb_num_labels=self.targetb_num_labels, scheduled_sampling=scheduled_sampling, lstm_dim=lstm_dim, l2=0.0001,
      beam_size=self.beam_size, length_model_inputs=length_model_inputs, prev_att_in_state=prev_att_in_state,
      targetb_blank_idx=self.targetb_blank_idx, use_att=use_attention, fast_rec_full=fast_rec_full,
      label_smoothing=label_smoothing, emit_extra_loss=emit_extra_loss, emit_loss_scale=1.0,
      efficient_loss=efficient_loss, time_reduction=time_red, ctx_size=ctx_size, fast_rec=fast_rec,
      with_silence=with_silence, sep_sil_model=sep_sil_model, sil_idx=sil_idx, sos_idx=sos_idx)
    if use_attention:
      self.network = add_attention(self.network, att_seg_emb_size=att_seg_emb_size, att_seg_use_emb=att_seg_use_emb,
        att_win_size=att_win_size, task=self.task, EncValueTotalDim=lstm_dim * 2, EncValueDecFactor=1,
        EncKeyTotalDim=lstm_dim, att_weight_feedback=att_weight_feedback, att_type=att_type,
        att_seg_clamp_size=att_seg_clamp_size, att_seg_left_size=att_seg_left_size,
        att_seg_right_size=att_seg_right_size, att_area=att_area, AttNumHeads=att_num_heads,
        EncValuePerHeadDim=int(lstm_dim * 2 // att_num_heads), l2=0.0001, AttentionDropout=0.1,
        EncKeyPerHeadDim=int(lstm_dim // att_num_heads))

    if self.task == "train":
      assert train_data_opts and dev_data_opts and devtrain_data_opts
      self.train = get_dataset_dict_new(vocab=self.vocab, epoch_split=self.epoch_split, **train_data_opts)
      self.dev = get_dataset_dict_new(vocab=self.vocab, epoch_split=1, **dev_data_opts)
      self.eval_datasets = {'devtrain': get_dataset_dict_new(vocab=self.vocab, epoch_split=1, **devtrain_data_opts)}
    else:
      assert self.search_data_key and self.label_type != "phonemes" and search_data_opts
      self.search_data = get_dataset_dict_new(vocab=self.vocab, epoch_split=1, **search_data_opts)

    if pretrain:
      self.pretrain = {'copy_param_mode': 'subset', 'construction_algo': CodeWrapper("custom_construction_algo")}
      # self.pretrain_epilog = ["pretrain = {'copy_param_mode': 'subset', 'construction_algo': custom_construction_algo}"]

    if self.task == "search":
      self.extern_data[self.target] = {"dim": self.target_num_labels, "sparse": True}


  def set_for_search(self, dataset_key):
    # super().set_for_search(dataset_key)
    # self.extern_data[self.target] = {"dim": self.target_num_labels, "sparse": True}
    return

  def set_config_for_search(self, config: ReturnnConfig, dataset_key):
    # super().set_config_for_search(config, dataset_key)
    # config.config["extern_data"][self.target] = {"dim": self.target_num_labels, "sparse": True}
    return
