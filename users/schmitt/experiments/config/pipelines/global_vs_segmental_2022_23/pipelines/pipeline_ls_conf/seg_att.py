from typing import Dict, Optional, Tuple, Any, List
import copy

from sisyphus import Path

from i6_core.returnn.training import Checkpoint

# from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog import ReturnnDecodingExperimentV2


default_import_model_name = "glob.conformer.mohammad.5.6"


def seg_att_import_global_global_ctc_align(
        n_epochs_list: Tuple[int] = (10, 100),
        const_lr_list: Tuple[float] = (1e-4,),
):
  for n_epochs in n_epochs_list:
    for const_lr in const_lr_list:
      alias = "models/ls_conformer/import_%s/seg_att_global_ctc_align/%d-epochs_%f-const-lr" % (
        default_import_model_name, n_epochs, const_lr)

      config_builder = get_seg_att_config_builder()

      train_exp = SegmentalTrainExperiment(
        config_builder=config_builder,
        alias=alias,
        n_epochs=n_epochs,
        import_model_train_epoch1=external_checkpoints[default_import_model_name],
        align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
        lr_opts={
          "type": "const_then_linear",
          "const_lr": const_lr,
          "const_frac": 1 / 3,
          "final_lr": 1e-6,
          "num_epochs": n_epochs
        },
      )
      checkpoints, model_dir, learning_rates = train_exp.run_train()

      recog_seg_att_import_global(
        alias=alias,
        config_builder=config_builder,
        checkpoint=checkpoints[n_epochs],
        analyse=True,
        search_corpus_key="dev-other",
        att_weight_seq_tags=[
          "dev-other/6467-94831-0006/6467-94831-0006",  # global 2 err, win-size-8 + win-size-128 + seg correct
          "dev-other/8254-84205-0021/8254-84205-0021",  # seg + win-size-8 + win-size-128 2 err, global correct
          "dev-other/6123-59150-0002/6123-59150-0002",  # seg + win-size-8 2 err, win-size-128 1 err, global correct
          "dev-other/1585-131718-0027/1585-131718-0027",  # global 2 err, win-size-8 + win-size-128 + seg correct
          "dev-other/1585-157660-0007/1585-157660-0007",  # seg 5 err, win-size-8 + win-size-128 4 err, global correct
          "dev-other/6123-59150-0008/6123-59150-0008",  # global 2 err, win-size-8 + win-size-128 1 err, seg correct
          "dev-other/1650-167613-0026/1650-167613-0026",  # seg 2 err, win-size-8 + win-size-128 3 err, global correct
          "dev-other/1686-142278-0018/1686-142278-0018",  # global 2 err, win-size-8 + win-size-128 + seg correct
          "dev-other/1701-141759-0026/1701-141759-0026",  # seg + win-size-8 + win-size-128 2 err, global correct
          "dev-other/2506-11278-0017/2506-11278-0017",  # all correct
          "dev-other/2506-11278-0025/2506-11278-0025",  # all correct
          "dev-other/2506-13150-0004/2506-13150-0004",  # all correct
          "dev-other/3660-172182-0035/3660-172182-0035",  # seg + win-size-8 + win-size-128 2 err, global correct
          "dev-other/4153-186222-0014/4153-186222-0014",  # global 3 err, win-size-8 + win-size-128 1 err, seg correct
          "dev-other/4570-14911-0000/4570-14911-0000",  # global 2 err, win-size-8 + win-size-128 1 err, seg correct
          "dev-other/5849-50873-0033/5849-50873-0033",  # seg 2 err, global + win-size-8 + win-size-128 correct
          "dev-other/6123-59186-0009/6123-59186-0009",  # seg 1 err, win-size-8 1 err, global + win-size-128 correct
          "dev-other/6267-65525-0049/6267-65525-0049",  # global 2 err, win-size-8 + win-size-128 + seg correct
          "dev-other/8288-274162-0025/8288-274162-0025",  # global 3 err, win-size-8 + win-size-128 + seg correct
        ],
      )


def seg_att_import_global_global_ctc_align_align_augment(
        n_epochs_list: Tuple[int] = (10, 100),
        const_lr_list: Tuple[float] = (1e-4,),
):
  for n_epochs in n_epochs_list:
    for const_lr in const_lr_list:
      alias = "models/ls_conformer/import_%s/seg_att_global_ctc_align_align_augment/%d-epochs_%f-const-lr" % (
        default_import_model_name, n_epochs, const_lr)

      config_builder = get_seg_att_config_builder()

      train_exp = SegmentalTrainExperiment(
        config_builder=config_builder,
        alias=alias,
        n_epochs=n_epochs,
        import_model_train_epoch1=external_checkpoints[default_import_model_name],
        align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
        align_augment=True,
        lr_opts={
          "type": "const_then_linear",
          "const_lr": const_lr,
          "const_frac": 1 / 3,
          "final_lr": 1e-6,
          "num_epochs": n_epochs
        }
      )
      checkpoints, model_dir, learning_rates = train_exp.run_train()

      recog_seg_att_import_global(
        alias=alias,
        config_builder=config_builder,
        checkpoint=checkpoints[n_epochs],
        analyse=True,
        search_corpus_key="dev-other"
      )


def get_seg_att_config_builder(
        use_positional_embedding: bool = False,
        search_remove_eos: bool = False,
):
  model_type = "librispeech_conformer_seg_att"
  variant_name = "seg.conformer.like-global"
  variant_params = copy.deepcopy(models[model_type][variant_name])
  variant_params["network"]["use_weight_feedback"] = False
  variant_params["network"]["use_positional_embedding"] = use_positional_embedding
  variant_params["network"]["search_remove_eos"] = search_remove_eos

  config_builder = LibrispeechConformerSegmentalAttentionConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
  )

  return config_builder


# def train_seg_att_import_global(
#         alias: str,
#         config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
#         align_targets: Dict[str, Path],
#         n_epochs: int,
#         import_model_name: str,
#         const_lr: float = 1e-4,
#         const_frac: float = 1/3,
#         final_lr: float = 1e-6,
#         cleanup_old_models: Optional[Dict] = None,
#         only_train_length_model: bool = False,
#         align_augment: bool = False,
# ):
#   cleanup_old_models = cleanup_old_models if cleanup_old_models is not None else {"keep_best_n": 1, "keep_last_n": 1}
#
#   train_opts = {
#     "cleanup_old_models": cleanup_old_models,
#     "lr_opts": {
#       "type": "const_then_linear",
#       "const_lr": const_lr,
#       "const_frac": const_frac,
#       "final_lr": final_lr,
#       "num_epochs": n_epochs
#     },
#     "import_model_train_epoch1": external_checkpoints[import_model_name],
#     "dataset_opts": {
#       "hdf_targets": align_targets
#     },
#     "only_train_length_model": only_train_length_model,
#     "align_augment": align_augment
#   }
#
#   checkpoints, model_dir, learning_rates = run_train(
#     config_builder=config_builder,
#     variant_params=config_builder.variant_params,
#     n_epochs=n_epochs,
#     train_opts=train_opts,
#     alias=alias
#   )
#
#   return checkpoints, model_dir, learning_rates


def recog_seg_att_import_global(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        checkpoint: Checkpoint,
        search_corpus_key: str,
        concat_num: Optional[int] = None,
        search_rqmt: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        analyse: bool = False,
        att_weight_seq_tags: Optional[List[str]] = None,
):
  recog_exp = ReturnnDecodingExperimentV2(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    corpus_key=search_corpus_key,
    concat_num=concat_num,
    search_rqmt=search_rqmt,
    batch_size=batch_size
  )
  recog_exp.run_eval()

  if analyse:
    if concat_num is not None:
      raise NotImplementedError

    recog_exp.run_analysis(
      ground_truth_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[search_corpus_key],
      att_weight_ref_alignment_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[search_corpus_key],
      att_weight_ref_alignment_blank_idx=10025,
      att_weight_seq_tags=att_weight_seq_tags,
    )
    # run_analysis(
    #   config_builder=config_builder,
    #   variant_params=config_builder.variant_params,
    #   ground_truth_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[analysis_corpus_key],
    #   att_weight_ref_alignment_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[analysis_corpus_key],
    #   corpus_key=analysis_corpus_key,
    #   att_weight_ref_alignment_blank_idx=10025,  # TODO: change this to a non-hardcoded index
    #   forward_recog_opts=forward_recog_opts,
    #   checkpoint=checkpoint,
    #   alias=alias
    # )


# def train_recog_seg_att_import_global(
#         alias: str,
#         align_targets: Dict[str, Path],
#         n_epochs: int,
#         import_model_name: str,
#         const_lr: float = 1e-4,
#         const_frac: float = 1/3,
#         final_lr: float = 1e-6,
#         cleanup_old_models: Optional[Dict] = None,
#         align_augment: bool = False,
#         only_train_length_model: bool = False,
#         chunking_opts: Optional[Dict] = None,
#         use_positional_embedding: bool = False,
#         analyse=False,
# ):
#   config_builder = get_seg_att_config_builder(
#     use_positional_embedding=use_positional_embedding,
#   )
#
#   train_exp = SegmentalTrainExperiment(
#     config_builder=config_builder,
#     alias=alias,
#     n_epochs=n_epochs,
#     import_model_name=import_model_name,
#     const_lr=const_lr,
#     const_frac=const_frac,
#     final_lr=final_lr,
#     cleanup_old_models=cleanup_old_models,
#     align_augment=align_augment,
#     align_targets=align_targets,
#     chunking_opts=chunking_opts,
#     only_train_length_model=only_train_length_model
#   )
#
#   checkpoints, model_dir, learning_rates = train_exp.run_train()
#
#   recog_seg_att_import_global(
#     alias=alias,
#     config_builder=config_builder,
#     checkpoint=checkpoints[n_epochs],
#     analyse=analyse,
#     search_corpus_key="dev-other"
#   )
#
#   return checkpoints
