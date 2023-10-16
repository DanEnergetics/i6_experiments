from typing import Optional, Dict, Any, Tuple, Callable
from sisyphus import tk

from i6_core import corpus as corpus_recipe
from i6_core.text import PipelineJob
from i6_core.returnn import ReturnnDumpHDFJob
from i6_core.features import FeatureExtractionJob

from i6_experiments.common.datasets.tedlium2.constants import DURATIONS, NUM_SEGMENTS
from i6_experiments.common.setups.rasr.gmm_system import GmmSystem
from i6_experiments.common.setups.rasr.util import HdfDataInput, AllophoneLabeling, ReturnnRasrDataInput, ForcedAlignmentArgs
from i6_experiments.common.datasets.tedlium2.lexicon import get_g2p_augmented_bliss_lexicon
from .default_tools import RETURNN_EXE, RETURNN_RC_ROOT


def build_hdf_data_input(
    features: tk.Path,
    allophone_labeling: AllophoneLabeling,
    alignments: tk.Path,
    segment_list: Optional[tk.Path] = None,
    alias_prefix: Optional[str] = None,
    partition_epoch: int = 1,
    acoustic_mixtures: Optional = None,
    seq_ordering: str = "sorted"
):

    feat_dataset = {
        "class": "SprintCacheDataset",
        "data": {
            "data": {
                "filename": features,
                "data_type": "feat",
                "allophone_labeling": {
                    "silence_phone": allophone_labeling.silence_phone,
                    "allophone_file": allophone_labeling.allophone_file,
                    "state_tying_file": allophone_labeling.state_tying_file,
                },
            }
        },
        "seq_list_filter_file": segment_list,
    }

    feat_job = ReturnnDumpHDFJob(
        data=feat_dataset,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
    )
    if alias_prefix is not None:
        feat_job.add_alias(alias_prefix + "/dump_features")
    feat_hdf = feat_job.out_hdf
    align_dataset = {
        "data": {
            "data": {
                "filename": alignments,
                "data_type": "align",
                "allophone_labeling": {
                    "silence_phone": allophone_labeling.silence_phone,
                    "allophone_file": allophone_labeling.allophone_file,
                    "state_tying_file": allophone_labeling.state_tying_file,
                },
            }
        },
        "seq_list_filter_file": segment_list,
        "class": "SprintCacheDataset",
    }
    align_job = ReturnnDumpHDFJob(data=align_dataset, returnn_python_exe=RETURNN_EXE, returnn_root=RETURNN_RC_ROOT)
    if alias_prefix is not None:
        align_job.add_alias(alias_prefix + "/dump_alignments")
    align_hdf = align_job.out_hdf

    return HdfDataInput(features=feat_hdf, alignments=align_hdf, partition_epoch=partition_epoch, acoustic_mixtures=acoustic_mixtures, seq_ordering=seq_ordering, feat_args={"segment_file": segment_list})


def dump_features_for_hybrid_training(
    gmm_system: GmmSystem,
    feature_extraction_args: Dict[str, Any],
    feature_extraction_class: Callable[[Any, ...], FeatureExtractionJob],
) -> Tuple[tk.Path, tk.Path, tk.Path]:
    features = {}
    for name in ["nn-train", "nn-cv", "nn-devtrain"]:
        #print(gmm_system.crp[name].corpus_config.file, name)
        features[name] = list(feature_extraction_class(gmm_system.crp[name], **feature_extraction_args).out_feature_bundle.values())[0]
    print(features["nn-train"])
    return features["nn-train"], features["nn-cv"], features["nn-devtrain"]

def run_forced_align_step(gmm_system, step_args):
    train_corpus_keys = step_args.pop("train_corpus_keys", gmm_system.train_corpora)
    target_corpus_keys = step_args.pop("target_corpus_keys")
    bliss_lexicon = step_args.pop("bliss_lexicon", None)
    for corpus in train_corpus_keys:
        for trg_key in target_corpus_keys:
            forced_align_trg_key = trg_key + "_forced-align"
            gmm_system.add_overlay(trg_key, forced_align_trg_key)
            if bliss_lexicon:
                gmm_system._init_lexicon(forced_align_trg_key, **bliss_lexicon)

            gmm_system.forced_align(
                target_corpus_key=forced_align_trg_key,
                feature_scorer_corpus_key=corpus,
                **step_args,
            )

def get_corpus_data_inputs(
    gmm_system: GmmSystem,
    feature_extraction_args: Dict[str, Any],
    feature_extraction_class: Callable[[Any], FeatureExtractionJob],
    alias_prefix: Optional[str] = None,
) -> Tuple[
    Dict[str, HdfDataInput],
    Dict[str, HdfDataInput],
    Dict[str, HdfDataInput],
    Dict[str, ReturnnRasrDataInput],
    Dict[str, ReturnnRasrDataInput],
]:

    train_corpus_path = gmm_system.corpora["train"].corpus_file
    cv_corpus_path = gmm_system.corpora["dev"].corpus_file

    cv_corpus_path = corpus_recipe.FilterCorpusRemoveUnknownWordSegmentsJob(
        bliss_corpus=cv_corpus_path,
        bliss_lexicon=get_g2p_augmented_bliss_lexicon(),
        all_unknown=False
    ).out_corpus

    total_train_num_segments = NUM_SEGMENTS["train"]

    all_train_segments = corpus_recipe.SegmentCorpusJob(train_corpus_path, 1).out_single_segment_files[1]
    cv_segments = corpus_recipe.SegmentCorpusJob(cv_corpus_path, 1).out_single_segment_files[1]

    dev_train_size = 500 / total_train_num_segments
    splitted_train_segments_job = corpus_recipe.ShuffleAndSplitSegmentsJob(
        all_train_segments,
        {"devtrain": dev_train_size, "unused": 1 - dev_train_size},
    )
    devtrain_segments = splitted_train_segments_job.out_segments["devtrain"]

    devtrain_corpus_path = corpus_recipe.FilterCorpusBySegmentsJob(train_corpus_path, devtrain_segments).out_corpus
    # ******************** NN Init ********************
    gmm_system.add_overlay("train", "nn-train")
    gmm_system.crp["nn-train"].segment_path = all_train_segments
    gmm_system.crp["nn-train"].concurrent = 1
    gmm_system.crp["nn-train"].corpus_config.file = train_corpus_path
    gmm_system.crp["nn-train"].corpus_duration = DURATIONS["train"]

    gmm_system.add_overlay("dev", "nn-cv")
    gmm_system.crp["nn-cv"].corpus_config.file = cv_corpus_path
    gmm_system.crp["nn-cv"].segment_path = cv_segments
    gmm_system.crp["nn-cv"].concurrent = 1
    gmm_system.crp["nn-cv"].corpus_duration = DURATIONS["dev"]

    gmm_system.add_overlay("train", "nn-devtrain")
    gmm_system.crp["nn-devtrain"].segment_path = devtrain_segments
    gmm_system.crp["nn-devtrain"].concurrent = 1
    #gmm_system.crp["nn-devtrain"].corpus_config.file = devtrain_corpus_path
    gmm_system.crp["nn-devtrain"].corpus_duration = DURATIONS["train"] * dev_train_size

    # ******************** extract features ********************

    train_features, cv_features, devtrain_features = dump_features_for_hybrid_training(
        gmm_system,
        feature_extraction_args,
        feature_extraction_class,
    )

    allophone_labeling = AllophoneLabeling(
        silence_phone="[SILENCE]",
        allophone_file=gmm_system.allophone_files["train"],
        state_tying_file=gmm_system.jobs["train"]["state_tying"].out_state_tying,
    )

    forced_align_args = ForcedAlignmentArgs(
        name="nn-cv",
        target_corpus_keys=["nn-cv"],
        flow="uncached_mfcc+context+lda+vtln+cmllr",  # TODO??
        feature_scorer="train_vtln+sat",
        scorer_index=-1,
        bliss_lexicon={
            "filename": get_g2p_augmented_bliss_lexicon(),
            "normalize_pronunciation": False,
        },
        dump_alignment=True,
    )

    run_forced_align_step(gmm_system, forced_align_args)

    nn_train_data = build_hdf_data_input(
        features=train_features,
        alignments=gmm_system.outputs["train"]["final"].as_returnn_rasr_data_input().alignments.alternatives["bundle"],
        allophone_labeling=allophone_labeling,
        alias_prefix=alias_prefix + "/nn_train_data",
        partition_epoch=5,
        acoustic_mixtures=gmm_system.outputs["train"]["final"].acoustic_mixtures,
        seq_ordering="laplace:.1000"
    )
    tk.register_output(f"{alias_prefix}/nn_train_data/features", nn_train_data.features)
    tk.register_output(f"{alias_prefix}/nn_train_data/alignments", nn_train_data.alignments)
    nn_devtrain_data = build_hdf_data_input(
        features=devtrain_features,
        alignments=gmm_system.outputs["train"]["final"].as_returnn_rasr_data_input().alignments.alternatives["bundle"],
        allophone_labeling=allophone_labeling,
        segment_list=devtrain_segments,
        alias_prefix=alias_prefix + "/nn_devtrain_data",
        partition_epoch=1,
        #acoustic_mixtures=gmm_system.outputs["train"]["final"].acoustic_mixtures,
        seq_ordering="sorted"
    )
    tk.register_output(f"{alias_prefix}/nn_devtrain_data/features", nn_devtrain_data.features)
    tk.register_output(f"{alias_prefix}/nn_devtrain_data/alignments", nn_devtrain_data.alignments)
    nn_cv_data = build_hdf_data_input(
        features=cv_features,
        alignments=gmm_system.alignments["nn-cv_forced-align"]["nn-cv"].alternatives["bundle"],
        allophone_labeling=allophone_labeling,
        alias_prefix=alias_prefix + "/nn_cv_data",
        partition_epoch=1,
        #acoustic_mixtures=gmm_system.outputs["dev"]["final"].acoustic_mixtures,
        seq_ordering="sorted"
    )
    tk.register_output(f"{alias_prefix}/nn_cv_data/features", nn_cv_data.features)
    tk.register_output(f"{alias_prefix}/nn_cv_data/alignments", nn_cv_data.alignments)

    nn_train_data_inputs = {
        "train.train": nn_train_data,
    }
    nn_devtrain_data_inputs = {
        "train.devtrain": nn_devtrain_data,
    }

    nn_cv_data_inputs = {
        "dev.cv": nn_cv_data,
    }

    nn_dev_data_inputs = {
        "dev": gmm_system.outputs["dev"]["final"].as_returnn_rasr_data_input(),
        # "dev_kaldi_small": gmm_system.outputs["dev_kaldi_small_4_gram"]["final"].as_returnn_rasr_data_input()
    }
    nn_test_data_inputs = {
        # "test": gmm_system.outputs["test"][
        #    "final"
        # ].as_returnn_rasr_data_input(),
    }

    return (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
    )
