from sisyphus import *

import os

from recipe.i6_experiments.users.mann.setups.nn_system import BaseSystem, NNSystem, ExpConfig, FilterAlignmentPlugin
from recipe.i6_experiments.users.mann.setups.librispeech.nn_system import LibriNNSystem
from recipe.i6_experiments.users.mann.setups import prior
from recipe.i6_experiments.common.datasets import switchboard
from i6_experiments.users.mann.setups.legacy_corpus import swb1 as legacy
from recipe.i6_core import features
from recipe.i6_core import returnn
from recipe.i6_core import meta

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput, RasrInitArgs
from .common import (
    init_segment_order_shuffle,
    init_extended_train_corpus,
    Binaries,
)

init_nn_args = {
    'name': 'crnn',
    'corpus': 'train',
    'dev_size': 0.05,
    'alignment_logs': True,
}

default_nn_training_args = {
    'feature_corpus': 'train',
    'num_epochs': 300,
    'partition_epochs': {'train': 6, 'dev' : 1},
    'save_interval': 4,
    'num_classes': lambda s: s.num_classes(),
    'time_rqmt': 120,
    'mem_rqmt' : 24,
    'use_python_control': True,
    'log_verbosity': 3,
    'feature_flow': 'gt'
}

default_scorer_args = {
    'prior_mixtures': ('train', 'init_mixture'),
    'prior_scale': 0.70,
    'feature_dimension': 40
}

default_recognition_args = {
    'corpus': 'dev',
    'pronunciation_scale': 1.0,
    'lm_scale': 10.,
    'search_parameters': {
        'beam-pruning': 16.0,
        'beam-pruning-limit': 100000,
        'word-end-pruning': 0.5,
        'word-end-pruning-limit': 10000},
    'lattice_to_ctm_kwargs' : { 'fill_empty_segments' : True,
                                'best_path_algo': 'bellman-ford' },
    'rtf': 50,
}

default_cart_lda_args = {
    'corpus': 'train',
    'initial_flow': 'gt',
    'context_flow': 'gt',
    'context_size':  15,
    'alignment': 'init_align',
    'num_dim': 40,
    'num_iter':  2,
    'eigenvalue_args': {},
    'generalized_eigenvalue_args': {'all': {'verification_tolerance': 1e14} }
}

# import paths
PREFIX_PATH = "/work/asr3/michel/setups-data/SWB_sis/"
PREFIX_PATH_asr4 = "/work/asr4/michel/setups-data/SWB_sis/"

external_alignments = {
    "tuske": Path('/work/asr2/zeyer/setups-data/switchboard/2016-01-28--crnn/tuske__2016_01_28__align.combined.train', cached=True),
    "init_align": Path('/work/asr3/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.j3oDeQH1UNjp/output/alignment.cache.bundle', cached=True),
    "init_gmm": Path('/work/asr4/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.BF7Xi6M0bF2X/output/alignment.cache.bundle', cached=True),
}

external_alignment_logs = {
    k: Path(
        v.get_path()[:-len('cache.bundle')] + f'alignment.log.{id}.gz'
    ) for k, v in external_alignments.items() for id in range(1, 201)
    if k != 'tuske'
}

default_alignment_logs = ['/work/asr3/michel/setups-data/SWB_sis/' + \
    'mm/alignment/AlignmentJob.j3oDeQH1UNjp/output/alignment.log.{id}.gz' \
        .format(id=id) for id in range(1, 201)]
extra_alignment_logs = [
    f'/work/asr4/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.BF7Xi6M0bF2X/output/alignment.log.{id}.gz'
    for id in range(1, 201)
]
default_cart_file = Path(PREFIX_PATH + "cart/estimate/EstimateCartJob.Wxfsr7efOgnu/output/cart.tree.xml.gz", cached=True)

default_mixture_path = Path(PREFIX_PATH_asr4 + "mm/mixtures/EstimateMixturesJob.accumulate.Fb561bWZLwiJ/output/am.mix",cached=True)
default_mono_mixture_path = Path(PREFIX_PATH_asr4 + "mm/mixtures/EstimateMixturesJob.accumulate.m5wLIWW876pl/output/am.mix", cached=True)
default_feature_paths = {
    'train': PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.Jlfrg2riiRX3/output/gt.cache.bundle",
    'dev'  : PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.dVkMNkHYPXb4/output/gt.cache.bundle",
    'eval' : PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.O4lUG0y7lrKt/output/gt.cache.bundle",
    "dev_extra": "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/features/gammatones/FeatureExtraction.Gammatone.pp9W8m2Z8mHU/output/gt.cache.bundle",
}

SCTK_PATH = Path('/u/beck/programs/sctk-2.4.0/bin/')

TOTAL_FRAMES = 91_026_136

def get_prior_args():
    return {
        "total_frames": TOTAL_FRAMES,
        "eps": 1e-12,
    }

CORPUS_NAME = "switchboard-1"

import enum
class BinarySetup(enum.Enum):
    Download = 0
    Legacy = 1

def get_legacy_switchboard_system(binaries: BinarySetup=BinarySetup.Download):
    """Returns the an NNSystem for the legacy switchboard corpus setup."""
    # setup binaries and environment
    epochs = [12, 24, 32, 80, 160, 240, 320]
    if binaries == BinarySetup.Download:
        binaries = init_binaries()
        init_env()

    # corpus mappings
    subcorpus_mapping = { 'train': 'full', 'dev': 'dev_zoltan', 'eval': 'hub5-01'}
    train_eval_mapping = { 'train': 'train', 'dev': 'eval', 'eval': 'eval'}

    system = NNSystem(
        num_input=40,
        epochs=epochs,
        native_ops_path=binaries.native_lstm,
    )

    system.rasr_init_args = RasrInitArgs(
        am_args=None,
        costa_args=None,
        feature_extraction_args=None,
        scorer="hub5",
        scorer_args={"sctk_binary_path": SCTK_PATH},
    )

    system.dev_copora = ["dev"]

    # Create the system
    for c, subcorpus in subcorpus_mapping.items():
        corpus = legacy.corpora[c][subcorpus]
        
        rasr_input = RasrDataInput(
            corpus_object=corpus,
            lexicon={
                "filename": legacy.lexica[train_eval_mapping[c]],
                "normalize_pronunciation": False,
            },
            lm={
                "filename": legacy.lms[train_eval_mapping[c]],
                "type": "ARPA",
                "scale": 12.0},
            concurrent=legacy.concurrent[train_eval_mapping[c]].get(subcorpus, 20),
            glm=legacy.glm_path.get(subcorpus, None),
            stm=legacy.stm_path.get(subcorpus, None),
        )
        system.add_corpus(c, rasr_input, add_lm=(c != "train"))
        system.feature_flows[c]['gt'] = features.basic_cache_flow(
            Path(default_feature_paths[c], cached=True),
        )
        system.feature_bundles[c]['gt'] = tk.Path(default_feature_paths[c], cached=True)

    system.alignments["train"].update(external_alignments.copy())
    system.mixtures['train']['init_mixture'] = default_mixture_path
    system._init_am()

    st = system.crp["base"].acoustic_model_config.state_tying
    st.type = "cart"
    st.file = default_cart_file
    system.set_num_classes("cart", 9001)
    system.crp["train"].acoustic_model_config = system.crp["base"].acoustic_model_config._copy()
    system.crp["train"].acoustic_model_config.state_tying = system.crp["base"].acoustic_model_config.state_tying
    del system.crp['train'].acoustic_model_config.tdp

    for args in ["default_nn_training_args", "default_scorer_args", "default_recognition_args"]:
        setattr(system, args, globals()[args])
    
    system.init_nn(**init_nn_args)
    for c in ["dev", "eval"]:
        system.set_hub5_scorer(corpus=c, sctk_binary_path=SCTK_PATH)

    # plugins
    system.plugins["filter_alignment"] = FilterAlignmentPlugin(system, **init_nn_args)
    return system

def init_prior_system(system):
    system.prior_system = prior.PriorSystem(system, TOTAL_FRAMES)

def add_zhou_corpus(system):
    from recipe.i6_core import features
    from recipe.i6_core import corpus

    all_segments = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/zhou-files-dev/segments"
    cv_feature_bundle = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/features/gammatones/FeatureExtraction.Gammatone.pp9W8m2Z8mHU/output/gt.cache.bundle"
    corpus_file = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/zhou-files-dev/hub5_00.corpus.cleaned.gz"

    overlay_name = "dev_zhou"
    system.add_overlay("dev", overlay_name)
    system.crp[overlay_name].concurrent = 1
    system.crp[overlay_name].segment_path = all_segments
    system.crp[overlay_name].corpus_config = corpus_config = system.crp[overlay_name].corpus_config._copy()
    system.crp[overlay_name].corpus_config.file = corpus_file
    system.all_segments[overlay_name] = all_segments

    system.crp[overlay_name].acoustic_model_config = system.crp[overlay_name].acoustic_model_config._copy()
    del system.crp[overlay_name].acoustic_model_config.tdp
    system.feature_bundles[overlay_name]["gt"] = tk.Path(cv_feature_bundle, cached=True)
    system.feature_flows[overlay_name]["gt"] = flow = features.basic_cache_flow(tk.Path(cv_feature_bundle, cached=True))

def get_bw_switchboard_system():
    from .librispeech import default_tf_native_ops
    binaries = Binaries(
        returnn=None,
        native_lstm=default_tf_native_ops,
        rasr=None,
    )
    system = get_legacy_switchboard_system(binaries)
    # setup monophones
    system.set_state_tying("monophone")
    # setup prior from transcription
    system.prior_system = prior.PriorSystem(system, TOTAL_FRAMES)
    system.default_nn_training_args["num_epochs"] = 300
    return system

def get():
    res = get_bw_switchboard_system()
    add_zhou_corpus(res)

    init_extended_train_corpus(
        res, CORPUS_NAME,
        cv_source_corpus="dev_zhou",
        legacy_trainer=True,
    )

    res.init_dump_system(
        segments=[
            "switchboard-1/sw02001A/sw2001A-ms98-a-0041",
            "switchboard-1/sw02001A/sw2001A-ms98-a-0047",
            "switchboard-1/sw02001B/sw2001B-ms98-a-0004",
            "switchboard-1/sw02001B/sw2001B-ms98-a-0024"
        ],
        occurrence_thresholds=(0.1, 0.05),
    )

    return res
