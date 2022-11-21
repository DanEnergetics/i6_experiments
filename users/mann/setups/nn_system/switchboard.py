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

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput
from .common import init_segment_order_shuffle

SCTK_PATH = Path('/u/beck/programs/sctk-2.4.0/bin/')

init_nn_args = {
    'name': 'crnn',
    'corpus': 'train',
    'dev_size': 0.05,
    'alignment_logs': True,
}

default_nn_training_args = {
    'feature_corpus': 'train',
    'alignment': ('train', 'init_align', -1),
    'num_epochs': 320,
    'partition_epochs': {'train': 6, 'dev' : 1},
    'save_interval': 4,
    'num_classes': lambda s: s.num_classes(),
    'time_rqmt': 120,
    'mem_rqmt' : 12,
    'use_python_control': True,
    'log_verbosity': 5,
    'feature_flow': 'gt'
}

default_scorer_args = {
    'prior_mixtures': ('train', 'init_mixture'),
    'prior_scale': 0.70,
    'feature_dimension': 40
}

default_recognition_args = {
    'corpus': 'dev',
    # 'flow': 'gt',
    'pronunciation_scale': 1.0,
    'lm_scale': 10.,
    'search_parameters': {
        'beam-pruning': 16.0,
        'beam-pruning-limit': 100000,
        'word-end-pruning': 0.5,
        'word-end-pruning-limit': 10000},
    'lattice_to_ctm_kwargs' : { 'fill_empty_segments' : True,
                                'best_path_algo': 'bellman-ford' },
    'rtf': 50
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
default_reduced_segment_path = '/u/mann/experiments/librispeech/recipe/setups/mann/librispeech/reduced.train.segments'
PREFIX_PATH                       = "/work/asr3/michel/setups-data/SWB_sis/"
PREFIX_PATH_asr4                  = "/work/asr4/michel/setups-data/SWB_sis/"
default_allophones_file      = PREFIX_PATH + "allophones/StoreAllophones.wNiR4cF7cdOE/output/allophones"
default_alignment_file       = Path('/work/asr3/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.j3oDeQH1UNjp/output/alignment.cache.bundle', cached=True)
extra_alignment_file         = Path('/work/asr4/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.BF7Xi6M0bF2X/output/alignment.cache.bundle', cached=True) # gmm
tuske_alignment_file = Path('/work/asr2/zeyer/setups-data/switchboard/2016-01-28--crnn/tuske__2016_01_28__align.combined.train', cached=True)
default_alignment_logs = ['/work/asr3/michel/setups-data/SWB_sis/' + \
    'mm/alignment/AlignmentJob.j3oDeQH1UNjp/output/alignment.log.{id}.gz' \
        .format(id=id) for id in range(1, 201)]
extra_alignment_logs = [
    f'/work/asr4/michel/setups-data/SWB_sis/mm/alignment/AlignmentJob.BF7Xi6M0bF2X/output/alignment.log.{id}.gz'
    for id in range(1, 201)
]
default_cart_file            = Path(PREFIX_PATH + "cart/estimate/EstimateCartJob.Wxfsr7efOgnu/output/cart.tree.xml.gz", cached=True)

default_mixture_path  = Path(PREFIX_PATH_asr4 + "mm/mixtures/EstimateMixturesJob.accumulate.Fb561bWZLwiJ/output/am.mix",cached=True)
default_mono_mixture_path = Path(PREFIX_PATH_asr4 + "mm/mixtures/EstimateMixturesJob.accumulate.m5wLIWW876pl/output/am.mix", cached=True)
default_feature_paths = {
    'train': PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.Jlfrg2riiRX3/output/gt.cache.bundle",
    'dev'  : PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.dVkMNkHYPXb4/output/gt.cache.bundle",
    'eval' : PREFIX_PATH + "features/extraction/FeatureExtraction.Gammatone.O4lUG0y7lrKt/output/gt.cache.bundle",
    "dev_extra": "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/features/gammatones/FeatureExtraction.Gammatone.pp9W8m2Z8mHU/output/gt.cache.bundle",
}

RETURNN_PYTHON_HOME = Path('/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1')
RETURNN_PYTHON_EXE = Path('/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1/bin/python3.8')

RETURNN_PYTHON_EXE_NEW = Path("/work/tools/asr/python/3.8.0_tf_2.3.4-generic+cuda10.1+mkl/bin/python3")
RETURNN_PYTHON_HOME_NEW = Path("/work/tools/asr/python/3.8.0_tf_2.3.4-generic+cuda10.1+mkl")

RETURNN_REPOSITORY_URL = 'https://github.com/rwth-i6/returnn.git'

RASR_BINARY_PATH = Path('/work/tools/asr/rasr/20220603_github_default/arch/linux-x86_64-standard')

SCTK_PATH = Path('/u/beck/programs/sctk-2.4.0/bin/')

TOTAL_FRAMES = 91026136

from collections import namedtuple
Binaries = namedtuple('Binaries', ['returnn', 'native_lstm', 'rasr'])

def init_binaries():
    # clone returnn
    from i6_core.tools import CloneGitRepositoryJob
    returnn_root_job = CloneGitRepositoryJob(
        RETURNN_REPOSITORY_URL,
    )
    returnn_root_job.add_alias('returnn')
    returnn_root = returnn_root_job.out_repository

    # compile lstm ops
    native_lstm = returnn.CompileNativeOpJob(
        "NativeLstm2",
        returnn_root=returnn_root,
        returnn_python_exe=RETURNN_PYTHON_EXE_NEW,
        search_numpy_blas=True
    ).out_op

    # compile rasr
    from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
    rasr_binary_path = compile_rasr_binaries_i6mode()
    # rasr_binary_path = None
    return Binaries(returnn_root, native_lstm, rasr_binary_path)

def init_env():
    # append compile op python libs to default environment
    lib_subdir = "lib/python3.8/site-packages"
    libs = ["numpy.libs", "scipy.libs", "tensorflow"]
    path_buffer = ""
    for lib in libs:
        path_buffer += ":" + RETURNN_PYTHON_HOME_NEW.join_right(lib_subdir).join_right(lib) 
    gs.DEFAULT_ENVIRONMENT_SET["LD_LIBRARY_PATH"] += path_buffer

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
        rasr_binary_path=binaries.rasr,
        native_ops_path=binaries.native_lstm,
        returnn_python_exe=RETURNN_PYTHON_EXE,
        returnn_python_home=RETURNN_PYTHON_HOME,
        returnn_root=binaries.returnn,
    )

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
        )

        system.add_corpus(c, rasr_input, add_lm=(c != "train"))
        system.feature_flows[c]['gt'] = features.basic_cache_flow(
            Path(default_feature_paths[c], cached=True),
        )
        system.feature_bundles[c]['gt'] = tk.Path(default_feature_paths[c], cached=True)
    system.alignments['train']['init_align'] = default_alignment_file
    system.alignments['train']['init_gmm'] = extra_alignment_file
    system.alignments['train']['tuske'] = tuske_alignment_file
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
        # add glm and stm files
        system.glm_files[c] = legacy.glm_path[subcorpus_mapping[c]]
        system.stm_files[c] = legacy.stm_path[subcorpus_mapping[c]]
        system.set_hub5_scorer(corpus=c, sctk_binary_path=SCTK_PATH)

    # plugins
    system.plugins["filter_alignment"] = FilterAlignmentPlugin(system, **init_nn_args)
    return system

def init_prior_system(system):
    system.prior_system = prior.PriorSystem(system, TOTAL_FRAMES)

def get_bw_switchboard_system():
    from .librispeech import default_tf_native_ops
    binaries = Binaries(
        returnn=tk.Path(gs.RETURNN_ROOT),
        native_lstm=default_tf_native_ops,
        rasr=tk.Path(gs.RASR_ROOT).join_right('arch/linux-x86_64-standard'),
    )
    system = get_legacy_switchboard_system(binaries)
    # setup monophones
    system.set_state_tying("monophone")
    # setup prior from transcription
    system.prior_system = prior.PriorSystem(system, TOTAL_FRAMES)
    system.default_nn_training_args["num_epochs"] = 300
    return system

def init_extended_train_corpus(system, reinit_shuffle=True):
    overlay_name = "train_magic"
    system.add_overlay("train", overlay_name)

    from recipe.i6_core import features
    from recipe.i6_core import corpus

    cv_feature_bundle = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/features/gammatones/FeatureExtraction.Gammatone.pp9W8m2Z8mHU/output/gt.cache.bundle"
    overlay_name = "returnn_train_magic"
    system.add_overlay("train_magic", overlay_name)
    system.crp[overlay_name].concurrent = 1
    system.crp[overlay_name].corpus_config = corpus_config = system.crp[overlay_name].corpus_config._copy()
    system.crp[overlay_name].segment_path = corpus.SegmentCorpusJob(corpus_config.file, num_segments=1).out_single_segment_files[1]

    overlay_name = "returnn_cv_magic"
    system.add_overlay("dev", overlay_name)
    system.crp[overlay_name].concurrent = 1
    system.crp[overlay_name].segment_path = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/zhou-files-dev/segments"
    system.crp[overlay_name].corpus_config = corpus_config = system.crp[overlay_name].corpus_config._copy()
    system.crp[overlay_name].corpus_config.file = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/cv-from-hub5-00/zhou-files-dev/hub5_00.corpus.cleaned.gz"
    system.crp[overlay_name].acoustic_model_config = system.crp[overlay_name].acoustic_model_config._copy()
    del system.crp[overlay_name].acoustic_model_config.tdp
    system.feature_bundles[overlay_name]["gt"] = tk.Path(cv_feature_bundle, cached=True)
    system.feature_flows[overlay_name]["gt"] = flow = features.basic_cache_flow(tk.Path(cv_feature_bundle, cached=True))

    merged_corpus = corpus.MergeCorporaJob(
        [system.crp[f"returnn_{k}_magic"].corpus_config.file for k in ["train", "cv"]],
        name="switchboard-1",
        merge_strategy=corpus.MergeStrategy.FLAT,
    ).out_merged_corpus
    system.crp["train_magic"].corpus_config = corpus_config = system.crp["train_magic"].corpus_config._copy()
    system.crp["train_magic"].corpus_config.file = merged_corpus

    if reinit_shuffle:
        chunk_size = 300
        if isinstance(reinit_shuffle, int):
            chunk_size = reinit_shuffle
        init_segment_order_shuffle(system, "returnn_train_magic", 300)
    
    from i6_experiments.users.mann.setups.nn_system.trainer import RasrTrainer
    system.set_trainer(RasrTrainer())

    return {
        "feature_corpus": "train_magic",
        "train_corpus": "returnn_train_magic",
        "dev_corpus": "returnn_cv_magic",
    }



from collections import UserDict
class CustomDict(UserDict):
    """Custom dict that lets you map values of specific keys
    to a different value.
    """
    def map_item(self, key, func):
        d = self.copy()
        d[key] = func(d[key])
        return d
    
    def map(self, **kwargs):
        d = self.copy()
        for key, func in kwargs.items():
            d[key] = func(d[key])
        return d

# make cart questions and estimate cart on alignment
def make_cart(
    system: BaseSystem,
    hmm_partition: int=3,
    as_lut=False
):
    # create cart questions
    from i6_core.cart import PythonCartQuestions
    from i6_core.meta import CartAndLDA
    steps = legacy.cart_steps
    if hmm_partition == 1:
        steps = list(filter(lambda x: x["name"] != "hmm-state", steps))
    cart_questions = PythonCartQuestions(
        phonemes=legacy.cart_phonemes,
        steps=steps,
        hmm_states=hmm_partition,
    )
    
    args = CustomDict(default_cart_lda_args.copy())
    corpus = args.pop("corpus") 
    context_size = args.pop("context_size")
    select_feature_flow = lambda flow: meta.select_element(system.feature_flows, corpus, flow)
    select_alignment = lambda alignment: meta.select_element(system.alignments, corpus, alignment)

    from i6_core import lda
    get_ctx_flow = lambda flow: lda.add_context_flow(
            feature_net = system.feature_flows[corpus][flow],
            max_size = context_size,
            right = int(context_size / 2)
        )
    args = args.map(
        initial_flow=select_feature_flow,
        context_flow=get_ctx_flow,
        alignment=select_alignment,
    )
    
    # system.crp[corpus].acoustic_model_config.hmm.states_per_phone = hmm_partition
    cart_and_lda = CartAndLDA(
        original_crp=system.crp[corpus],
        questions=cart_questions,
        **args
    )

    system.set_state_tying("cart", cart_file=cart_and_lda.last_cart_tree)
    system.set_num_classes("cart", cart_and_lda.last_num_cart_labels)

    if as_lut:
        lut_cart = system.get_state_tying_file() # direct state-tying from cart with still has 3 states per phone
        print(lut_cart)
        for crp in system.crp.values():
            crp.acoustic_model_config.hmm.states_per_phone = 1
        system.set_state_tying("lookup", cart_file=lut_cart)
        lut = system.get_state_tying_file()
        system.set_state_tying("lut", file=lut)
        system.set_num_classes("lut", cart_and_lda.last_num_cart_labels)

    return None
