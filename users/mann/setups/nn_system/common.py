__all__ = ["init_segment_order_shuffle", "rm_segment_order_shuffle", "init_binaries", "init_env"]

from sisyphus import Path, gs

import copy
from collections import namedtuple

from i6_core import returnn

from . import BaseSystem

Binaries = namedtuple('Binaries', ['returnn', 'native_lstm', 'rasr'])

RETURNN_REPOSITORY_URL = 'https://github.com/rwth-i6/returnn.git'
RETURNN_PYTHON_HOME = Path('/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1')
RETURNN_PYTHON_EXE = Path('/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1/bin/python3.8')

def init_binaries(
    returnn_python_exe=RETURNN_PYTHON_EXE,
    rasr_branch=None,
    rasr_commit=None,
):
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
        returnn_python_exe=returnn_python_exe,
        search_numpy_blas=False,
    ).out_op

    # compile rasr
    from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
    rasr_binary_path = compile_rasr_binaries_i6mode(
        branch=rasr_branch, commit=rasr_commit)
    return Binaries(returnn_root, native_lstm, rasr_binary_path)

def init_env(returnn_python_home=RETURNN_PYTHON_HOME):
    # append compile op python libs to default environment
    lib_subdir = "lib/python3.8/site-packages"
    libs = ["numpy.libs", "scipy.libs", "tensorflow"]
    path_buffer = ""
    for lib in libs:
        path_buffer += ":" + returnn_python_home.join_right(lib_subdir).join_right(lib) 
    # gs.DEFAULT_ENVIRONMENT_SET["LD_LIBRARY_PATH"] += path_buffer
    gs.DEFAULT_ENVIRONMENT_SET["PYTHON_PATH"] = returnn_python_home


def init_segment_order_shuffle(system, train_corpus="crnn_train", chunk_size=1000):
    system.csp[train_corpus] = copy.deepcopy(system.csp[train_corpus])
    system.csp[train_corpus].corpus_config.segment_order_shuffle = True
    system.csp[train_corpus].corpus_config.segment_order_sort_by_time_length = True
    system.csp[train_corpus].corpus_config.segment_order_sort_by_time_length_chunk_size = chunk_size

def rm_segment_order_shuffle(system, train_corpus="crnn_train"):
    del system.csp[train_corpus].corpus_config.segment_order_shuffle
    del system.csp[train_corpus].corpus_config.segment_order_sort_by_time_length
    del system.csp[train_corpus].corpus_config.segment_order_sort_by_time_length_chunk_size

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

def init_extended_train_corpus(
    system,
    corpus_name,
    train_source_corpus="train",
    cv_source_corpus="dev",
    reinit_shuffle=True,
):
    overlay_name = "train_magic"
    system.add_overlay(train_source_corpus, overlay_name)

    from recipe.i6_core import features
    from recipe.i6_core import corpus
    # from recipe

    overlay_name = "returnn_train_magic"
    system.add_overlay("train_magic", overlay_name)
    system.crp[overlay_name].concurrent = 1
    system.crp[overlay_name].corpus_config = corpus_config = system.crp[overlay_name].corpus_config._copy()

    all_segments = corpus.SegmentCorpusJob(corpus_config.file, num_segments=1)
    system.crp[overlay_name].segment_path = all_segments.out_single_segment_files[1]
    system.jobs[overlay_name]["all_segments"] = all_segments

    overlay_name = "returnn_cv_magic"
    system.add_overlay(cv_source_corpus, overlay_name)
    system.crp[overlay_name].concurrent = 1
    system.crp[overlay_name].corpus_config = corpus_config = system.crp[overlay_name].corpus_config._copy()
    corpus_config.file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(corpus_config.file, system.crp[overlay_name].lexicon_config.file).out_corpus

    all_segments = corpus.SegmentCorpusJob(corpus_config.file, num_segments=1)
    system.crp[overlay_name].segment_path = all_segments.out_single_segment_files[1]
    system.jobs[overlay_name]["all_segments"] = all_segments

    system.crp[overlay_name].acoustic_model_config = system.crp[overlay_name].acoustic_model_config._copy()
    del system.crp[overlay_name].acoustic_model_config.tdp

    merged_corpus = corpus.MergeCorporaJob(
        [system.crp[f"returnn_{k}_magic"].corpus_config.file for k in ["train", "cv"]],
        name=corpus_name,
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
