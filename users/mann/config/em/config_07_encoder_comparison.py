from sisyphus import *

import os
import copy

from recipe.i6_experiments.users.mann.setups.nn_system.base_system import NNSystem, ExpConfig, RecognitionConfig, ConfigBuilder
import recipe.i6_experiments.users.mann.setups.nn_system.switchboard as swb
import recipe.i6_experiments.users.mann.setups.nn_system.librispeech as lbs
from recipe.i6_experiments.users.mann.setups import prior
from i6_experiments.users.mann.nn.config import make_baseline as make_tdnn_baseline
from i6_experiments.users.mann.nn import specaugment, learning_rates
from recipe.i6_experiments.common.datasets import librispeech

from recipe.i6_experiments.common.setups.rasr.util import RasrDataInput
from recipe.i6_experiments.common.setups.rasr import RasrSystem
from i6_core import rasr

fname = os.path.split(__file__)[1].split('.')[0]
gs.ALIAS_AND_OUTPUT_SUBDIR = fname

lbs_system = lbs.get_libri_1k_system()
swb_system = swb.get_bw_switchboard_system()
for binary in ["rasr_binary_path", "native_ops_path", "returnn_python_exe", "returnn_python_home", "returnn_root"]:
    setattr(swb_system, binary, getattr(lbs_system, binary))
lbs.init_segment_order_shuffle(swb_system)

baseline_bw = swb_system.baselines['bw_lstm_tina_swb']()
specaugment.set_config(baseline_bw.config)

from recipe.i6_experiments.users.mann.setups.tdps import CombinedModel, SimpleTransitionModel
from i6_core import rasr
# tdp_model = CombinedModel.from_fwd_probs(3/8, 1/60, 0.0)
tdp_model_tina = CombinedModel.from_fwd_probs(3/9, 1/40, 0.0)
tinas_recog_config = RecognitionConfig(
    lm_scale=3.0,
    beam_pruning=22.0,
    prior_scale=0.3,
    tdp_scale=0.1
)
swb_system.compile_configs["baseline_lstm"] = swb_system.baselines["viterbi_lstm"]()
exp_config = ExpConfig(
    compile_crnn_config=swb_system.baselines["viterbi_lstm"](),
    training_args={
        "num_classes": None,
        "alignment": None
    },
    fast_bw_args={
        "acoustic_model_extra_config": tdp_model_tina.to_acoustic_model_config(),
        "normalize_lemma_sequence_scores": False,
        # "fix_tdps_applicator": True,
    },
    # recognition_args={"extra_config": lbs.custom_recog_tdps()},
    recognition_args=tinas_recog_config.to_dict(),
    epochs=[12, 24, 48, 120, 240, 300],
    scorer_args={"prior_mixtures": None},
    reestimate_prior="transcription_prior",
    dump_epochs=[12, 300],
)

swb_system.init_dump_system(
    segments=[
        "switchboard-1/sw02001A/sw2001A-ms98-a-0041",
        "switchboard-1/sw02001A/sw2001A-ms98-a-0047",
        "switchboard-1/sw02001B/sw2001B-ms98-a-0004",
        "switchboard-1/sw02001B/sw2001B-ms98-a-0024"
    ],
    occurrence_thresholds=(0.1, 0.05),
)

configs = {}

from i6_experiments.users.mann.nn.config import TINA_UPDATES_1K, TINA_NETWORK_CONFIG, TINA_UPDATES_SWB
builder = (
    ConfigBuilder(swb_system)
    .set_lstm()
    .set_tina_scales()
    .set_config_args(TINA_UPDATES_SWB)
    .set_network_args(TINA_NETWORK_CONFIG)
    .set_transcription_prior()
    .set_specaugment()
)

def del_learning_rate(config):
    del config.config["learning_rate"]

builder.transforms.append(del_learning_rate)

#--------------------------------- init configs ---------------------------------------------------

for arch in ["lstm", "tdnn", "ffnn"]:
    config = getattr(builder, f"set_{arch}")().build()
    configs[arch] = config


#---------------------------------- compare with different prior scales ---------------------------

from i6_experiments.users.mann.experimental import helpers

ts = helpers.TuningSystem(swb_system, {})

for arch in ["lstm", "tdnn", "ffnn"]:
    config = copy.deepcopy(configs[arch])
    for prior_scale in [0.0, 0.1, 0.3]:
        config.prior_scale = prior_scale
        swb_system.run_exp(
            f"{arch}.prior_scale-{prior_scale}",
            crnn_config=config,
            exp_config=exp_config.replace(
                compile_crnn_config=None
            ) if arch in ["tdnn", "ffnn"] else exp_config,
        )

#-------------------------------------- clean up models -------------------------------------------

def clean(gpu=False):
    keep_epochs = sorted(set(
        exp_config.epochs + [4, 8]
    ))
    for name in swb_system.nn_config_dicts["train"]:
        swb_system.clean(
            name, keep_epochs,
            cleaner_args={ "gpu": int(gpu), }
        )
