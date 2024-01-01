"""
Defines the data inputs for any RASR based LibriSpeech task
"""
from sisyphus import tk, gs
from dataclasses import dataclass
from typing import Dict

from i6_experiments.common.datasets.switchboard.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.switchboard.corpus_eval import (
    get_hub5e00,
    get_hub5e01,
    get_rt03s,
    get_hub5e00_corpus_object,
    get_hub5e01_corpus_object,
    get_rt03s_corpus_object
)

from i6_experiments.common.datasets.switchboard.corpus_train import (
    get_train_corpus_object_ldc,
    get_train_corpus_object_i6_legacy
)

from i6_experiments.common.setups.rasr import (
    RasrDataInput,
    util,
    gmm_system,
)

from i6_experiments.common.setups.rasr.util import (
    RasrSteps, OutputArgs, RasrInitArgs
)

from i6_experiments.common.baselines.librispeech.default_tools import SCTK_BINARY_PATH


@dataclass()
class CorpusData:
    """
    Helper class to define all RasrDataInputs to be passed to the `System` class
    """

    train_data: Dict[str, RasrDataInput]
    dev_data: Dict[str, RasrDataInput]
    test_data: Dict[str, RasrDataInput]


def get_corpus_data_inputs(use_legacy=True, use_legacy_lexicon=False, normalize_pronunciation=False) -> CorpusData:
    """
    Create the corpus data for any LibriSpeech RASR setup

    :return: (train_data, dev_data, test_data)
    """

    # Dictionary containing all LibriSpeech CorpusObject entries
    if use_legacy:
        corpus_object_dict = get_train_corpus_object_i6_legacy()
    else:
        corpus_object_dict = get_train_corpus_object_ldc()

    LM_PATH_EVAL = tk.Path(
        "/work/asr3/michel/mann/misc/tmp/dependencies/swb.fsh.4gr.voc30k.LM.gz",
        hash_overwrite="DEFAULT_SWITCHBOARD_LM"
    )

    temporary_lm = {
        # "filename": tk.Path(
        #   "/work/asr4/vieting/setups/swb/dependencies/swb.fsh.4gr.voc30k.LM.gz",
        #   hash_overwrite="/home/tuske/work/ASR/switchboard/corpus/lm/data/mylm/swb.fsh.4gr.voc30k.LM.gz",
        # ),
        "filename": LM_PATH_EVAL,
        "type": "ARPA",
        "scale": 10,
    }


    # This is the standard LibriSpeech lexicon
    if use_legacy_lexicon:
        lexicon = {
            "filename": tk.Path("/u/corpora/speech/switchboard-1/lexicon/train.lex.v1_0_3.ci.gz"),
            "normalize_pronunciation": normalize_pronunciation,
        }
    else:
        lexicon = {
            "filename": get_bliss_lexicon(),
            "normalize_pronunciation": normalize_pronunciation,
        }

    # Here we define all corpora that are used.
    # The training corpus is dynamic based on which subset we want to use,
    # but dev and test are fixed.
    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs["switchboard"] = RasrDataInput(
        corpus_object=corpus_object_dict,
        concurrent=60,
        lexicon=lexicon,
        lm=None,
    )


    hub5e00 = get_hub5e00()
    hub5e00_corpus_object = get_hub5e00_corpus_object()

    hub5e01 = get_hub5e01()
    hub5e01_corpus_object = get_hub5e01_corpus_object()

    rt03s = get_rt03s()
    rt03s_corpus_object = get_rt03s_corpus_object()

    dev_data_inputs["hub5e00"] = RasrDataInput(
        corpus_object=hub5e00_corpus_object,
        lexicon=lexicon,
        lm=temporary_lm,
        concurrent=10,
        stm=hub5e00.stm,
        glm=hub5e00.glm,
    )

    test_data_inputs["hub5e01"] = RasrDataInput(
        corpus_object=hub5e01_corpus_object,
        lexicon=lexicon,
        lm=temporary_lm,
        concurrent=10,
        stm=hub5e01.stm,
        glm=hub5e01.glm,
    )

    test_data_inputs["rt03s"] = RasrDataInput(
        corpus_object=rt03s_corpus_object,
        lexicon=lexicon,
        lm=temporary_lm,
        concurrent=10,
        stm=rt03s.stm,
        glm=rt03s.glm,
    )

    return CorpusData(
        train_data=train_data_inputs,
        dev_data=dev_data_inputs,
        test_data=test_data_inputs,
    )


def get_init_args():
    dc_detection = False
    samples_options = {
        "audio_format": "wav",
        "dc_detection": dc_detection,
    }

    am_args = {
        "state_tying": "monophone",
        "states_per_phone": 3,
        "state_repetitions": 1,
        "across_word_model": True,
        "early_recombination": False,
        "tdp_scale": 1.0,
        "tdp_transition": (3.0, 0.0, "infinity", 0.0),  # loop, forward, skip, exit
        "tdp_silence": (0.0, 3.0, "infinity", 20.0),
        "tying_type": "global-and-nonword",
        "nonword_phones": "[LAUGHTER],[NOISE],[VOCALIZEDNOISE]",
        "tdp_nonword": (
            0.0,
            3.0,
            "infinity",
            6.0,
        ),  # only used when tying_type = global-and-nonword
    }


    costa_args = {"eval_recordings": True, "eval_lm": True}

    feature_extraction_args = {
        "gt": {
            "gt_options": {
                "minfreq": 100,
                "maxfreq": 3800,
                "channels": 40,
                "warp_freqbreak": 3700,
                "tempint_type": "hanning",
                "tempint_shift": 0.01,
                "tempint_length": 0.025,
                "flush_before_gap": True,
                "do_specint": False,
                "specint_type": "hanning",
                "specint_shift": 4,
                "specint_length": 9,
                "normalize": True,
                "preemphasis": True,
                "legacy_scaling": False,
                "without_samples": False,
                "samples_options": samples_options,
                "normalization_options": {},
            }
        },
    }

    scorer_args = {
        "sctk_binary_path": SCTK_BINARY_PATH,
    }

    return RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
        scorer_args=scorer_args,
        scorer="hub5",
    )


def get_baseline_gt_system(
    rasr_binary_path,
    alias_prefix="baselines/switchboard",
):
    stored_alias_subdir = gs.ALIAS_AND_OUTPUT_SUBDIR
    gs.ALIAS_AND_OUTPUT_SUBDIR = alias_prefix

    rasr_init_args = get_init_args()

    final_output_args = OutputArgs("final")
    final_output_args.define_corpus_type("switchboard", "train")
    final_output_args.define_corpus_type("hub5e00", "dev")
    #final_output_args.define_corpus_type("dev-other", "dev")

    # enable this if you want to create features for the following training, e.g. Hybrid
    final_output_args.add_feature_to_extract("gt")

    steps = RasrSteps()
    steps.add_step("extract", rasr_init_args.feature_extraction_args)
    steps.add_step("output", final_output_args)

    corpus_data = get_corpus_data_inputs(use_legacy=False, use_legacy_lexicon=False, normalize_pronunciation=False)

    system = gmm_system.GmmSystem(rasr_binary_path=rasr_binary_path)
    system.init_system(
        rasr_init_args=rasr_init_args,
        train_data=corpus_data.train_data,
        dev_data=corpus_data.dev_data,
        test_data=corpus_data.test_data,
    )
    system.run(steps)

    gs.ALIAS_AND_OUTPUT_SUBDIR = stored_alias_subdir

    return system