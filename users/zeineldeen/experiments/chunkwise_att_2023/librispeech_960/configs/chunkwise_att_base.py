"""
based on:
users/zeineldeen/experiments/conformer_att_2022/librispeech_960/configs/baseline_960h_v2.py
"""

import copy, os

import numpy

from sisyphus import tk

from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.librispeech_960.chunkwise_attention_asr_config import (
    create_config,
    ConformerEncoderArgs,
    RNNDecoderArgs,
)

from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.additional_config import (
    apply_fairseq_init_to_conformer,
)
from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.librispeech_960.data import (
    build_training_datasets,
    build_test_dataset,
    build_chunkwise_training_datasets,
)
from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.default_tools import (
    RETURNN_EXE,
    RETURNN_ROOT,
    RETURNN_CPU_EXE,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.feature_extraction_net import (
    log10_net_10ms,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.pipeline import (
    training,
    search,
    get_average_checkpoint,
    get_best_checkpoint,
    search_single,
)
from i6_experiments.users.zeineldeen.models.lm import generic_lm
from i6_experiments.users.zeineldeen.models.lm.transformer_lm import TransformerLM
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960 import (
    ilm_helpers,
)
from i6_experiments.users.rossenbach.experiments.librispeech.kazuki_lm.experiment import (
    get_lm,
    ZeineldeenLM,
)

from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023 import tools_eval_funcs
from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023 import tools

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint
from i6_core.returnn.forward import ReturnnForwardJob

train_jobs_map = {}  # dict[str, ReturnnTrainJob]
train_job_avg_ckpt = {}
train_job_best_epoch = {}

BPE_10K = 10000
BPE_5K = 5000
BPE_1K = 1000

# --------------------------- LM --------------------------- #

lstm_10k_lm_opts = {
    "lm_subnet": generic_lm.libri_lstm_bpe10k_net,
    "lm_model": generic_lm.libri_lstm_bpe10k_model,
    "name": "lstm",
}

lstm_lm_opts_map = {
    BPE_10K: lstm_10k_lm_opts,
}

trafo_lm_net = TransformerLM(
    source="prev:output", num_layers=24, vocab_size=10025, use_as_ext_lm=True
)
trafo_lm_net.create_network()
trafo_10k_lm_opts = {
    "lm_subnet": trafo_lm_net.network.get_net(),
    "load_on_init_opts": {
        "filename": "/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/transfo_24_d00.4096_1024.sgd.lr1.8_heads/bk-net-model/network.023",
        "params_prefix": "",
        "load_if_prefix": "lm_output/",
    },
    "name": "trafo",
}

bpe5k_lm = get_lm("ls960_trafo24_bs3000_5ep_5kbpe")  # type: ZeineldeenLM
trafo_5k_lm_opts = {
    "lm_subnet": bpe5k_lm.combination_network,
    "load_on_init_opts": {
        "filename": get_best_checkpoint(
            bpe5k_lm.train_job, key="dev_score_output/output"
        ),
        "params_prefix": "",
        "load_if_prefix": "lm_output/",
    },
    "name": "trafo",
}

trafo_lm_opts_map = {
    BPE_10K: trafo_10k_lm_opts,
    BPE_5K: trafo_5k_lm_opts,
}


# ----------------------------------------------------------- #


def baseline():
    abs_name = os.path.abspath(__file__)
    prefix_name = os.path.basename(abs_name)[: -len(".py")]

    def get_test_dataset_tuples(bpe_size, ignored_data=None):
        test_dataset_tuples = {}
        for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
            if ignored_data and testset in ignored_data:
                continue
            test_dataset_tuples[testset] = build_test_dataset(
                testset,
                use_raw_features=True,
                bpe_size=bpe_size,
            )
        return test_dataset_tuples

    def run_train(
        exp_name,
        train_args,
        train_data,
        feature_extraction_net,
        num_epochs,
        recog_epochs,
        time_rqmt=168,
        **kwargs,
    ):
        exp_prefix = os.path.join(prefix_name, exp_name)
        returnn_config = create_config(
            training_datasets=train_data,
            **train_args,
            feature_extraction_net=feature_extraction_net,
            recog_epochs=recog_epochs,
        )
        train_job = training(
            exp_prefix,
            returnn_config,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            num_epochs=num_epochs,
            time_rqmt=time_rqmt,
        )
        return train_job

    def run_single_search(
        exp_name,
        train_data,
        search_args,
        checkpoint,
        feature_extraction_net,
        recog_dataset,
        recog_ref,
        mem_rqmt=8,
        time_rqmt=4,
        **kwargs,
    ):

        exp_prefix = os.path.join(prefix_name, exp_name)
        returnn_search_config = create_config(
            training_datasets=train_data,
            **search_args,
            feature_extraction_net=feature_extraction_net,
            is_recog=True,
        )
        search_single(
            exp_prefix,
            returnn_search_config,
            checkpoint,
            recognition_dataset=recog_dataset,
            recognition_reference=recog_ref,
            returnn_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
            mem_rqmt=mem_rqmt,
            time_rqmt=time_rqmt,
        )

    def run_lm_fusion(
        lm_type,
        exp_name,
        epoch,
        test_set_names,
        lm_scales,
        train_job,
        train_data,
        feature_net,
        bpe_size,
        args,
        beam_size=12,
        prior_scales=None,
        prior_type=None,
        mini_lstm_ckpt=None,
        length_norm=True,
        prior_type_name=None,
        coverage_scale=None,
        coverage_threshold=None,
        **kwargs,
    ):
        assert lm_type in ["lstm", "trafo"], "lm type should be lstm or trafo"

        if isinstance(lm_scales, float):
            lm_scales = [lm_scales]
        if prior_scales and isinstance(prior_scales, float):
            prior_scales = [prior_scales]
        if isinstance(test_set_names, str):
            test_set_names = [test_set_names]
        assert isinstance(test_set_names, list)

        if epoch == "avg":
            search_checkpoint = train_job_avg_ckpt[exp_name]
        elif epoch == "best":
            search_checkpoint = train_job_best_epoch[exp_name]
        else:
            assert isinstance(
                epoch, int
            ), "epoch must be either a defined integer or a string in {avg, best}."
            search_checkpoint = train_job.out_checkpoints[epoch]

        ext_lm_opts = (
            lstm_lm_opts_map[bpe_size]
            if lm_type == "lstm"
            else trafo_lm_opts_map[bpe_size]
        )

        time_rqmt = 1.0

        search_args = copy.deepcopy(args)

        if lm_type == "lstm":
            if beam_size > 128:
                search_args["batch_size"] = 4000 * 160

        if lm_type == "trafo":
            search_args["batch_size"] = 4000 * 160 if beam_size <= 32 else 2000 * 160
            time_rqmt = 2
            if beam_size > 50:
                time_rqmt = 3

        search_args["beam_size"] = beam_size
        if kwargs.get("batch_size", None):
            search_args["batch_size"] = kwargs["batch_size"]

        if not length_norm:
            search_args["decoder_args"].length_normalization = False

        if "decoder_args" in kwargs:
            for k, v in kwargs["decoder_args"].items():
                setattr(search_args["decoder_args"], k, v)

        scales = [(e,) for e in lm_scales]

        for test_set in test_set_names:

            if prior_scales:
                import itertools

                scales = itertools.product(lm_scales, prior_scales)

            for scale in scales:
                lm_scale = scale[0]
                prior_scale = scale[1] if len(scale) == 2 else None
                if prior_scale and prior_scale > lm_scale:
                    continue

                # External LM opts
                ext_lm_opts["lm_scale"] = lm_scale
                search_args["ext_lm_opts"] = ext_lm_opts

                # ILM opts
                if prior_scale:
                    ilm_opts = {
                        "scale": prior_scale,
                        "type": prior_type,
                        "ctx_dim": search_args[
                            "encoder_args"
                        ].enc_key_dim,  # this is needed for mini-lstm
                    }
                    # this is needed for mini-self-att
                    if hasattr(search_args["decoder_args"], "num_layers"):
                        ilm_opts["num_dec_layers"] = search_args[
                            "decoder_args"
                        ].num_layers
                        search_args["decoder_args"].create_ilm_decoder = True
                        search_args["decoder_args"].ilm_type = prior_type

                    ilm_opts.update(
                        kwargs.get("ilm_train_opts", {})
                    )  # example for FFN, etc

                    search_args["prior_lm_opts"] = ilm_opts
                    search_args["preload_from_files"] = {
                        "prior_lm": {
                            "filename": search_checkpoint,  # copy ASR decoder to be used as ILM decoder
                            "prefix": "prior_",
                        }
                    }
                    if prior_type == "mini_lstm" or prior_type == "ffn":
                        assert mini_lstm_ckpt, "Mini-LSTM checkpoint not set."
                        search_args["preload_from_files"].update(
                            {
                                "mini_lstm": {
                                    "filename": mini_lstm_ckpt,
                                    "prefix": "mini_",
                                }
                            }
                        )

                if prior_type_name is None:
                    prior_type_name = prior_type

                lm_desc = f"lm-scale-{lm_scale}"
                if prior_scale:
                    lm_desc += f"-prior-{prior_scale}-{prior_type_name}"
                lm_desc += f"-beam-{beam_size}"
                if length_norm is False:
                    lm_desc += "-woLenNorm"

                if coverage_scale and coverage_threshold:
                    assert isinstance(search_args["decoder_args"], RNNDecoderArgs)
                    search_args["decoder_args"].coverage_scale = coverage_scale
                    search_args["decoder_args"].coverage_threshold = coverage_threshold
                    lm_desc += (
                        f"_coverage-thre{coverage_threshold}-scale{coverage_scale}"
                    )

                name = f"{exp_name}/recog-{lm_type}-lm/ep-{epoch}/{lm_desc}/{test_set}"

                test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)

                run_single_search(
                    exp_name=name,
                    train_data=train_data,
                    search_args=search_args,
                    checkpoint=search_checkpoint,
                    feature_extraction_net=feature_net,
                    recog_dataset=test_dataset_tuples[test_set][0],
                    recog_ref=test_dataset_tuples[test_set][1],
                    time_rqmt=kwargs.get("time_rqmt", time_rqmt),
                )

    def run_search(
        exp_name,
        train_args,
        train_data,
        train_job,
        feature_extraction_net,
        num_epochs,
        search_args,
        recog_epochs,
        bpe_size,
        **kwargs,
    ):

        exp_prefix = os.path.join(prefix_name, exp_name)

        search_args = search_args if search_args is not None else copy.deepcopy(train_args)
        search_args['search_type'] = None

        returnn_search_config = create_config(
            training_datasets=train_data,
            **search_args,
            feature_extraction_net=feature_extraction_net,
            is_recog=True,
        )

        num_avg = kwargs.get("num_avg", 4)
        averaged_checkpoint = get_average_checkpoint(
            train_job,
            returnn_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
            num_average=num_avg,
        )
        if num_avg == 4:  # TODO: just for now to not break hashes
            train_job_avg_ckpt[exp_name] = averaged_checkpoint

        best_checkpoint = get_best_checkpoint(train_job)
        train_job_best_epoch[exp_name] = best_checkpoint

        if recog_epochs is None:
            default_recog_epochs = [10, 20, 40] + [80 * i for i in range(1, int(num_epochs / 80) + 1)]
            if num_epochs % 80 != 0:
                default_recog_epochs += [num_epochs]
        else:
            default_recog_epochs = recog_epochs

        test_dataset_tuples = get_test_dataset_tuples(
            bpe_size=bpe_size, ignored_data=kwargs.get("ignored_data", None))

        for ep in default_recog_epochs:
            search(
                exp_prefix + f"/recogs/ep-{ep}",
                returnn_search_config,
                train_job.out_checkpoints[ep],
                test_dataset_tuples,
                RETURNN_CPU_EXE,
                RETURNN_ROOT,
            )

        search(
            exp_prefix + "/default_last",
            returnn_search_config,
            train_job.out_checkpoints[num_epochs],
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
        )

        search(
            exp_prefix + "/default_best",
            returnn_search_config,
            best_checkpoint,
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
        )

        search(
            exp_prefix + f"/average_{num_avg}",
            returnn_search_config,
            averaged_checkpoint,
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
        )

    def run_exp(
        exp_name,
        train_args,
        feature_extraction_net=log10_net_10ms,
        num_epochs=300,
        search_args=None,
        recog_epochs=None,
        bpe_size=10000,
        partition_epoch=20,
        time_rqmt=168,
        train_fixed_alignment=None,
        cv_fixed_alignment=None,
        **kwargs,
    ):

        if train_fixed_alignment:
            assert cv_fixed_alignment, 'cv alignment is not set.'
            train_data = build_chunkwise_training_datasets(
                train_fixed_alignment=train_fixed_alignment,
                cv_fixed_alignment=cv_fixed_alignment,
                bpe_size=bpe_size,
                use_raw_features=True,
                partition_epoch=partition_epoch,
                epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
                link_speed_perturbation=train_args.get("speed_pert", True),
                seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
            )
        else:
            train_data = build_training_datasets(
                bpe_size=bpe_size,
                use_raw_features=True,
                partition_epoch=partition_epoch,
                epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
                link_speed_perturbation=train_args.get("speed_pert", True),
                seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
            )

        train_job = run_train(
            exp_name,
            train_args,
            train_data,
            feature_extraction_net,
            num_epochs,
            recog_epochs,
            time_rqmt=time_rqmt,
            **kwargs,
        )
        train_jobs_map[exp_name] = train_job

        run_search(
            exp_name,
            train_args,
            train_data,
            train_job,
            feature_extraction_net,
            num_epochs,
            search_args,
            recog_epochs,
            bpe_size=bpe_size,
            **kwargs,
        )
        return train_job, train_data

    def run_forward(
        exp_name,
        train_args,
        model_ckpt,
        hdf_layers=None,
        feature_extraction_net=log10_net_10ms,
        bpe_size=10000,
        time_rqmt=12,
        mem_rqmt=15,
        override_returnn_config=None,
        **kwargs,
    ):

        # build train, dev, and devtrain
        # - No speed pert
        # - Partition epoch 1
        # - No curr. learning

        train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            partition_epoch=1,
            epoch_wise_filter=None,
            link_speed_perturbation=False,
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )

        if train_args.get('dump_alignments_dataset', None):
            dump_dataset = train_args['dump_alignments_dataset']
        elif train_args.get('dump_ctc_dataset', None):
            dump_dataset = train_args['dump_ctc_dataset']
        else:
            raise Exception("No dump dataset specified.")

        assert dump_dataset in ['train', 'dev']

        exp_prefix = os.path.join(prefix_name, exp_name)

        if override_returnn_config:
            returnn_config = copy.deepcopy(override_returnn_config)
        else:
            returnn_config = create_config(
              training_datasets=train_data,
              **train_args,
              feature_extraction_net=feature_extraction_net,
            )

        forward_j = ReturnnForwardJob(
            model_checkpoint=Checkpoint(index_path=tk.Path(model_ckpt + '.index')),
            hdf_outputs=hdf_layers,
            returnn_config=returnn_config,
            returnn_python_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
            time_rqmt=time_rqmt,
            mem_rqmt=mem_rqmt,
            eval_mode=kwargs.get('do_eval', True),
            device=kwargs.get('device', 'gpu'),
        )

        forward_j.add_alias(exp_prefix + '/forward_hdf/' + dump_dataset)

        if hdf_layers is None:
            hdf_layers = ['output.hdf']

        for layer in hdf_layers:
            tk.register_output(
                os.path.join(exp_prefix, 'hdfs', dump_dataset),
                forward_j.out_hdf_files[layer])

        return forward_j.out_hdf_files


    def train_mini_lstm(
        exp_name,
        checkpoint,
        args,
        num_epochs=20,
        lr=8e-4,
        time_rqmt=4,
        l2=1e-4,
        name="mini_lstm",
        w_drop=False,
        use_dec_state=False,
        use_ffn=False,
        ffn_opts=None,
        **kwargs,
    ):
        if not w_drop:
            params_freeze_str = ilm_helpers.get_mini_lstm_params_freeze_str()
        else:
            if use_ffn:
                params_freeze_str = ilm_helpers.get_ffn_params_freeze_str_w_drop(
                    ffn_opts["num_ffn_layers"]
                )
            else:
                params_freeze_str = ilm_helpers.get_mini_lstm_params_freeze_str_w_drop()

        mini_lstm_args = copy.deepcopy(args)
        mini_lstm_args["batch_size"] = 20000 * 160
        mini_lstm_args["with_pretrain"] = False
        mini_lstm_args["lr"] = lr
        mini_lstm_args["allow_lr_scheduling"] = False
        mini_lstm_args["encoder_args"].with_ctc = False
        mini_lstm_args["keep_all_epochs"] = True  # keep everything
        mini_lstm_args["extra_str"] = params_freeze_str
        mini_lstm_args["preload_from_files"] = {
            "import": {
                "init_for_train": True,
                "ignore_missing": True,
                "filename": checkpoint,
            }
        }
        mini_lstm_args.update(kwargs)

        exp_prefix = os.path.join(prefix_name, exp_name, name)
        mini_lstm_train_data = build_training_datasets(
            bpe_size=10000,
            use_raw_features=True,
            epoch_wise_filter=None,
            link_speed_perturbation=False,  # depends only on text
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )
        returnn_config = create_config(
            training_datasets=mini_lstm_train_data,
            **mini_lstm_args,
            feature_extraction_net=log10_net_10ms,
        )

        inp = "s" if use_dec_state else "prev:target_embed"

        if use_ffn:
            x = inp
            activations = ffn_opts["activations"]
            for l in range(ffn_opts["num_ffn_layers"]):
                returnn_config.config["network"]["output"]["unit"][
                    "ffn_%02i" % (l + 1)
                ] = {
                    "class": "linear",
                    "n_out": ffn_opts["ffn_dims"][l],
                    "L2": l2,
                    "from": inp,
                    "activation": activations[l]
                    if activations and l < len(activations)
                    else None,
                }
                x = "ffn_%02i" % (l + 1)

            returnn_config.config["network"]["output"]["unit"]["att"] = {
                "class": "linear",
                "from": x,
                "activation": None,
                "n_out": mini_lstm_args["encoder_args"].enc_key_dim,
                "L2": l2,
            }
        else:
            # Mini-LSTM + FF

            returnn_config.config["network"]["output"]["unit"]["att_lstm"] = {
                "class": "rec",
                "unit": "nativelstm2",
                "from": inp,
                "n_out": 50,
            }

            returnn_config.config["network"]["output"]["unit"]["att"] = {
                "class": "linear",
                "from": "att_lstm",
                "activation": None,
                "n_out": mini_lstm_args["encoder_args"].enc_key_dim,
                "L2": l2,
            }

        train_job = training(
            exp_prefix,
            returnn_config,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            num_epochs=num_epochs,
            time_rqmt=time_rqmt,
        )
        return train_job

    def train_mini_self_att(
        exp_name,
        checkpoint,
        args,
        num_epochs=20,
        lr=8e-4,
        time_rqmt=4,
        name="mini_self_att",
        **kwargs,
    ):
        """
        Same idea as Mini-LSTM but use masked (mini-)self-attention models instead of cross attention.
        Note that each layer has its own (mini-)self-attention.

        In the case of transformer decoder, we want to replace cross-attention layers namely:
            transformer_decoder_{idx}_att_linear
        with masked self-attention models.
        """

        params_freeze_str = ilm_helpers.get_mini_self_att_params_freeze_str_w_drop(
            args["decoder_args"].num_layers
        )

        mini_self_att = copy.deepcopy(args)
        mini_self_att["batch_size"] = 20000 * 160  # TODO: does this fit now?
        mini_self_att["with_pretrain"] = False
        mini_self_att["lr"] = lr
        mini_self_att["allow_lr_scheduling"] = False
        mini_self_att["encoder_args"].with_ctc = False
        # mini_self_att['keep_all_epochs'] = True  # keep everything
        mini_self_att["extra_str"] = params_freeze_str
        mini_self_att["preload_from_files"] = {
            "import": {
                "init_for_train": True,
                "ignore_missing": True,
                "filename": checkpoint,
            }
        }
        if "decoder_args" in kwargs:
            assert isinstance(kwargs["decoder_args"], dict)
            for k, v in kwargs["decoder_args"].items():
                setattr(mini_self_att["decoder_args"], k, v)
            kwargs.pop("decoder_args")
        mini_self_att.update(kwargs)

        exp_prefix = os.path.join(prefix_name, exp_name, name)
        mini_self_att_train_data = build_training_datasets(
            bpe_size=10000,
            use_raw_features=True,
            epoch_wise_filter=None,
            link_speed_perturbation=False,  # depends only on text
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )

        # use masked self-att instead of cross-att with layer names having "ilm_" as prefix
        mini_self_att["decoder_args"].replace_cross_att_w_masked_self_att = True

        returnn_config = create_config(
            training_datasets=mini_self_att_train_data,
            **mini_self_att,
            feature_extraction_net=log10_net_10ms,
        )
        train_job = training(
            exp_prefix,
            returnn_config,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            num_epochs=num_epochs,
            time_rqmt=time_rqmt,
        )
        return train_job

    # --------------------------- General Settings --------------------------- #

    conformer_enc_args = ConformerEncoderArgs(
        num_blocks=12,
        input_layer="conv-6",
        att_num_heads=8,
        ff_dim=2048,
        enc_key_dim=512,
        conv_kernel_size=32,
        pos_enc="rel",
        dropout=0.1,
        att_dropout=0.1,
        l2=0.0001,
        use_sqrd_relu=True,
    )
    apply_fairseq_init_to_conformer(conformer_enc_args)
    conformer_enc_args.ctc_loss_scale = 1.0

    rnn_dec_args = RNNDecoderArgs()

    training_args = dict()
    training_args["speed_pert"] = True
    training_args['with_pretrain'] = False

    lstm_training_args = copy.deepcopy(training_args)
    lstm_training_args["batch_size"] = 15000 * 160  # frames * samples per frame

    lstm_dec_exp_args = copy.deepcopy(
        {
            **lstm_training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": rnn_dec_args,
        }
    )

    # --------------------------- Experiments --------------------------- #

    # Global attention baseline:
    #
    # dev-clean  2.28
    # dev-other  5.63
    # test-clean  2.48
    # test-other  5.71

    global_att_best_ckpt = "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/models-backup/best_att_100/avg_ckpt/epoch.2029"

    # from Albert:
    # with task=“train” and search_type=“end-of-chunk”, it would align on-the-fly
    # with task=“eval”, add a hdf-dump-layer, and search_type=“end-of-chunk”, you can dump it
    # with task=“train” and search_type default (None), it would train using a fixed alignment

    default_args = copy.deepcopy(lstm_dec_exp_args)
    default_args['learning_rates_list'] = list(numpy.linspace(8e-4, 1e-5, 60))
    default_args['retrain_checkpoint'] = global_att_best_ckpt
    default_args['chunk_size'] = 20
    default_args['chunk_step'] = 20 * 3 // 4
    default_args['search_type'] = 'end-of-chunk'  # align on-the-fly

    # TODO: train and align on the fly
    for chunk_size in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        args = copy.deepcopy(default_args)
        args['chunk_size'] = chunk_size
        chunk_step = chunk_size * 3 // 4
        args['chunk_step'] = chunk_step
        train_j, train_data = run_exp(f'base_chunkwise_att_chunk-{chunk_size}_step-{chunk_step}',
                train_args=args, num_epochs=60, epoch_wise_filter=None)

    # # TODO: tune LR with const + decay
    # for total_epochs in [2 * 20]:
    #     for start_lr in [1e-5, 1e-4]:
    #         for decay_pt in [3 / 4, 1 / 2] if start_lr <= 1e-5 else [1 / 2, 1 / 3]:
    #             end_lr = 1e-6
    #             args = copy.deepcopy(default_args)
    #             args['chunk_size'] = 20
    #             chunk_step = chunk_size * 3 // 4
    #             args['chunk_step'] = 15
    #             start_decay_pt = int(total_epochs * decay_pt)
    #             args['learning_rates_list'] = [start_lr] * start_decay_pt + list(
    #                 numpy.linspace(start_lr, end_lr, total_epochs - start_decay_pt))
    #             run_exp(
    #                 exp_name=f'base_chunkwise_att_chunk-{chunk_size}_step-{chunk_step}_linDecay{total_epochs}_{start_lr}_decayPt{decay_pt}',
    #                 train_args=args, num_epochs=total_epochs
    #             )

    # TODO: tune LR
    # for start_lr, end_lr in [(1e-5, 1e-6), (1e-5, 1e-5)]:
    #     args = copy.deepcopy(default_args)
    #     args['learning_rates_list'] = list(numpy.linspace(start_lr, end_lr, 60))
    #     run_exp(f'base_chunkwise_att_chunk-20_step-15_startLR-{start_lr}_endLR-{end_lr}',
    #             train_args=args, num_epochs=60, epoch_wise_filter=None)

    # --------------------------- Dumping Global Att Alignments --------------------------- #

    # TODO: dump alignment

    # dumped_align_map = {'train': {}, 'dev': {}}
    #
    # for dataset in ['train', 'dev']:
    #     for chunk_size in [1, 10, 20, 30, 40, 50]: # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    #         for chunk_step_factor in [1 / 2, 3 / 4, chunk_size]:
    #             args = copy.deepcopy(default_args)
    #             args['dump_alignments_dataset'] = dataset
    #             args['chunk_size'] = chunk_size
    #             chunk_step = max(1, int(chunk_size * chunk_step_factor))
    #             args['chunk_step'] = chunk_step
    #             res = run_forward(
    #                 f'dump_alignments_chunk-{chunk_size}_step-{chunk_step}',
    #                 train_args=args, model_ckpt=global_att_best_ckpt,
    #                 hdf_layers=[f'alignments-{dataset}.hdf'],
    #             )
    #             dumped_align_map[dataset][f'{chunk_size}_{chunk_step}'] = res[f'alignments-{dataset}.hdf']
    #
    # # TODO: without CTC
    # # TODO: without speed pert
    #
    # # train with fixed alignment
    # args = copy.deepcopy(default_args)
    # args['search_type'] = None
    # decay_pt = 60 * 1 // 3
    # args['learning_rates_list'] = [1e-5] * decay_pt + list(numpy.linspace(1e-5, 1e-6, 60 - decay_pt))
    # run_exp(
    #     exp_name=f'base_chunkwise_att_chunk-{20}_step-{15}_linDecay{40}_{1e-5}_decayPt{1/3}_fixed_align',
    #     train_args=args, num_epochs=60, train_fixed_alignment=dumped_align_map['train']['20_15'],
    #     cv_fixed_alignment=dumped_align_map['dev']['20_15'], epoch_wise_filter=None,
    # )
    # args['encoder_args'].with_ctc = False
    # run_exp(
    #     exp_name=f'base_chunkwise_att_chunk-{20}_step-{15}_linDecay{40}_{1e-5}_decayPt{1 / 3}_fixed_align_noCTC',
    #     train_args=args, num_epochs=60, train_fixed_alignment=dumped_align_map['train']['20_15'],
    #     cv_fixed_alignment=dumped_align_map['dev']['20_15'], epoch_wise_filter=None,
    # )
    # run_exp(
    #     exp_name=f'base_chunkwise_att_chunk-{20}_step-{15}_linDecay{40}_{1e-5}_decayPt{1 / 3}_fixed_align_noSpeedPert',
    #     train_args=args, num_epochs=60, train_fixed_alignment=dumped_align_map['train']['20_15'],
    #     cv_fixed_alignment=dumped_align_map['dev']['20_15'], epoch_wise_filter=None, speed_perturb=False,
    # )

    # --------------------------- Dumping CTC Alignments --------------------------- #

    def get_ctc_chunksyn_align_config(dataset_name, ctc_alignments, chunk_step, eoc_idx=0):
        from i6_experiments.common.setups.returnn import serialization
        config = ReturnnConfig({
            'extern_data': {
                "bpe_labels": {
                    "available_for_inference": True,
                    "dim": 10026,
                    "shape": (None,),
                    "sparse": True,
                },
                "bpe_labels_wo_blank": {
                    "available_for_inference": False,
                    "dim": 10025,  # without blank
                    "shape": (None,),
                    "sparse": True,
                }
            },
            "eval": {
                "class": "MetaDataset", "data_map": {"bpe_labels": ("hdf_dataset", "data")},
                    "datasets": {
                        "hdf_dataset": {
                            "class": "HDFDataset", "files": [ctc_alignments]
                        },
                    },
                "seq_order_control_dataset": "hdf_dataset",
            },
            "network": {
                "chunked_align": {
                    "class": "eval", "eval": tools_eval_funcs.get_chunked_align,
                    "out_type": tools_eval_funcs.get_chunked_align_out_type,
                    "from": "data:bpe_labels",
                    "eval_locals": {"chunk_step": chunk_step, "eoc_idx": eoc_idx},
                },
                "output": {
                    "class": "copy", "from": "chunked_align",
                    "target": "bpe_labels_wo_blank",
                }
            },
            'batch_size': 5000,
        })
        return serialization.get_serializable_config(config)


    # save time-sync -> chunk-sync converted alignments.
    ctc_align_wo_speed_pert = {
        'train': {}, 'dev': {},
    }

    for dataset in ['train', 'dev']:
        args = copy.deepcopy(default_args)
        args['dump_ctc_dataset'] = dataset
        args['batch_size'] *= 2

        # CTC alignment with blank.
        j = run_forward(
            f'dump_ctc_alignment_wo_speedPert',
            train_args=args, model_ckpt=global_att_best_ckpt,
            hdf_layers=[f'alignments-{dataset}.hdf'],
        )

        # convert w.r.t different chunk sizes and chunk steps
        for chunk_size in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            for chunk_step_factor in [1 / 2, 3 / 4, 1]:

                chunk_step = max(1, int(chunk_size * chunk_step_factor))

                ctc_chunk_sync_align = run_forward(
                    exp_name=f'ctc_chunk_sync_align_wo_speedPert_{chunk_size}-{chunk_step}',
                    train_args=args, model_ckpt=global_att_best_ckpt,
                    #hdf_layers=[f'alignments-{dataset}.hdf'],
                    override_returnn_config=get_ctc_chunksyn_align_config(
                        dataset, ctc_alignments=j[f'alignments-{dataset}.hdf'], chunk_step=chunk_step),
                    device='cpu',
                    do_eval=False,
                )
                ctc_align_wo_speed_pert[dataset][f'{chunk_size}_{chunk_step}'] = ctc_chunk_sync_align['output.hdf']


    # train with ctc chunk-sync alignment
    for total_epochs in [40]:
        for chunk_size in [1, 10, 20, 30, 40, 50]:
            for chunk_step_factor in [1 / 2, 3 / 4, 1]:

                train_args = copy.deepcopy(default_args)
                train_args['speed_pert'] = False  # no speed pert
                train_args['search_type'] = None  # fixed alignment

                decay_pt = total_epochs * 1 // 3

                train_args['chunk_size'] = chunk_size

                chunk_step = max(1, int(chunk_size * chunk_step_factor))
                train_args['chunk_step'] = chunk_step

                train_args['learning_rates_list'] = [1e-4] * decay_pt + list(numpy.linspace(1e-4, 1e-6, 40 - decay_pt))

                run_exp(
                    exp_name=f'base_chunkwise_att_chunk-{chunk_size}_step-{chunk_step}_linDecay{total_epochs}_{1e-4}_decayPt{1/3}_fixed_align',
                    train_args=train_args, num_epochs=total_epochs,
                    train_fixed_alignment=ctc_align_wo_speed_pert['train'][f'{chunk_size}_{chunk_step}'],
                    cv_fixed_alignment=ctc_align_wo_speed_pert['dev'][f'{chunk_size}_{chunk_step}'],
                    epoch_wise_filter=None, time_rqmt=72,
                )

                # check without CTC only for 1 exp
                if chunk_size == 20 and chunk_step == 3 / 4:
                    train_args['encoder_args'].with_ctc = False
                    run_exp(
                        exp_name=f'base_chunkwise_att_chunk-{chunk_size}_step-{chunk_step}_linDecay{total_epochs}_{1e-4}_decayPt{1/3}_fixed_align_noCTC',
                        train_args=train_args, num_epochs=total_epochs,
                        train_fixed_alignment=ctc_align_wo_speed_pert['train'][f'{chunk_size}_{chunk_step}'],
                        cv_fixed_alignment=ctc_align_wo_speed_pert['dev'][f'{chunk_size}_{chunk_step}'],
                        epoch_wise_filter=None, time_rqmt=72,
                    )
