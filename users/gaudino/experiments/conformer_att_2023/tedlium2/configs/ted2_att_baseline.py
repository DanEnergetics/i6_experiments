import copy, os

import numpy
from itertools import product

import sisyphus.toolkit as tk

from i6_experiments.users.gaudino.experiments.conformer_att_2023.tedlium2.attention_asr_config import (
    CTCDecoderArgs,
    create_config,
    ConformerEncoderArgs,
    TransformerDecoderArgs,
    RNNDecoderArgs,
    ConformerDecoderArgs,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.additional_config import (
    apply_fairseq_init_to_conformer,
    reset_params_init,
    apply_fairseq_init_to_transformer_decoder,
)
from i6_experiments.users.zeineldeen.experiments.conformer_att_2023.tedlium2.data import (
    build_training_datasets,
    build_test_dataset,
)
from i6_experiments.users.gaudino.experiments.conformer_att_2023.tedlium2.default_tools import (
    RETURNN_ROOT,
    RETURNN_CPU_EXE,
    SCTK_BINARY_PATH,
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
from i6_experiments.users.gaudino.models.asr.lm import tedlium_lm


train_jobs_map = {}  # dict[str, ReturnnTrainJob]
train_job_avg_ckpt = {}
train_job_best_epoch = {}

BPE_10K = 10000
BPE_5K = 5000
BPE_1K = 1000
BPE_500 = 500

# train:
# ------
# Seq-length 'data' Stats:
#   92973 seqs
#   Mean: 819.1473868757647
#   Std dev: 434.7168733027807
#   Min/max: 26 / 2049

# --------------------------- LM --------------------------- #

# LM data (runnnig words)
# trans 2250417 ~ 2.25M
# external: 12688261 ~ 12.7M
# Total: 14.9M

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

# bpe5k_lm = get_lm("ls960_trafo24_bs3000_5ep_5kbpe")  # type: ZeineldeenLM
# trafo_5k_lm_opts = {
#     "lm_subnet": bpe5k_lm.combination_network,
#     "load_on_init_opts": {
#         "filename": get_best_checkpoint(bpe5k_lm.train_job, key="dev_score_output/output"),
#         "params_prefix": "",
#         "load_if_prefix": "lm_output/",
#     },n
#     "name": "trafo",
# }

trafo_lm_opts_map = {
    BPE_10K: trafo_10k_lm_opts,
    # BPE_5K: trafo_5k_lm_opts,
}

prior_file = "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.2UG8sLxHNTMO/output/prior.txt"

# ----------------------------------------------------------- #


def compute_features_stats(
    output_dirname,
    feat_dim,
    bpe_size=10000,
    feature_extraction_net=log10_net_10ms,
    model_checkpoint=None,
    **kwargs,
):
    train_data = build_training_datasets(
        bpe_size=bpe_size,
        use_raw_features=True,
        epoch_wise_filter=None,
        link_speed_perturbation=False,
        seq_ordering="laplace:.1000",
        partition_epoch=1,
    )
    # Dump log-mel features into HDFDataset
    dump_features_config = {}
    dump_features_config["extern_data"] = train_data.extern_data
    dump_features_config["network"] = copy.deepcopy(feature_extraction_net)
    if model_checkpoint:
        dump_features_config["network"]["output"] = {
            "class": "hdf_dump",
            "from": "log_mel_features",
            "filename": "log_mel_features.hdf",
        }
    else:
        dump_features_config["network"]["output"] = {
            "class": "copy",
            "from": "log_mel_features",
        }
    dump_features_config["forward_batch_size"] = 20_000 * 80
    dump_features_config["use_tensorflow"] = True
    dump_features_config["eval"] = train_data.train.as_returnn_opts()
    from i6_core.returnn import ReturnnForwardJob, ReturnnConfig

    hdf_filename = "log_mel_features.hdf" if model_checkpoint else "output.hdf"

    dump_features_job = ReturnnForwardJob(
        returnn_config=ReturnnConfig(config=dump_features_config),
        returnn_python_exe=RETURNN_CPU_EXE,
        returnn_root=kwargs.get("returnn_root", RETURNN_ROOT),
        model_checkpoint=model_checkpoint,
        hdf_outputs=[hdf_filename] if model_checkpoint else [],
        device="cpu",
        mem_rqmt=15,
        time_rqmt=72,
        eval_mode=True if model_checkpoint else False,
    )
    dump_features_job.add_alias(
        f"ted2_stats/{output_dirname}/dump_train_log_mel_features"
    )
    tk.register_output(
        f"ted2_stats/{output_dirname}/log_mel_features.hdf",
        dump_features_job.out_hdf_files[hdf_filename],
    )

    # Extract features stats from HDFDataset
    extract_stats_returnn_config = ReturnnConfig(
        {
            "extern_data": {
                "data": {"dim": feat_dim},
            },
            "train": {
                "class": "HDFDataset",
                "files": [dump_features_job.out_hdf_files[hdf_filename]],
                "use_cache_manager": True,
            },
            "batch_size": 20_000 * 80,
            "use_tensorflow": True,
        }
    )
    from i6_core.returnn.dataset import ExtractDatasetMeanStddevJob

    extract_mean_stddev_job = ExtractDatasetMeanStddevJob(
        returnn_config=extract_stats_returnn_config,
        returnn_python_exe=RETURNN_CPU_EXE,
        returnn_root=kwargs.get("returnn_root", RETURNN_ROOT),
    )
    extract_mean_stddev_job.add_alias(
        f"ted2_stats/{output_dirname}/extract_mean_stddev"
    )

    tk.register_output(
        f"ted2_stats/{output_dirname}/mean_var", extract_mean_stddev_job.out_mean
    )
    tk.register_output(
        f"ted2_stats/{output_dirname}/std_dev_var", extract_mean_stddev_job.out_std_dev
    )
    tk.register_output(
        f"ted2_stats/{output_dirname}/mean_file", extract_mean_stddev_job.out_mean_file
    )
    tk.register_output(
        f"ted2_stats/{output_dirname}/std_dev_file",
        extract_mean_stddev_job.out_std_dev_file,
    )

    return (
        extract_mean_stddev_job.out_mean,
        extract_mean_stddev_job.out_std_dev,
        extract_mean_stddev_job.out_mean_file,
        extract_mean_stddev_job.out_std_dev_file,
    )


def conformer_baseline():
    abs_name = os.path.abspath(__file__)
    prefix_name = os.path.basename(abs_name)[: -len(".py")]

    def get_test_dataset_tuples(bpe_size):
        test_dataset_tuples = {}
        for testset in ["dev", "test"]:
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
            gpu_mem=kwargs.get("gpu_mem", 11),
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
        recog_bliss,
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
            recognition_bliss_corpus=recog_bliss,
            returnn_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
            mem_rqmt=mem_rqmt,
            time_rqmt=time_rqmt,
            **kwargs,
        )

    def run_decoding(
        exp_name,
        train_data,
        checkpoint,
        search_args,
        bpe_size,
        test_sets: list,
        feature_extraction_net=log10_net_10ms,
        time_rqmt: float = 1.0,
        remove_label=None,
        two_pass_rescore=False,
        **kwargs,
    ):
        test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)
        for test_set in test_sets:
            run_single_search(
                exp_name=exp_name + f"/recogs/{test_set}",
                train_data=train_data,
                search_args=search_args,
                checkpoint=checkpoint,
                feature_extraction_net=feature_extraction_net,
                recog_dataset=test_dataset_tuples[test_set][0],
                recog_ref=test_dataset_tuples[test_set][1],
                recog_bliss=test_dataset_tuples[test_set][2],
                time_rqmt=time_rqmt,
                remove_label=remove_label,
                # two_pass_rescore=two_pass_rescore,
                **kwargs,
            )

    def compute_ctc_prior(prior_exp_name, train_args, model_ckpt, bpe_size):
        exp_prefix = os.path.join(prefix_name, prior_exp_name)
        ctc_prior_train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            epoch_wise_filter=None,
            link_speed_perturbation=False,
            partition_epoch=1,
            seq_ordering="laplace:.1000",
        )
        returnn_config = create_config(
            training_datasets=ctc_prior_train_data,
            **train_args,
            feature_extraction_net=log10_net_10ms,
            with_pretrain=False,
        )
        returnn_config.config["network"]["output"] = {"class": "copy", "from": "ctc"}
        returnn_config.config["max_seq_length"] = -1
        from i6_core.returnn.extract_prior import ReturnnComputePriorJobV2

        prior_j = ReturnnComputePriorJobV2(
            model_checkpoint=model_ckpt,
            returnn_config=returnn_config,
            returnn_python_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
        )
        tk.register_output(
            exp_prefix + "/priors/ctc_prior_fix", prior_j.out_prior_txt_file
        )
        return prior_j.out_prior_txt_file

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
                    recog_bliss=test_dataset_tuples[test_set][2],
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

        search_args = search_args if search_args is not None else train_args

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
            default_recog_epochs = [40]
            default_recog_epochs += [80 * i for i in range(1, int(num_epochs / 80) + 1)]
            if num_epochs % 80 != 0:
                default_recog_epochs += [num_epochs]
        else:
            default_recog_epochs = recog_epochs

        test_dataset_tuples = get_test_dataset_tuples(bpe_size=bpe_size)

        run_only_avg = kwargs.get("run_only_avg", False)

        if not run_only_avg:
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
                use_sclite=True,
            )

        beam_size = search_args.get("beam_size", 12)
        if beam_size != 12:
            exp_prefix += f"_beam-{beam_size}"
        if search_args["decoder_args"].coverage_scale:
            exp_prefix += f"_coverage-thre{search_args['decoder_args'].coverage_threshold}-scale{search_args['decoder_args'].coverage_scale}"
        search(
            exp_prefix + f"/average_{num_avg}",
            returnn_search_config,
            averaged_checkpoint,
            test_dataset_tuples,
            RETURNN_CPU_EXE,
            RETURNN_ROOT,
            use_sclite=True,
        )

    def run_concat_seq_recog(
        exp_name,
        corpus_names,
        num,
        train_data,
        search_args,
        checkpoint,
        mem_rqmt=8,
        time_rqmt=1,
    ):
        exp_prefix = os.path.join(prefix_name, exp_name)

        from i6_experiments.users.zeineldeen.experiments.chunkwise_att_2023.concat_seqs import (
            ConcatDatasetSeqsJob,
            ConcatSeqsDataset,
            CreateConcatSeqsCTMAndSTMJob,
        )
        from i6_core.corpus.convert import CorpusToStmJob

        if isinstance(corpus_names, str):
            corpus_names = [corpus_names]
        assert isinstance(corpus_names, list)

        for corpus_name in corpus_names:
            test_datasets = get_test_dataset_tuples(bpe_size=BPE_1K)
            stm = CorpusToStmJob(
                bliss_corpus=test_datasets[corpus_name][2]
            ).out_stm_path
            tk.register_output(f"concat_seqs/{num}/orig_{corpus_name}_stm", stm)
            concat_dataset_seqs = ConcatDatasetSeqsJob(
                corpus_name="TED-LIUM-realease2", stm=stm, num=num, overlap_dur=None
            )
            tk.register_output(
                f"concat_seqs/{num}/{corpus_name}_stm", concat_dataset_seqs.out_stm
            )
            tk.register_output(
                f"concat_seqs/{num}/{corpus_name}_tags",
                concat_dataset_seqs.out_concat_seq_tags,
            )
            tk.register_output(
                f"concat_seqs/{num}/{corpus_name}_lens",
                concat_dataset_seqs.out_concat_seq_lens_py,
            )

            returnn_search_config = create_config(
                training_datasets=train_data,
                **search_args,
                feature_extraction_net=log10_net_10ms,
                is_recog=True,
            )

            returnn_concat_dataset = ConcatSeqsDataset(
                dataset=test_datasets[corpus_name][0].as_returnn_opts(),
                seq_tags=concat_dataset_seqs.out_concat_seq_tags,
                seq_lens_py=concat_dataset_seqs.out_orig_seq_lens_py,
            )

            _, search_words = search_single(
                os.path.join(exp_prefix, corpus_name),
                returnn_search_config,
                checkpoint,
                recognition_dataset=returnn_concat_dataset,
                recognition_reference=test_datasets[corpus_name][1],
                recognition_bliss_corpus=test_datasets[corpus_name][2],
                returnn_exe=RETURNN_CPU_EXE,
                returnn_root=RETURNN_ROOT,
                mem_rqmt=mem_rqmt,
                time_rqmt=time_rqmt,
                # no scoring
                use_sclite=False,
                use_returnn_compute_wer=False,
            )

            from i6_core.corpus.convert import CorpusToStmJob
            from i6_core.recognition.scoring import ScliteJob

            stm_file = concat_dataset_seqs.out_stm

            concat_ctm_and_stm_job = CreateConcatSeqsCTMAndSTMJob(
                recog_words_file=search_words,
                stm_py_file=concat_dataset_seqs.out_stm_py,
                stm_file=stm_file,
            )
            tk.register_output(
                exp_prefix + f"/{corpus_name}/sclite/stm",
                concat_ctm_and_stm_job.out_stm_file,
            )
            tk.register_output(
                exp_prefix + f"/{corpus_name}/sclite/ctm",
                concat_ctm_and_stm_job.out_ctm_file,
            )

            sclite_job = ScliteJob(
                ref=concat_ctm_and_stm_job.out_stm_file,
                hyp=concat_ctm_and_stm_job.out_ctm_file,
                sctk_binary_path=SCTK_BINARY_PATH,
            )
            tk.register_output(
                exp_prefix + f"/{corpus_name}/sclite/wer", sclite_job.out_wer
            )
            tk.register_output(
                exp_prefix + f"/{corpus_name}/sclite/report", sclite_job.out_report_dir
            )

    def run_exp(
        exp_name,
        train_args,
        feature_extraction_net=log10_net_10ms,
        num_epochs=300,
        search_args=None,
        recog_epochs=None,
        bpe_size=1000,
        partition_epoch=4,
        **kwargs,
    ):
        if train_args.get("retrain_checkpoint", None):
            assert (
                kwargs.get("epoch_wise_filter", None) is None
            ), "epoch_wise_filter should be disabled for retraining."
        train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
            link_speed_perturbation=train_args.get("speed_pert", True),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
            partition_epoch=partition_epoch,
            devtrain_subset=kwargs.get(
                "devtrain_subset", 507
            ),  # same as num of dev segments
        )
        train_job = run_train(
            exp_name,
            train_args,
            train_data,
            feature_extraction_net,
            num_epochs,
            recog_epochs,
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

        if kwargs.get("concat_recog_opts", None):
            ckpt_ = kwargs["concat_recog_opts"]["checkpoint"]
            if isinstance(ckpt_, str):
                assert ckpt_ in ["best", "avg"]
                if ckpt_ == "best":
                    concat_recog_ckpt = train_job_best_epoch[exp_name]
                else:
                    concat_recog_ckpt = train_job_avg_ckpt[exp_name]
            elif isinstance(ckpt_, int):
                concat_recog_ckpt = train_job.out_checkpoints[ckpt_]
            else:
                raise TypeError(
                    f"concat_recog_opts['checkpoint'] must be str or int, got {type(ckpt_)}"
                )
            concat_recog_search_args = kwargs["concat_recog_opts"].get(
                "search_args", None
            )
            search_args_ = copy.deepcopy(train_args)
            if concat_recog_search_args:
                search_args_.update(concat_recog_search_args)
            run_concat_seq_recog(
                exp_name=exp_name + f"_concat{kwargs['concat_recog_opts']['num']}",
                corpus_names=kwargs["concat_recog_opts"]["corpus_names"],
                num=kwargs["concat_recog_opts"]["num"],
                train_data=train_data,
                search_args=search_args_,
                checkpoint=concat_recog_ckpt,
            )

        return train_job, train_data

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
    )
    apply_fairseq_init_to_conformer(conformer_enc_args)
    conformer_enc_args.ctc_loss_scale = 1.0

    rnn_dec_args = RNNDecoderArgs()

    trafo_dec_args = TransformerDecoderArgs(
        num_layers=6,
        embed_dropout=0.1,
        label_smoothing=0.1,
        apply_embed_weight=True,
        pos_enc="rel",
    )
    apply_fairseq_init_to_transformer_decoder(trafo_dec_args)

    conformer_dec_args = ConformerDecoderArgs()
    apply_fairseq_init_to_conformer(conformer_dec_args)

    training_args = dict()

    # LR scheduling
    training_args["const_lr"] = [42, 100]  # use const LR during pretraining
    training_args["wup_start_lr"] = 0.0002
    training_args["wup"] = 20
    training_args["with_staged_network"] = True
    training_args["speed_pert"] = True

    trafo_training_args = copy.deepcopy(training_args)
    trafo_training_args["pretrain_opts"] = {
        "variant": 3,
        "initial_batch_size": 20000 * 160,
    }
    trafo_training_args["pretrain_reps"] = 5
    trafo_training_args["batch_size"] = 12000 * 160  # frames * samples per frame

    trafo_dec_exp_args = copy.deepcopy(
        {
            **trafo_training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": trafo_dec_args,
        }
    )

    conformer_dec_exp_args = copy.deepcopy(trafo_dec_exp_args)
    conformer_dec_exp_args["decoder_args"] = conformer_dec_args

    lstm_training_args = copy.deepcopy(training_args)
    lstm_training_args["pretrain_opts"] = {
        "variant": 3,
        "initial_batch_size": 22500 * 160,
    }
    lstm_training_args["pretrain_reps"] = 5
    lstm_training_args["batch_size"] = 15000 * 160  # frames * samples per frame

    lstm_dec_exp_args = copy.deepcopy(
        {
            **lstm_training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": rnn_dec_args,
        }
    )

    # --------------------------- Experiments --------------------------- #

    oclr_args = copy.deepcopy(lstm_dec_exp_args)
    oclr_args["oclr_opts"] = {
        "peak_lr": 9e-4,
        "final_lr": 1e-6,
    }
    oclr_args["encoder_args"].input_layer = "conv-6"
    oclr_args["encoder_args"].use_sqrd_relu = True
    oclr_args["max_seq_length"] = None

    # add hardcoded paths because DelayedFormat breaks hashes otherwise
    # _, _, global_mean, global_std = compute_features_stats(output_dirname="logmel_80", feat_dim=80)
    global_mean = tk.Path(
        "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.UHCZghp269OR/output/mean"
    )
    global_std = tk.Path(
        "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/dataset/ExtractDatasetMeanStddevJob.UHCZghp269OR/output/std_dev"
    )

    # step-based: 8.5/8.2
    # epoch-based: 8.6/8.2
    # for bpe_size in [BPE_1K]:
    #     for ep in [50 * 4]:
    #         for lr in [8e-4]:
    #             args = copy.deepcopy(oclr_args)
    #             args["oclr_opts"]["total_ep"] = ep
    #             args["oclr_opts"]["cycle_ep"] = int(0.45 * ep)
    #             args["oclr_opts"]["n_step"] = 1480
    #             args["oclr_opts"]["peak_lr"] = lr
    #             exp_name = f"base_bpe{bpe_size}_peakLR{lr}_ep{ep}"
    #             run_exp(
    #                 exp_name,
    #                 args,
    #                 num_epochs=ep,
    #                 epoch_wise_filter=None,
    #                 bpe_size=bpe_size,
    #                 partition_epoch=4,
    #                 devtrain_subset=3000,
    #             )

    # --------------------- V1 ---------------------
    def get_base_v1_args(lr, ep, enc_drop=0.1, pretrain_reps=3, use_legacy_stats=True):
        #  base_bpe1000_peakLR0.0008_ep200_globalNorm_epochOCLR_pre3_fixZoneout_encDrop0.1_woDepthConvPre
        # Average ckpt: 8.19/7.64 (50 epochs)
        # - Epoch-based OCLR with peak LR 8e-4
        # - EncDrop 0.1, fixed zoneout
        # - Pretrain 3, no depthwise conv pretrain
        # - Feature global normalization

        base_v1_args = copy.deepcopy(oclr_args)
        base_v1_args.pop("oclr_opts")
        cyc_ep = int(0.45 * ep)
        # Epoch-based OCLR
        base_v1_args["learning_rates_list"] = (
            list(numpy.linspace(lr / 10, lr, cyc_ep))
            + list(numpy.linspace(lr, lr / 10, cyc_ep))
            + list(numpy.linspace(lr / 10, 1e-6, ep - 2 * cyc_ep))
        )
        base_v1_args["global_stats"] = {
            "mean": global_mean,
            "stddev": global_std,
            "use_legacy_version": use_legacy_stats,
        }
        base_v1_args["pretrain_reps"] = pretrain_reps
        base_v1_args["pretrain_opts"]["ignored_keys_for_reduce_dim"] = [
            "conv_kernel_size"
        ]
        base_v1_args["encoder_args"].dropout = enc_drop
        base_v1_args["encoder_args"].dropout_in = enc_drop
        base_v1_args["encoder_args"].att_dropout = enc_drop
        base_v1_args["encoder_args"].num_blocks = 12
        base_v1_args["encoder_args"].mhsa_weight_dropout = 0.1
        base_v1_args["encoder_args"].ff_weight_dropout = 0.1
        base_v1_args["encoder_args"].conv_weight_dropout = 0.1

        base_v1_args["decoder_args"].use_zoneout_output = True
        base_v1_args["decoder_args"].embed_dim = 256
        base_v1_args["decoder_args"].att_dropout = 0.0

        exp_name = f"base_bpe1000_peakLR{lr}_ep{ep}_globalNorm_epochOCLR_pre{pretrain_reps}_fixZoneout_encDrop{enc_drop}_woDepthConvPre_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12"
        return base_v1_args, exp_name

    # baseline v1
    # WERs: 8.2/7.6
    base_v1_args, exp_name = get_base_v1_args(8e-4, 50 * 4)
    # run_exp(
    #     exp_name,
    #     base_v1_args,
    #     num_epochs=50 * 4,
    #     epoch_wise_filter=None,
    #     bpe_size=BPE_1K,
    #     partition_epoch=4,
    #     devtrain_subset=3000,
    # )

    # monotonic att weights loss
    # for scale in [1e-3, 5e-3, 1e-2]:
    #     args, exp_name = get_base_v1_args(8e-4, 50 * 4)
    #     args["decoder_args"].monotonic_att_weights_loss_scale = scale
    #     run_exp(
    #         exp_name + f"_monotonicAttLoss{scale}",
    #         args,
    #         num_epochs=50 * 4,
    #         epoch_wise_filter=None,
    #         bpe_size=BPE_1K,
    #         partition_epoch=4,
    #     )

    # for scale in [1e-1, 1e-2]:
    #     args, exp_name = get_base_v1_args(8e-4, 50 * 4)
    #     args["decoder_args"].att_weights_variance_loss_scale = scale
    #     run_exp(
    #         exp_name + f"_attWeightsVarLoss{scale}",
    #         args,
    #         num_epochs=50 * 4,
    #         epoch_wise_filter=None,
    #         bpe_size=BPE_1K,
    #         partition_epoch=4,
    #     )

    # TODO: longer training with more regularization
    # TODO: embed dropout?
    # for num_blocks in [12]:
    #     for ep in [100 * 4]:
    #         for lr in [8e-4]:
    #             for weight_drop in [0.1]:
    #                 for enc_drop in [0.1, 0.15, 0.2]:
    #                     base_v1_args, exp_name = get_base_v1_args(lr, ep, enc_drop=enc_drop)
    #                     args = copy.deepcopy(base_v1_args)
    #
    #                     args["encoder_args"].num_blocks = num_blocks
    #                     args["encoder_args"].mhsa_weight_dropout = weight_drop
    #                     args["encoder_args"].ff_weight_dropout = weight_drop
    #                     args["encoder_args"].conv_weight_dropout = weight_drop
    #
    #                     name = exp_name + f"_weightDrop{weight_drop}_numBlocks{num_blocks}"
    #                     run_exp(
    #                         name,
    #                         args,
    #                         num_epochs=ep,
    #                         epoch_wise_filter=None,
    #                         bpe_size=BPE_1K,
    #                         partition_epoch=4,
    #                     )

    ep = 100 * 4
    lr = 8e-4
    enc_drop = 0.15

    base_v1_args, exp_name = get_base_v1_args(lr, ep, enc_drop=enc_drop)
    args = copy.deepcopy(base_v1_args)

    # train base model
    name = exp_name
    train_j, train_data = run_exp(
        name,
        args,
        num_epochs=ep,
        epoch_wise_filter=None,
        bpe_size=BPE_1K,
        partition_epoch=4,
    )

    # att + ctc opts
    search_args = copy.deepcopy(args)
    for scales in [(0.4, 0.6), (0.7, 0.3), (0.85, 0.15)]:
        for beam_size in [12]:
            search_args["beam_size"] = beam_size
            att_scale, ctc_scale = scales
            search_args["decoder_args"] = CTCDecoderArgs(
                add_att_dec=True,
                att_scale=att_scale,
                ctc_scale=ctc_scale,
                att_masking_fix=True,
                target_dim=1057,
                target_embed_dim=256,
            )
            run_decoding(
                f"opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}",
                train_data,
                checkpoint=train_job_avg_ckpt[name],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
            )

    # ctc greedy decoding
    search_args["decoder_args"] = CTCDecoderArgs(target_dim=1057)

    run_decoding(
        f"ctc_greedy",
        train_data,
        checkpoint=train_job_avg_ckpt[name],
        search_args=search_args,
        bpe_size=BPE_1K,
        test_sets=["dev"],
        remove_label={"<s>", "<blank>"},
        use_sclite=True,
    )

    tedlium_lm_opts = {
        "lm_subnet": tedlium_lm.tedlium_lm_net,
        "load_on_init_opts": tedlium_lm.tedlium_lm_load_on_init,
        "name": "trafo",
    }

    # ctc + lm decoding
    # for beam_size, ctc_scale, lm_scale in product([12, 32], [1.0], [0.3, 0.4]):
    #     search_args = copy.deepcopy(args)
    #     search_args["beam_size"] = beam_size
    #     lm_type = "trafo"
    #     ext_lm_opts = tedlium_lm_opts
    #     search_args["decoder_args"] = CTCDecoderArgs(
    #         add_att_dec=False,
    #         ctc_scale=ctc_scale,
    #         add_ext_lm=True,
    #         lm_type=lm_type,
    #         ext_lm_opts=ext_lm_opts,
    #         lm_scale=lm_scale,
    #         target_dim=1057,
    #         target_embed_dim=256,
    #     )
    #     run_decoding(
    #         f"opts_ctc{ctc_scale}_lm{lm_scale}_beam{beam_size}",
    #         train_data,
    #         checkpoint=train_job_avg_ckpt[name],
    #         search_args=search_args,
    #         bpe_size=BPE_1K,
    #         test_sets=["dev"],
    #         remove_label={"<s>", "<blank>"},
    #         use_sclite=True,
    #     )

    # ctc + att + lm decoding
    # for beam_size, scales, lm_scale in product([48], [(0.7, 0.3)], [0.3, 0.35, 0.4]):
    #     search_args = copy.deepcopy(args)
    #     search_args["beam_size"] = beam_size
    #     att_scale, ctc_scale = scales
    #     # prior_scale = 0.3
    #     lm_type = "lstm"
    #     ext_lm_opts = tedlium_lm_opts
    #     search_args["decoder_args"] = CTCDecoderArgs(
    #         add_att_dec=True,
    #         att_scale=att_scale,
    #         ctc_scale=ctc_scale,
    #         att_masking_fix=True,
    #         add_ext_lm=True,
    #         lm_type=lm_type,
    #         ext_lm_opts=ext_lm_opts,
    #         lm_scale=lm_scale,
    #     )
    #     run_decoding(
    #         f"opts_ctc{ctc_scale}_att{att_scale}_lm{lm_scale}_beam{beam_size}",
    #         train_data,
    #         checkpoint=train_job_avg_ckpt[name],
    #         search_args=search_args,
    #         bpe_size=BPE_1K,
    #         test_sets=["dev"],
    #         remove_label={"<s>", "<blank>"},
    #         use_sclite=True,
    #     )

    # compute ctc prior
    prior_args = copy.deepcopy(args)
    prior_args["decoder_args"] = CTCDecoderArgs()
    # prior_file = compute_ctc_prior(
    #     name, prior_args, train_job_avg_ckpt[name], bpe_size=BPE_1K
    # )
    prior_file = "/u/luca.gaudino/setups/2023-10-15--conformer-no-app/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.2UG8sLxHNTMO/output/prior.txt"

    # try prior correction
    for scales in [(0.7, 0.3, 0.4)]:
        for beam_size in [12, 32, 64]:
            search_args["beam_size"] = beam_size
            search_args["ctc_log_prior_file"] = prior_file
            att_scale, ctc_scale, prior_scale = scales
            search_args["decoder_args"] = CTCDecoderArgs(
                add_att_dec=True,
                att_scale=att_scale,
                ctc_scale=ctc_scale,
                att_masking_fix=True,
                target_dim=1057,
                target_embed_dim=256,
                ctc_prior_correction=True,
                prior_scale=prior_scale,
            )
            run_decoding(
                f"opts_ctc{ctc_scale}_att{att_scale}_beam{beam_size}_prior{prior_scale}",
                train_data,
                checkpoint=train_job_avg_ckpt[name],
                search_args=search_args,
                bpe_size=BPE_1K,
                test_sets=["dev"],
                remove_label={"<s>", "<blank>"},
                use_sclite=True,
            )

    # train only CTC
    only_ctc_name = name + "_onlyCTC"
    only_ctc_args = copy.deepcopy(args)
    only_ctc_args["decoder_args"].ce_loss_scale = 0.0
    _, train_data = run_exp(
        only_ctc_name,
        only_ctc_args,
        num_epochs=ep,
        epoch_wise_filter=None,
        bpe_size=BPE_1K,
        partition_epoch=4,
        search_args={"ctc_decode": True, "ctc_blank_idx": 1057, **only_ctc_args},
    )
    prior_file_ctc_only = compute_ctc_prior(
        only_ctc_name, prior_args, train_job_avg_ckpt[only_ctc_name], bpe_size=BPE_1K
    )
    # best checkpoint path "/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.9o6iL7eblZwa/output/models/epoch.400"

    # train scale CTC
    scale_ctc_name = name + "_ctcScale0.3"
    scale_ctc_args = copy.deepcopy(args)
    scale_ctc_args["encoder_args"].ctc_loss_scale = 0.3 / 0.7  # AED scale is 1.0
    _, train_data = run_exp(
        scale_ctc_name,
        scale_ctc_args,
        num_epochs=ep,
        epoch_wise_filter=None,
        bpe_size=BPE_1K,
        partition_epoch=4,
    )

    # att + ctc opts
    prior_file_scale = compute_ctc_prior(
        scale_ctc_name, prior_args, train_job_avg_ckpt[scale_ctc_name], bpe_size=BPE_1K
    )
    search_args = copy.deepcopy(args)
    for scales in [(0.7, 0.3), (0.65, 0.35)]:
        for beam_size in [12, 32, 64]:
            for prior_scale in [0.4]:
                search_args["beam_size"] = beam_size
                search_args["ctc_log_prior_file"] = prior_file_scale
                att_scale, ctc_scale = scales
                search_args["decoder_args"] = CTCDecoderArgs(
                    add_att_dec=True,
                    att_scale=att_scale,
                    ctc_scale=ctc_scale,
                    att_masking_fix=True,
                    target_dim=1057,
                    target_embed_dim=256,
                    ctc_prior_correction=True,
                    prior_scale=prior_scale,
                )
                run_decoding(
                    f"model_ctc_0.3_att_0.7/opts_ctc{ctc_scale}_att{att_scale}_prior{prior_scale}_beam{beam_size}",
                    train_data,
                    checkpoint=train_job_avg_ckpt[scale_ctc_name],
                    search_args=search_args,
                    bpe_size=BPE_1K,
                    test_sets=["dev", "test"],
                    remove_label={"<s>", "<blank>"},
                    use_sclite=True,
                )

    # separate encoders
    # some model + ctc only
    for beam_size, scales, prior_scale in product(
        [32, 64], [(0.8, 0.2)], [0.4]
    ):
        search_args["beam_size"] = beam_size
        search_args["ctc_log_prior_file"] = prior_file_ctc_only
        att_scale, ctc_scale = scales
        search_args["decoder_args"] = CTCDecoderArgs(
            add_att_dec=True,
            att_scale=att_scale,
            ctc_scale=ctc_scale,
            att_masking_fix=True,
            target_dim=1057,
            target_embed_dim=256,
            ctc_prior_correction=True,
            prior_scale=prior_scale,
        )
        search_args[
            "second_encoder_ckpt"
        ] = "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.9o6iL7eblZwa/output/models/epoch.400"
        # search_args["second_encoder_ckpt"] = train_job_avg_ckpt[only_ctc_name]
        run_decoding(
            f"model_aed1.0ctc0.3__ctc_only/opts_ctc{ctc_scale}_att{att_scale}_prior{prior_scale}_beam{beam_size}",
            train_data,
            checkpoint=train_job_avg_ckpt[scale_ctc_name],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev", "test"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
        )

    if True:
        # ctc greedy of separate encoder as sanity check
        search_args["beam_size"] = beam_size
        search_args["ctc_log_prior_file"] = prior_file_ctc_only
        att_scale, ctc_scale = scales
        search_args["decoder_args"] = CTCDecoderArgs(target_dim=1057)
        search_args[
            "second_encoder_ckpt"
        ] = "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.9o6iL7eblZwa/output/models/epoch.400"
        # search_args["second_encoder_ckpt"] = train_job_avg_ckpt[only_ctc_name]
        run_decoding(
            f"model_base__ctc_only/ctc_greedy",
            train_data,
            checkpoint=train_job_avg_ckpt[name],
            search_args=search_args,
            bpe_size=BPE_1K,
            test_sets=["dev"],
            remove_label={"<s>", "<blank>"},
            use_sclite=True,
        )

    # # additional trainings
    for scales in [(1.0, 0.2), (0.7, 0.3), (0.9, 0.1)]:
        # for scales in []:
        # train scale CTC
        att_scale, ctc_scale = scales
        # keep wrong name
        # name = name + "_weightDrop0.1_decAttDrop0.0_embedDim256_numBlocks12"
        scale_ctc_name = name + f"_ce{att_scale}_ctc{ctc_scale}"
        scale_ctc_args = copy.deepcopy(args)
        scale_ctc_args["encoder_args"].ctc_loss_scale = ctc_scale  # AED scale is 1.0
        scale_ctc_args["decoder_args"].ce_loss_scale = att_scale
        _, train_data = run_exp(
            scale_ctc_name,
            scale_ctc_args,
            num_epochs=ep,
            epoch_wise_filter=None,
            bpe_size=BPE_1K,
            partition_epoch=4,
        )

        # att + ctc opts
        prior_file_scale = compute_ctc_prior(
            scale_ctc_name,
            prior_args,
            train_job_avg_ckpt[scale_ctc_name],
            bpe_size=BPE_1K,
        )

    # train att only
    for pretrain_reps in [4, 5]:
        # train base model
        att_only_args, exp_name = get_base_v1_args(
            lr, ep, pretrain_reps=pretrain_reps, enc_drop=enc_drop
        )
        att_only_args["encoder_args"].with_ctc = False
        exp_name = exp_name + "_noctc"
        train_j, train_data = run_exp(
            exp_name,
            att_only_args,
            num_epochs=ep,
            epoch_wise_filter=None,
            bpe_size=BPE_1K,
            partition_epoch=4,
        )

    # att only with curriculum learning
    att_only_args, exp_name = get_base_v1_args(
        lr, ep, pretrain_reps=pretrain_reps, enc_drop=enc_drop
    )
    att_only_args["encoder_args"].with_ctc = False
    exp_name = exp_name + "_noctc_currL1"
    train_j, train_data = run_exp(
        exp_name,
        att_only_args,
        num_epochs=ep,
        epoch_wise_filter=[(1, 2, 400), (2, 4, 800)],
        bpe_size=BPE_1K,
        partition_epoch=4,
    )


    # more pretrain reps
    scale_ctc_args, exp_name = get_base_v1_args(
        lr, ep, pretrain_reps=4, enc_drop=enc_drop
    )
    att_scale, ctc_scale = (0.9, 0.1)
    scale_ctc_name = exp_name + f"_ce{att_scale}_ctc{ctc_scale}"
    scale_ctc_args["encoder_args"].ctc_loss_scale = ctc_scale  # AED scale is 1.0
    scale_ctc_args["decoder_args"].ce_loss_scale = att_scale
    _, train_data = run_exp(
        scale_ctc_name,
        scale_ctc_args,
        num_epochs=ep,
        epoch_wise_filter=None,
        bpe_size=BPE_1K,
        partition_epoch=4,
    )

    # att + ctc opts
    prior_file_scale = compute_ctc_prior(
        scale_ctc_name,
        prior_args,
        train_job_avg_ckpt[scale_ctc_name],
        bpe_size=BPE_1K,
    )





    # --- old code ---

    # # lower lr -> did not converge
    # scale_ctc_args, exp_name = get_base_v1_args(7e-4, ep, enc_drop=enc_drop)
    # att_scale, ctc_scale = (0.9, 0.1)
    # scale_ctc_name = exp_name + f"_ce{att_scale}_ctc{ctc_scale}"
    # scale_ctc_args["encoder_args"].ctc_loss_scale = ctc_scale  # AED scale is 1.0
    # scale_ctc_args["decoder_args"].ce_loss_scale = att_scale
    # _, train_data = run_exp(
    #     scale_ctc_name,
    #     scale_ctc_args,
    #     num_epochs=ep,
    #     epoch_wise_filter=None,
    #     bpe_size=BPE_1K,
    #     partition_epoch=4,
    # )
    #
    # # att + ctc opts
    # prior_file_scale = compute_ctc_prior(
    #     scale_ctc_name,
    #     prior_args,
    #     train_job_avg_ckpt[scale_ctc_name],
    #     bpe_size=BPE_1K,
    # )

    # for num_blocks in [14]:
    #     for ep in [100 * 4]:
    #         for lr in [8e-4]:
    #             for target_embed_dim in [256]:
    #                 for att_drop in [0.0]:
    #                     for weight_drop in [0.1]:
    #                         for enc_drop in [0.15]:
    #                             base_v1_args, exp_name = get_base_v1_args(lr, ep, enc_drop=enc_drop)
    #                             args = copy.deepcopy(base_v1_args)
    #                             search_args = copy.deepcopy(base_v1_args)
    #                             search_args["recursion_limit"] = 6000
    #
    #                             args["encoder_args"].num_blocks = num_blocks
    #                             args["encoder_args"].mhsa_weight_dropout = weight_drop
    #                             args["encoder_args"].ff_weight_dropout = weight_drop
    #                             args["encoder_args"].conv_weight_dropout = weight_drop
    #
    #                             args["decoder_args"].embed_dim = target_embed_dim
    #                             args["decoder_args"].att_dropout = att_drop
    #
    #                             args["batch_size"] *= 2
    #                             args["pretrain_opts"]["initial_batch_size"] *= 2
    #                             args["accum_grad"] = 1
    #
    #                             name = (
    #                                 exp_name
    #                                 + f"_weightDrop{weight_drop}_decAttDrop{att_drop}_embedDim{target_embed_dim}_numBlocks{num_blocks}_bs30k"
    #                             )
    #                             run_exp(
    #                                 name,
    #                                 args,
    #                                 num_epochs=ep,
    #                                 epoch_wise_filter=None,
    #                                 bpe_size=BPE_1K,
    #                                 partition_epoch=4,
    #                                 gpu_mem=24,
    #                                 search_args=search_args,
    #                             )
    #
    # for num_blocks in [16]:
    #     for ep in [100 * 4]:
    #         for lr in [8e-4]:
    #             for target_embed_dim in [256]:
    #                 for att_drop in [0.0]:
    #                     for weight_drop in [0.1]:
    #                         for enc_drop in [0.15]:
    #                             base_v1_args, exp_name = get_base_v1_args(lr, ep, enc_drop=enc_drop)
    #                             args = copy.deepcopy(base_v1_args)
    #                             search_args = copy.deepcopy(base_v1_args)
    #
    #                             args["encoder_args"].num_blocks = num_blocks
    #                             args["encoder_args"].mhsa_weight_dropout = weight_drop
    #                             args["encoder_args"].ff_weight_dropout = weight_drop
    #                             args["encoder_args"].conv_weight_dropout = weight_drop
    #
    #                             args["encoder_args"].enc_key_dim = 384
    #                             args["encoder_args"].att_num_heads = 6
    #                             args["encoder_args"].ff_dim = 1536
    #
    #                             args["decoder_args"].embed_dim = target_embed_dim
    #                             args["decoder_args"].att_dropout = att_drop
    #
    #                             args["batch_size"] *= 2
    #                             args["accum_grad"] = 1
    #
    #                             # modify pretrain
    #                             args["pretrain_opts"]["initial_batch_size"] *= 2
    #                             args["pretrain_opts"]["initial_dim_factor"] = 256 / 384
    #
    #                             name = (
    #                                 exp_name
    #                                 + f"_weightDrop{weight_drop}_decAttDrop{att_drop}_embedDim{target_embed_dim}_numBlocks{num_blocks}_dim384_bs30k"
    #                             )
    #                             run_exp(
    #                                 name,
    #                                 args,
    #                                 num_epochs=ep,
    #                                 epoch_wise_filter=None,
    #                                 bpe_size=BPE_1K,
    #                                 partition_epoch=4,
    #                                 gpu_mem=24,
    #                                 search_args=search_args,
    #                             )
