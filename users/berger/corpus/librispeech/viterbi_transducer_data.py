from typing import Dict, List
import copy

from i6_core import corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.returnn.hdf import BlissToPcmHDFJob
from i6_experiments.users.berger.args.returnn.dataset import MetaDatasetBuilder
from i6_experiments.users.berger.systems.dataclasses import AlignmentData, FeatureType
from . import data
from ..general import BasicSetupData
from sisyphus import tk
from ...args.jobs.rasr_init_args import get_feature_extraction_args_16kHz
from ...helpers import build_rasr_feature_hdfs


def get_librispeech_data(
    returnn_root: tk.Path,
    returnn_python_exe: tk.Path,
    alignments: Dict[str, AlignmentData],
    rasr_binary_path: tk.Path,
    rasr_arch: str = "linux-x86_64-standard",
    train_key: str = "train-other-960",
    dev_keys: List[str] = ["dev-clean", "dev-other"],
    test_keys: List[str] = ["test-clean", "test-other"],
    add_unknown: bool = False,
    augmented_lexicon: bool = True,
    use_wei_lexicon: bool = False,
    feature_type: FeatureType = FeatureType.SAMPLES,
    **kwargs,
) -> BasicSetupData:
    # ********** Data inputs **********

    (
        train_data_inputs,
        dev_data_inputs,
        test_data_inputs,
    ) = data.get_data_inputs(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        ctc_lexicon=True,
        use_augmented_lexicon=augmented_lexicon,
        use_wei_lexicon=use_wei_lexicon,
        add_all_allophones=True,
        audio_format="wav",  # Note: OGGZip dataset lead to length mismatches between features and alignment
        add_unknown_phoneme_and_mapping=add_unknown,
        **kwargs,
    )

    # ********** Train data **********

    train_corpus_object = copy.deepcopy(train_data_inputs[train_key].corpus_object)
    train_corpus = train_corpus_object.corpus_file
    train_lexicon = train_data_inputs[train_key].lexicon.filename
    assert train_corpus is not None

    if (not add_unknown and not augmented_lexicon) or use_wei_lexicon:
        train_corpus = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
            train_corpus,
            train_lexicon,
            all_unknown=False,
        ).out_corpus
        train_corpus_object.corpus_file = train_corpus

    if feature_type == FeatureType.GAMMATONE_16K or feature_type == FeatureType.GAMMATONE_CACHED_16K:
        gt_args = get_feature_extraction_args_16kHz()["gt"]
        train_feature_hdf = build_rasr_feature_hdfs(
            train_corpus_object,
            split=train_data_inputs[train_key].concurrent,
            feature_type="gt",
            feature_extraction_args=gt_args,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
        )
    elif feature_type == FeatureType.SAMPLES:
        train_feature_hdf_job = BlissToPcmHDFJob(
            train_corpus,
            rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
            returnn_root=returnn_root,
        )
        train_feature_hdf_job.rqmt["mem"] = 8
        train_feature_hdf_job.rqmt["time"] = 24
        train_feature_hdf = [train_feature_hdf_job.out_hdf]
    else:
        raise NotImplementedError
    train_alignment_hdf = alignments[f"{train_key}_align"].get_hdf(
        returnn_python_exe=returnn_python_exe, returnn_root=returnn_root
    )

    train_dataset_builder = MetaDatasetBuilder()
    train_dataset_builder.add_hdf_dataset(
        name="data",
        hdf_files=train_feature_hdf,
        key_mapping={"data": "data"},
    )

    train_dataset_builder.add_hdf_dataset(
        name="classes",
        hdf_files=train_alignment_hdf,
        dataset_config={
            "partition_epoch": 20,
            "seq_ordering": "laplace:.1000",
        },
        key_mapping={"data": "classes"},
        control=True,
    )
    train_data_config = train_dataset_builder.get_dict()

    # ********** CV data **********

    cv_data_inputs = copy.deepcopy(dev_data_inputs)

    if (not add_unknown and not augmented_lexicon) or use_wei_lexicon:
        for corpus_object in [cv_data_inputs[key].corpus_object for key in dev_keys]:
            assert corpus_object.corpus_file is not None
            corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
                corpus_object.corpus_file,
                train_lexicon,
                all_unknown=False,
            ).out_corpus

    if feature_type == FeatureType.GAMMATONE_16K or feature_type == FeatureType.GAMMATONE_CACHED_16K:
        gt_args = get_feature_extraction_args_16kHz()["gt"]
        cv_feature_hdfs = sum(
            [
                build_rasr_feature_hdfs(
                    data_input.corpus_object,
                    split=data_input.concurrent,
                    feature_type="gt",
                    feature_extraction_args=gt_args,
                    returnn_python_exe=returnn_python_exe,
                    returnn_root=returnn_root,
                    rasr_binary_path=rasr_binary_path,
                    rasr_arch=rasr_arch,
                    single_hdf=True,
                )
                for key, data_input in cv_data_inputs.items()
                if key in dev_keys
            ],
            [],
        )
    elif feature_type == FeatureType.SAMPLES:
        cv_feature_hdfs = [
            BlissToPcmHDFJob(
                data_input.corpus_object.corpus_file,
                rounding=BlissToPcmHDFJob.RoundingScheme.rasr_compatible,
                returnn_root=returnn_root,
            ).out_hdf
            for key, data_input in cv_data_inputs.items()
            if key in dev_keys
        ]
    else:
        raise NotImplementedError
    cv_alignment_hdfs = [
        alignments[f"{dev_key}_align"].get_hdf(returnn_python_exe=returnn_python_exe, returnn_root=returnn_root)
        for dev_key in dev_keys
    ]

    cv_dataset_builder = MetaDatasetBuilder()
    cv_dataset_builder.add_hdf_dataset(
        name="data",
        hdf_files=cv_feature_hdfs,
        key_mapping={"data": "data"},
    )

    cv_dataset_builder.add_hdf_dataset(
        name="classes",
        hdf_files=cv_alignment_hdfs,
        dataset_config={
            "partition_epoch": 1,
            "seq_ordering": "sorted",
        },
        key_mapping={"data": "classes"},
        control=True,
    )
    cv_data_config = cv_dataset_builder.get_dict()

    # ********** Recog lexicon **********

    recog_lexicon = AddEowPhonemesToLexiconJob(train_lexicon).out_lexicon

    for rasr_input in {**dev_data_inputs, **test_data_inputs}.values():
        rasr_input.lexicon.filename = recog_lexicon

    # ********** Align data **********

    align_data_inputs = {
        f"{key}_align": copy.deepcopy(data_input)
        for key, data_input in {**train_data_inputs, **dev_data_inputs}.items()
    }
    for data_input in align_data_inputs.values():
        data_input.lexicon.filename = recog_lexicon
        if (not add_unknown and not augmented_lexicon) or use_wei_lexicon:
            assert data_input.corpus_object.corpus_file is not None
            data_input.corpus_object.corpus_file = corpus.FilterCorpusRemoveUnknownWordSegmentsJob(
                data_input.corpus_object.corpus_file,
                train_lexicon,
                all_unknown=False,
            ).out_corpus

    return BasicSetupData(
        train_key=train_key,
        dev_keys=dev_keys,
        test_keys=test_keys,
        align_keys=[f"{train_key}_align", *[f"{key}_align" for key in dev_keys]],
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        data_inputs={
            **train_data_inputs,
            **dev_data_inputs,
            **test_data_inputs,
            **align_data_inputs,
        },
    )
