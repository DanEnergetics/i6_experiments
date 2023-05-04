__all__ = [
    "ReturnnRasrDataInput",
    "AllophoneLabeling",
    "OggZipRasrCacheDataInput",
    "OggZipExternRasrDataInput",
    "OggZipHdfDataInput",
    "HdfDataInput",
    "NextGenHdfDataInput",
    "ReturnnRawAlignmentHdfTrainingDataInput",
    "AllowedReturnnTrainingDataInput",
]

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from sisyphus import tk

import i6_core.am as am
import i6_core.rasr as rasr

from i6_core.returnn.hdf import BlissToPcmHDFJob, RasrAlignmentDumpHDFJob
from i6_core.util import MultiPath

RasrCacheTypes = Union[tk.Path, str, MultiPath, rasr.FlowNetwork]


@dataclass(frozen=True)
class RasrDataInput:
    features: RasrCacheTypes


class ReturnnRasrDataInput:
    """
    Holds the data for ReturnnRasrTrainingJob.
    """

    def __init__(
        self,
        name: str,
        crp: Optional[rasr.CommonRasrParameters] = None,
        alignments: Optional[RasrCacheTypes] = None,
        feature_flow: Optional[Union[rasr.FlowNetwork, Dict[str, rasr.FlowNetwork]]] = None,
        features: Optional[Union[RasrCacheTypes, Dict[str, RasrCacheTypes]]] = None,
        acoustic_mixtures: Optional[Union[tk.Path, str]] = None,
        feature_scorers: Optional[Dict[str, Type[rasr.FeatureScorer]]] = None,
        shuffle_data: bool = True,
        stm: Optional[tk.Path] = None,
        glm: Optional[tk.Path] = None,
    ):
        self.name = name
        self.crp = crp
        self.alignments = alignments
        self.feature_flow = feature_flow
        self.features = features
        self.acoustic_mixtures = acoustic_mixtures
        self.feature_scorers = feature_scorers
        self.shuffle_data = shuffle_data
        self.stm = stm
        self.glm = glm

    @staticmethod
    def get_data_dict():
        return {
            "class": "ExternSprintDataset",
            "sprintTrainerExecPath": "sprint-executables/nn-trainer",
            "sprintConfigStr": "",
            "suppress_load_seqs_print": True,
        }

    def build_crp(
        self,
        am_args,
        corpus_object,
        concurrent,
        segment_path,
        lexicon_args,
        cart_tree_path=None,
        allophone_file=None,
        lm_args=None,
    ):
        """
        constructs and returns a CommonRasrParameters from the given settings and files
        """
        crp = rasr.CommonRasrParameters()
        rasr.crp_add_default_output(crp)
        crp.acoustic_model_config = am.acoustic_model_config(**am_args)
        rasr.crp_set_corpus(crp, corpus_object)
        crp.concurrent = concurrent
        crp.segment_path = segment_path

        crp.lexicon_config = rasr.RasrConfig()
        crp.lexicon_config.file = lexicon_args["filename"]
        crp.lexicon_config.normalize_pronunciation = lexicon_args["normalize_pronunciation"]

        if "add_from_lexicon" in lexicon_args:
            crp.acoustic_model_config.allophones.add_from_lexicon = lexicon_args["add_from_lexicon"]
        if "add_all" in lexicon_args:
            crp.acoustic_model_config.allophones.add_all = lexicon_args["add_all"]

        if cart_tree_path is not None:
            crp.acoustic_model_config.state_tying.type = "cart"
            crp.acoustic_model_config.state_tying.file = cart_tree_path

        if lm_args is not None:
            crp.language_model_config = rasr.RasrConfig()
            crp.language_model_config.type = lm_args["type"]
            crp.language_model_config.file = lm_args["filename"]
            crp.language_model_config.scale = lm_args["scale"]

        if allophone_file is not None:
            crp.acoustic_model_config.allophones.add_from_file = allophone_file

        self.crp = crp

    def update_crp_with(
        self,
        *,
        corpus_file: Optional[tk.Path] = None,
        audio_dir: Optional[Union[str, tk.Path]] = None,
        corpus_duration: Optional[int] = None,
        segment_path: Optional[Union[str, tk.Path]] = None,
        concurrent: Optional[int] = None,
        shuffle_data: bool = True,
    ):
        if corpus_file is not None:
            self.crp.corpus_config.file = corpus_file
        if audio_dir is not None:
            self.crp.corpus_config.audio_dir = audio_dir
        if corpus_duration is not None:
            self.crp.corpus_duration = corpus_duration
        if segment_path is not None:
            self.crp.segment_path = segment_path
        if concurrent is not None:
            self.crp.concurrent = concurrent

        if self.shuffle_data or shuffle_data:
            self.crp.corpus_config.segment_order_shuffle = True
            self.crp.corpus_config.segment_order_sort_by_time_length = True
            self.crp.corpus_config.segment_order_sort_by_time_length_chunk_size = 384

    def get_crp(self, **kwargs) -> rasr.CommonRasrParameters:
        """
        constructs and returns a CommonRasrParameters from the given settings and files
        :rtype CommonRasrParameters:
        """
        if self.crp is None:
            self.build_crp(**kwargs)

        if self.shuffle_data:
            self.crp.corpus_config.segment_order_shuffle = True
            self.crp.corpus_config.segment_order_sort_by_time_length = True
            self.crp.corpus_config.segment_order_sort_by_time_length_chunk_size = 384

        return self.crp


@dataclass()
class AllophoneLabeling:
    silence_phoneme: str
    allophone_file: tk.Path
    phoneme_file: Optional[tk.Path] = None
    state_tying_file: Optional[tk.Path] = None


class OggZipRasrCacheDataInput:
    def __init__(
        self,
        oggzip_files: List[tk.Path],
        audio: Dict,
        alignment_file: tk.Path,
        allophone_labeling: AllophoneLabeling,
        partition_epoch: int = 1,
        seq_ordering: str = "laplace:.1000",
        *,
        meta_args: Optional[Dict[str, Any]] = None,
        ogg_args: Optional[Dict[str, Any]] = None,
        rasr_args: Optional[Dict[str, Any]] = None,
        acoustic_mixtures: Optional[tk.Path] = None,
    ):
        """
        :param oggzip_files: zipped ogg files which contain the audio
        :param audio: e.g. {"features": "raw", "sample_rate": 16000} for raw waveform input with a sample rate of 16 kHz
        :param alignment_file: hdf files which contain dumped RASR alignments
        :param allophone_labeling: labels for the RASR alignments
        :param partition_epoch: if >1, split the full dataset into multiple sub-epochs
        :param seq_ordering: sort the sequences in the dataset, e.g. "random" or "laplace:.100"
        :param meta_args: parameters for the `MetaDataset`
        :param ogg_args: parameters for the `OggZipDataset`
        :param rasr_args: parameters for the `SprintCacheDataset`
        :param acoustic_mixtures: path to a RASR acoustic mixture file (used in System classes, not RETURNN training)
        """
        self.oggzip_files = oggzip_files
        self.audio = audio
        self.alignment_file = alignment_file
        self.allophone_labeling = allophone_labeling
        self.partition_epoch = partition_epoch
        self.seq_ordering = seq_ordering
        self.meta_args = meta_args
        self.ogg_args = ogg_args
        self.rasr_args = rasr_args
        self.acoustic_mixtures = acoustic_mixtures

    def get_data_dict(self):
        return {
            "class": "MetaDataset",
            "data_map": {"classes": ("rasr", "classes"), "data": ("ogg", "data")},
            "datasets": {
                "rasr": {
                    "class": "SprintCacheDataset",
                    "data": {
                        "classes": {
                            "filename": self.alignment_file,
                            "data_type": "align",
                            "allophone_labeling": asdict(self.allophone_labeling),
                        },
                    },
                    "use_cache_manager": True,
                    **(self.rasr_args or {}),
                },
                "ogg": {
                    "class": "OggZipDataset",
                    "audio": self.audio,
                    "path": self.oggzip_files,
                    "use_cache_manager": True,
                    **(self.ogg_args or {}),
                },
            },
            "partition_epoch": self.partition_epoch,
            "seq_ordering": self.seq_ordering,
            **(self.meta_args or {}),
        }


class OggZipExternRasrDataInput:
    def __init__(
        self,
        oggzip_files: List[tk.Path],
        audio: Dict,
        alignment_file: tk.Path,
        rasr_exe: tk.Path,
        rasr_config_str: str,
        partition_epoch: int = 1,
        seq_ordering: str = "laplace:.1000",
        reduce_target_factor: int = 1,
        *,
        meta_args: Optional[Dict[str, Any]] = None,
        ogg_args: Optional[Dict[str, Any]] = None,
        rasr_args: Optional[Dict[str, Any]] = None,
        acoustic_mixtures: Optional[tk.Path] = None,
    ):
        """
        :param oggzip_files: zipped ogg files which contain the audio
        :param audio: e.g. {"features": "raw", "sample_rate": 16000} for raw waveform input with a sample rate of 16 kHz
        :param alignment_file: hdf files which contain dumped RASR alignments
        :param rasr_exe: path to RASR NN trainer executable
        :param rasr_config_str: str of rasr parameters
        :param partition_epoch: if >1, split the full dataset into multiple sub-epochs
        :param seq_ordering: sort the sequences in the dataset, e.g. "random" or "laplace:.100"
        :param reduce_target_factor: reduce the alignment by a factor
        :param meta_args: parameters for the `MetaDataset`
        :param ogg_args: parameters for the `OggZipDataset`
        :param rasr_args: parameters for the `SprintCacheDataset`
        :param acoustic_mixtures: path to a RASR acoustic mixture file (used in System classes, not RETURNN training)
        """
        self.oggzip_files = oggzip_files
        self.audio = audio
        self.alignment_file = alignment_file
        self.rasr_exe = rasr_exe
        self.rasr_config_str = rasr_config_str
        self.partition_epoch = partition_epoch
        self.seq_ordering = seq_ordering
        self.reduce_target_factor = reduce_target_factor
        self.meta_args = meta_args
        self.ogg_args = ogg_args
        self.rasr_args = rasr_args
        self.acoustic_mixtures = acoustic_mixtures

    def get_data_dict(self):
        return {
            "class": "MetaDataset",
            "data_map": {"classes": ("rasr", "classes"), "data": ("ogg", "data")},
            "datasets": {
                "rasr": {
                    "class": "SprintCacheDataset",
                    "sprintConfigSts": self.rasr_config_str,
                    "sprintTrainerExecPath": self.rasr_exe,
                    "partition_epoch": self.partition_epoch,
                    "suppress_load_seqs_print": True,
                    "reduce_target_factor": self.reduce_target_factor,
                    **(self.rasr_args or {}),
                },
                "ogg": {
                    "class": "OggZipDataset",
                    "audio": self.audio,
                    "path": self.oggzip_files,
                    "use_cache_manager": True,
                    **(self.ogg_args or {}),
                },
            },
            "seq_order_control_dataset": "rasr",
            **(self.meta_args or {}),
        }


class OggZipHdfDataInput:
    def __init__(
        self,
        oggzip_files: List[tk.Path],
        alignments: List[tk.Path],
        audio: Dict,
        partition_epoch: int = 1,
        seq_ordering: str = "laplace:.1000",
        meta_args: Optional[Dict[str, Any]] = None,
        ogg_args: Optional[Dict[str, Any]] = None,
        hdf_args: Optional[Dict[str, Any]] = None,
        acoustic_mixtures: Optional[tk.Path] = None,
    ):
        """
        :param oggzip_files: zipped ogg files which contain the audio
        :param alignments: hdf files which contain dumped RASR alignments
        :param audio: e.g. {"features": "raw", "sample_rate": 16000} for raw waveform input with a sample rate of 16 kHz
        :param partition_epoch: if >1, split the full dataset into multiple sub-epochs
        :param seq_ordering: sort the sequences in the dataset, e.g. "random" or "laplace:.100"
        :param meta_args: parameters for the `MetaDataset`
        :param ogg_args: parameters for the `OggZipDataset`
        :param hdf_args: parameters for the `HdfDataset`
        :param acoustic_mixtures: path to a RASR acoustic mixture file (used in System classes, not RETURNN training)
        """
        self.oggzip_files = oggzip_files
        self.alignments = alignments
        self.audio = audio
        self.partition_epoch = partition_epoch
        self.seq_ordering = seq_ordering
        self.meta_args = meta_args
        self.ogg_args = ogg_args
        self.hdf_args = hdf_args
        self.acoustic_mixtures = acoustic_mixtures

    def get_data_dict(self):
        return {
            "class": "MetaDataset",
            "data_map": {"classes": ("hdf", "classes"), "data": ("ogg", "data")},
            "datasets": {
                "hdf": {
                    "class": "HDFDataset",
                    "files": self.alignments,
                    "use_cache_manager": True,
                    **(self.hdf_args or {}),
                },
                "ogg": {
                    "class": "OggZipDataset",
                    "audio": self.audio,
                    "partition_epoch": self.partition_epoch,
                    "path": self.oggzip_files,
                    "seq_ordering": self.seq_ordering,
                    "use_cache_manager": True,
                    **(self.ogg_args or {}),
                },
            },
            "seq_order_control_dataset": "ogg",
            **(self.meta_args or {}),
        }


class HdfDataInput:
    def __init__(
        self,
        features: List[tk.Path],
        alignments: List[tk.Path],
        partition_epoch: int = 1,
        seq_ordering: str = "laplace:.1000",
        *,
        meta_args: Optional[Dict[str, Any]] = None,
        align_args: Optional[Dict[str, Any]] = None,
        feat_args: Optional[Dict[str, Any]] = None,
        acoustic_mixtures: Optional[tk.Path] = None,
    ):
        """
        :param features: hdf files which contain raw wve form or features, like GT or MFCC
        :param alignments: hdf files which contain dumped RASR alignments
        :param partition_epoch: if >1, split the full dataset into multiple sub-epochs
        :param seq_ordering: sort the sequences in the dataset, e.g. "random" or "laplace:.100"
        :param meta_args: parameters for the `MetaDataset`
        :param align_args: parameters for the `HDFDataset` for the alignments
        :param feat_args: parameters for the `HDFDataset` for the features
        :param acoustic_mixtures: path to a RASR acoustic mixture file (used in System classes, not RETURNN training)
        """
        self.features = features
        self.alignments = alignments
        self.partition_epoch = partition_epoch
        self.seq_ordering = seq_ordering
        self.meta_args = meta_args
        self.align_args = align_args
        self.feat_args = feat_args
        self.acoustic_mixtures = acoustic_mixtures

    def get_data_dict(self):
        return {
            "class": "MetaDataset",
            "data_map": {"classes": ("align", "classes"), "data": ("feat", "data")},
            "datasets": {
                "align": {
                    "class": "HDFDataset",
                    "files": self.alignments,
                    "use_cache_manager": True,
                    **(self.align_args or {}),
                },
                "feat": {
                    "class": "HDFDataset",
                    "files": self.features,
                    "use_cache_manager": True,
                    **(self.feat_args or {}),
                },
            },
            "partition_epoch": self.partition_epoch,
            "seq_ordering": self.seq_ordering,
            **(self.meta_args or {}),
        }


class NextGenHdfDataInput:
    def __init__(
        self,
        streams: Dict[str, List[tk.Path]],
        data_map: Dict[str, Tuple[str, str]],
        partition_epoch: int = 1,
        seq_ordering: str = "laplace:.1000",
        *,
        meta_args: Optional[Dict[str, Any]] = None,
        stream_args: Optional[Dict[str, Dict[str, Any]]] = None,
        acoustic_mixtures: Optional[tk.Path] = None,
    ):
        """
        :param streams: `NextGenHDFDataset` for different data streams
        :param data_map: a data map specifying the connection between the data stored in the HDF and RETURNN.
                         Key is the RETURNN name, first value is the name in the `datasets` from `MetaDataset`,
                         second value the name in the HDF.
        :param partition_epoch: if >1, split the full dataset into multiple sub-epochs
        :param seq_ordering: sort the sequences in the dataset, e.g. "random" or "laplace:.100"
        :param meta_args: parameters for the `MetaDataset`
        :param stream_args: parameters for the different `NextGenHDFDataset`
        :param acoustic_mixtures: path to a RASR acoustic mixture file (used in System classes, not RETURNN training)
        """
        self.streams = streams
        self.data_map = data_map
        self.partition_epoch = partition_epoch
        self.seq_ordering = seq_ordering
        self.meta_args = meta_args
        self.stream_args = stream_args
        self.acoustic_mixtures = acoustic_mixtures

        assert sorted(list(streams.keys())) == sorted([x[0] for x in data_map.values()])

    def get_data_dict(self):
        d = {
            "class": "MetaDataset",
            "data_map": {},
            "datasets": {},
            "partition_epoch": self.partition_epoch,
            "seq_ordering": self.seq_ordering,
            **(self.meta_args or {}),
        }
        for k, v in self.data_map.items():
            d["data_map"][k] = v

        for k, v in self.streams.items():
            d["datasets"][k] = {
                "class": "NextGenHDFDataset",
                "files": v,
                "use_cache_manager": True,
            }
            if self.stream_args is not None:
                d["datasets"][k].update(**self.stream_args[k] or {})

        return d


@dataclass()
class ReturnnRawAlignmentHdfTrainingDataInput:
    bliss_corpus: tk.Path
    alignment_caches: List[tk.Path]
    state_tying_file: tk.Path
    allophone_file: tk.Path
    returnn_root: tk.Path
    seq_ordering: str

    def get_data_dict(self):
        raw_hdf_path = BlissToPcmHDFJob(
            bliss_corpus=self.bliss_corpus,
            returnn_root=self.returnn_root,
        ).out_hdf
        alignment_hdf_path = RasrAlignmentDumpHDFJob(
            alignment_caches=self.alignment_caches,
            allophone_file=self.allophone_file,
            state_tying_file=self.state_tying_file,
            returnn_root=self.returnn_root,
        ).out_hdf_files

        data = {
            "class": "MetaDataset",
            "data_map": {"classes": ("alignments", "data"), "data": ("features", "data")},
            "datasets": {
                "alignments": {
                    "class": "HDFDataset",
                    "files": alignment_hdf_path,
                    "seq_ordering": self.seq_ordering,
                },
                "features": {
                    "class": "HDFDataset",
                    "files": [raw_hdf_path],
                },
            },
            "seq_order_control_dataset": "alignments",
        }

        return data


AllowedReturnnTrainingDataInput = Union[
    Dict,
    OggZipRasrCacheDataInput,
    OggZipExternRasrDataInput,
    OggZipHdfDataInput,
    NextGenHdfDataInput,
    ReturnnRawAlignmentHdfTrainingDataInput,
]
