"""
NLTK TIMIT (small subset of TIMIT, but freely available via NLTK)
"""

from __future__ import annotations
from typing import Optional, Union, Dict, Any
from returnn_common.datasets.interface import DatasetConfig, VocabConfig


class NltkTimit(DatasetConfig):
    """
    NLTK TIMIT (small subset of TIMIT, but freely available via NLTK)
    """

    def __init__(self, *,
                 main_key: Optional[str] = None,
                 random_permute_audio: Union[None, bool, Dict[str, Any]] = None,  # for train/main_key
                 ):
        super(NltkTimit, self).__init__()
        self.vocab = _timit_vocab
        self.main_key = main_key
        if random_permute_audio is None:
            random_permute_audio = {"rnd_zoom_order": 0}
        self.random_permute_audio = random_permute_audio

    # noinspection PyMethodMayBeStatic
    def get_dataset(self, key: str) -> Dict[str, Any]:
        """dataset"""
        assert key in {"train", "dev", "devtrain"}
        # num_seqs = {'train': 3696, 'dev': 192}  # full TIMIT
        num_seqs = {'train': 144, 'dev': 16, "devtrain": 144}
        d = {
            "class": "NltkTimitDataset",
            "with_delta": True,
            "train": key.endswith("train"),
            "seq_ordering": "laplace:.10" if key == "train" else "sorted",
            "estimated_num_seqs": num_seqs[key],
        }
        if key.startswith("dev"):
            d["fixed_random_seed"] = 1
        if self.random_permute_audio and key in {"train", self.main_key}:
            d["random_permute_audio"] = self.random_permute_audio
        return d

    def get_extern_data(self) -> Dict[str, Dict[str]]:
        """extern data"""
        from returnn.tf.util.data import FeatureDim, SpatialDim, batch_dim
        time_dim = SpatialDim("time")
        feature_dim = FeatureDim("audio", 40 * 2)  # keep consistent with above
        classes_dim = FeatureDim("phones", dimension=self.vocab.get_num_classes())
        return {
            "data": {"dim_tags": [batch_dim, time_dim, feature_dim]},
            "classes": {
                "dim_tags": [batch_dim, time_dim],
                "sparse_dim": classes_dim,
                "vocab": self.vocab.get_opts()},
        }

    def get_train_dataset(self) -> Dict[str]:
        """train"""
        return self.get_dataset("train")

    def get_eval_datasets(self) -> Dict[str, Dict[str]]:
        """dev/devtrain/eval or so"""
        return {
            "dev": self.get_dataset("dev"),
            "devtrain": self.get_dataset("devtrain"),
        }

    def get_main_dataset(self) -> Dict[str, Any]:
        """main dataset"""
        assert self.main_key, "main key not defined"
        return self.get_dataset(self.main_key)

    def get_main_name(self) -> str:
        """main name"""
        assert self.main_key, "main key not defined"
        return self.main_key


class TimitVocab(VocabConfig):
    """
    TIMIT phone vocab
    """

    def get_num_classes(self) -> int:
        """
        Get num classes
        """
        return 61

    def get_opts(self) -> Dict[str, Any]:
        """
        Options for RETURNN vocab,
        e.g. as defined in `Data`, `extern_data`, :func:`Vocabulary.create_vocab` (in RETURNN).
        """
        # TODO...
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def get_eos_idx(self) -> Optional[int]:
        """end-of-sequence (EOS)"""
        return None

    def get_bos_idx(self) -> Optional[int]:
        """beginning-of-sequence (BOS)"""
        return self.get_eos_idx()


_timit_vocab = TimitVocab()
