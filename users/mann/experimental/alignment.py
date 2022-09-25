from sisyphus import Job, Path, Task, tk, gs

import sys, os
import textwrap
import tempfile

import numpy as np

from i6_core.returnn import ReturnnConfig

class _HelperConfig(ReturnnConfig):
    PYTHON_CODE = textwrap.dedent(
        """\
        ${SUPPORT_CODE}

        ${PROLOG}
    
        ${REGULAR_CONFIG}
    
        ${EPILOG}
        """
    )


class TransformerConfig:
    def call(self):
        raise NotImplementedError()

from i6_core.util import get_val

class ShiftAlign(TransformerConfig):
    def __init__(self, shift, pad):
        self.shift = shift
        self.pad = pad

    def call(self, alignment):
        pad = get_val(self.pad)
        data = np.pad(alignment[self.shift:], (0, self.shift), 'constant', constant_values=(0, pad))
        return data

class TransformAlignmentJob(Job):
    def __init__(
        self,
        transformer_config,
        alignment,
        start_seq=0,
        end_seq=float("inf"),
        returnn_root=None,
    ):
        assert isinstance(transformer_config, TransformerConfig)
        self.transformer_config = transformer_config
        self.alignment = alignment
        self.start_seq = start_seq
        self.end_seq = end_seq
        self.returnn_root = returnn_root or gs.RETURNN_ROOT

        self.out_alignment = self.output_path("out.hdf")

        self.rqmts = {"cpu": 1, "mem": 8, "time": 2}
    
    def tasks(self):
        # yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmts)
    
    def run(self):
        sys.path.append(self.returnn_root)
        import returnn.datasets.hdf as rnn
        import numpy as np
        dataset = rnn.HDFDataset(
            files=[self.alignment],
            use_cache_manager=True
        )
        dataset.init_seq_order(epoch=1)

        # (fd, tmp_hdf_file) = tempfile.mkstemp(prefix=gs.TMP_PREFIX, suffix=".hdf")
        # os.close(fd)
        
        hdf_writer = rnn.SimpleHDFWriter(
            self.out_alignment.get_path(),
            dim=None,
            ndim=1
        )
        
        seq_idx = self.start_seq
        while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= self.end_seq:
            # seq_idxs += [seq_idx]
            dataset.load_seqs(seq_idx, seq_idx + 1)
            alignment = dataset.get_data(seq_idx, "classes")
            seq_tag = dataset.get_tag(seq_idx)

            out_alignment = self.transformer_config.call(alignment)

            hdf_writer.insert_batch(
                np.array([out_alignment]),
                np.array([len(out_alignment)]),
                np.array([seq_tag])
            )

            seq_idx += 1
        
        hdf_writer.close()
        # shutil.move(tmp_hdf_file, self.out_alignment.get_path())

