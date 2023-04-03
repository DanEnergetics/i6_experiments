from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode


#RASR_BINARY_PATH = compile_rasr_binaries_i6mode(commit="907eec4f4e36c11153f6ab6b5dd7675116f909f6")
RASR_BINARY_PATH = compile_rasr_binaries_i6mode(branch="bene_unpushed_assert")
RASR_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"


RETURNN_EXE = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")

RETURNN_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn", commit="055994fdaf4fbfbfd7a8ba1d7806d3bcf7cf6599").out_repository
RETURNN_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_RC_ROOT"

RETURNN_COMMON = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn_common", commit="1204f5d377399da6267fcd45e41c21985c73aa14", checkout_folder_name="returnn_common").out_repository
RETURNN_COMMON.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_COMMON"
