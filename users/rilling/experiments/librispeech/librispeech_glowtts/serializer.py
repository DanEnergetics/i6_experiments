import copy
from sisyphus import tk
from typing import Any, Dict

from i6_experiments.common.setups.returnn_common.serialization import (
    Collection,
    ExternData,
    Network,
    PythonEnlargeStackWorkaroundNonhashedCode,
    ExplicitHash
)

from i6_experiments.common.setups.serialization import Import, ExternalImport

from i6_experiments.users.rossenbach.common_setups.returnn.serializer import PartialImport

from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection as TorchCollection,
    PyTorchModel,
)

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream

PACKAGE = __package__


def get_network_serializer(
        training: bool,
        returnn_common_root: tk.Path,
        datastreams: Dict[str, Datastream],
        network_module: str,
        net_args: Dict[str, Any],
        debug=False,
        **kwargs,
) -> Collection:
    """

    :param training
    :param returnn_common_root
    :param datastreams:
    :param net_args:
    :param debug:
    :return:
    """
    extern_data = [
        datastream.as_nnet_constructor_data(key)
        for key, datastream in datastreams.items()
    ]

    rc_recursionlimit = PythonEnlargeStackWorkaroundNonhashedCode
    rc_extern_data = ExternData(extern_data=extern_data)

    rc_package = PACKAGE + ".rc_networks"
    rc_construction_code = Import(rc_package + "." + network_module + ".construct_network")

    d = copy.deepcopy(net_args)
    if training is False:
        d["training"] = False

    rc_network = Network(
        net_func_name=rc_construction_code.object_name,
        net_func_map={key: key for key in datastreams.keys()}, # names just have to match
        net_kwargs=d,
    )

    serializer = Collection(
        serializer_objects=[
            rc_recursionlimit,
            rc_extern_data,
            rc_construction_code,
            rc_network,
        ],
        returnn_common_root=returnn_common_root,
        make_local_package_copy=not debug,
        packages={
            rc_package,
        },
    )

    return serializer

def get_pytorch_serializer(
        network_module: str,
        net_args: Dict[str, Any],
        use_custom_engine=False,
        forward=False,
        forward_args={},
        debug=False,
        target="audio",
        **kwargs
) -> TorchCollection:

    package = PACKAGE + ".pytorch_networks"

    pytorch_model_import = Import(
        code_object_path=package + ".%s.Model" % network_module
    )
    pytorch_train_step = Import(
        code_object_path=package + ".%s.train_step" % network_module
    )
    pytorch_model = PyTorchModel(
        model_class_name=pytorch_model_import.object_name,
        model_kwargs=net_args,
    )
    serializer_objects = [
        pytorch_model_import,
        pytorch_train_step,
        pytorch_model,
    ]
    if forward:
        if target == "audio":
            vocoder_dependencies = ExternalImport(tk.Path("/u/lukas.rilling/src/vocoder_collection/univnet", hash_overwrite="UNIVNET_TOOLS"))
            forward_step = PartialImport(
                code_object_path=package + ".%s.forward_step" % network_module,
                unhashed_package_root=PACKAGE,
                hashed_arguments=forward_args or {},
                unhashed_arguments={}
            )
            init_hook = Import(
                code_object_path=package + ".%s.forward_init_hook" % network_module,
                unhashed_package_root=PACKAGE
            )
            finish_hook = Import(
                code_object_path=package + ".%s.forward_finish_hook" % network_module,
                unhashed_package_root=PACKAGE
            )
            serializer_objects.extend(
                [vocoder_dependencies, forward_step, init_hook, finish_hook]
            )
        elif target == "spectrograms":
            forward_step = PartialImport(
                code_object_path=package + ".%s.forward_step_spectrograms" % network_module,
                unhashed_package_root=PACKAGE,
                hashed_arguments=forward_args or {},
                unhashed_arguments={},
                import_as="forward_step"
            )
            init_hook = Import(
                code_object_path=package + ".%s.forward_init_hook_spectrograms" % network_module,
                unhashed_package_root=PACKAGE,
                import_as="forward_init_hook"
            )
            finish_hook = Import(
                code_object_path=package + ".%s.forward_finish_hook_spectrograms" % network_module,
                unhashed_package_root=PACKAGE,
                import_as="forward_finish_hook"
            )
            serializer_objects.extend(
                [forward_step, init_hook, finish_hook]
            )
        elif target == "durations":
            forward_step = PartialImport(
                code_object_path=package + ".%s.forward_step_durations" % network_module,
                unhashed_package_root=PACKAGE,
                hashed_arguments=forward_args or {},
                unhashed_arguments={},
                import_as="forward_step"
            )
            init_hook = Import(
                code_object_path=package + ".%s.forward_init_hook_durations" % network_module,
                unhashed_package_root=PACKAGE,
                import_as="forward_init_hook"
            )
            finish_hook = Import(
                code_object_path=package + ".%s.forward_finish_hook_durations" % network_module,
                unhashed_package_root=PACKAGE,
                import_as="forward_finish_hook"
            )
            serializer_objects.extend(
                [forward_step, init_hook, finish_hook]
            )
        elif target == "latent_space":
            forward_step = PartialImport(
                code_object_path=package + ".%s.forward_step_latent_space" % network_module,
                unhashed_package_root=PACKAGE,
                hashed_arguments=forward_args or {},
                unhashed_arguments={},
                import_as="forward_step"
            )
            init_hook = Import(
                code_object_path=package + ".%s.forward_init_hook_latent_space" % network_module,
                unhashed_package_root=PACKAGE,
                import_as="forward_init_hook"
            )
            finish_hook = Import(
                code_object_path=package + ".%s.forward_finish_hook_latent_space" % network_module,
                unhashed_package_root=PACKAGE,
                import_as="forward_finish_hook"
            )
            serializer_objects.extend(
                [forward_step, init_hook, finish_hook]
            )
    if use_custom_engine:
        pytorch_engine = Import(
            code_object_path=package + ".%s.CustomEngine" % network_module,
            unhashed_package_root=PACKAGE
        )
        serializer_objects.append(pytorch_engine)
    serializer = TorchCollection(
        serializer_objects=serializer_objects,
        make_local_package_copy=not debug,
        packages={
            package,
        },
    )

    return serializer
