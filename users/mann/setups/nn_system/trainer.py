from sisyphus.delayed_ops import DelayedFormat

import copy
from collections import ChainMap

from i6_core.rasr import WriteRasrConfigJob, RasrConfig, RasrCommand
from i6_core.returnn import ReturnnTrainingJob, ReturnnRasrDumpHDFJob, ReturnnRasrTrainingJob, ReturnnComputePriorJob
from i6_core.meta.system import select_element
from i6_experiments.users.mann.experimental.write import PickleSegmentsJob, WriteFlowNetworkJob
from i6_experiments.users.mann.nn.util import DelayedCodeWrapper, maybe_add_dependencies

class BaseTrainer:

    def __init__(self, system=None):
        if system:
            self.set_system(system)
        self.configs = {}

    def set_system(self, system, **kwargs):
        self.system = system
    
    def save_job(self, feature_corpus, name, job):
        self.system.jobs[feature_corpus]['train_nn_%s' % name] = job
        self.system.nn_models[feature_corpus][name] = job.out_models
        self.system.nn_checkpoints[feature_corpus][name] = job.out_checkpoints
        self.system.nn_configs[feature_corpus][name] = job.out_returnn_config_file

    def extract_prior(self, name, crnn_config, training_args, epoch):
        crnn_config = self.configs[name]
        score_features = ReturnnComputePriorJob(
            model_checkpoint=self.system.nn_checkpoints[training_args["feature_corpus"]][name][epoch],
            returnn_config=crnn_config,
            # model_checkpoint = self.jobs[training_args['feature_corpus']]['train_nn_%s' % name].out_checkpoints[epoch],
            returnn_python_exe = training_args.get("returnn_python_exe", None),
            returnn_root = training_args.get("returnn_root", None)
        )
        return score_features

    @staticmethod
    def write_helper(crp, feature_flow, alignment,
            num_classes=None,
            disregarded_classes=None, class_label_file=None,
            buffer_size=200 * 1024,
            extra_rasr_config=None,
            use_python_control=True,
            **kwargs
        ):
            kwargs = locals()
            del kwargs["kwargs"]
            return WriteRasrConfigJob(**kwargs)

    def write(self, corpus, feature_corpus, feature_flow, alignment, num_classes, **kwargs):
        j = self.write_helper(
            crp = self.system.csp[corpus],
            feature_flow = self.system.feature_flows[feature_corpus][feature_flow],
            alignment = select_element(self.system.alignments, feature_corpus, alignment),
            num_classes = self.system.functor_value(num_classes),
            **kwargs)
        return j

    def train_helper(
            self,
            returnn_config,
            log_verbosity=3, device='gpu',
            num_epochs=1, save_interval=1, keep_epochs=None,
            time_rqmt=4, mem_rqmt=4, cpu_rqmt=2,
            returnn_python_exe=None, returnn_root=None,
            **_ignored
        ):
            kwargs = locals()
            del kwargs["_ignored"], kwargs["self"]
            return ReturnnTrainingJob(**kwargs)

class BaseTrainer(BaseTrainer):

    def make_sprint_dataset(
            self,
            name,
            corpus, feature_corpus, feature_flow, alignment, num_classes,
            estimated_num_seqs=None, partition_epochs=None,
            **kwargs
        ):
            kwargs = locals()
            kwargs.update(kwargs.pop("kwargs", {}))
            del kwargs["self"]
            return self.write(**kwargs).create_dataset_config(crp=self.system.crp[corpus], **kwargs)

    def make_combined_ds(self, arg_mapping):
        keys = ["alignment", "teacher"]
        acc_num_seqs = sum(args["estimated_num_seqs"] for args in arg_mapping.values())
        datasets = {
            key: self.make_sprint_dataset(**arg_mapping[key]) for key in keys
        }
        return {
            "class": "CombinedDataset",
            "datasets": datasets,
            "data_map": {
                ("alignment", "data"): "data",
                ("teacher", "data"): "data",
                ("alignment", "classes"): "classes"
            },
            "seq_ordering": "random_dataset",
            "estimated_num_seqs": acc_num_seqs,
        }
    
    def train(self, name, train_data, dev_data, crnn_config, feature_corpus, **kwargs):
        train_data = self.make_combined_ds(train_data)
        dev_data = self.make_sprint_dataset(**dev_data)
        training_args = ChainMap(locals().copy(), kwargs)
        del training_args["self"], training_args["kwargs"]
        j = self.train_helper(**training_args)
        # feature_corpus = "train"
        self.system.jobs[feature_corpus]['train_nn_%s' % name] = j
        self.system.nn_models[feature_corpus][name] = j.out_models
        self.system.nn_checkpoints[feature_corpus][name] = j.out_checkpoints
        self.system.nn_configs[feature_corpus][name] = j.out_returnn_config_file

class HdfAlignTrainer(BaseTrainer):

    def get_segments_pkl(self, overlay_name):
        return PickleSegmentsJob(self.system.csp[overlay_name].segment_path).out_segment_pickle
    
    @staticmethod
    def write_rasr_train_config(
        crp,
        feature_flow_file,
        alignment=None,
        num_classes=None,
        disregarded_classes=None,
        class_label_file=None,
        buffer_size=200 * 1024,
        partition_epochs=None,
        extra_rasr_config=None,
        extra_rasr_post_config=None,
        use_python_control=True,
        **_ignored,
    ):
        kwargs = locals().copy()
        del kwargs["feature_flow_file"], kwargs["extra_rasr_config"], kwargs["_ignored"]
        extra_rasr_config = extra_rasr_config or RasrConfig()
        extra_rasr_config.neural_network_trainer.feature_extraction.file = feature_flow_file
        config, post_config = ReturnnRasrTrainingJob.create_config(
            extra_rasr_config=extra_rasr_config,
            **kwargs,
        )
        write_rasr_config = WriteRasrConfigJob(config, post_config)
        return write_rasr_config.out_config
    
    @staticmethod
    def get_rasr_dataset_config(crp, dataset_name, config_file, partition_epochs=None, estimated_num_seqs=None, **_ignored):
        """ Returns a dataset config for use inside a returnn config.
        
        "function" attribute must be called such that the the config path is recognized by sisyphus
        as a "Path" object. """ 
        config_str = DelayedFormat(
            "--config={} --*.LOGFILE=nn-trainer.{}.log --*.TASK=1",
            config_file, dataset_name
        )
        dataset = { 
            'class'                 : 'ExternSprintDataset',
            'sprintTrainerExecPath' : RasrCommand.select_exe(crp.nn_trainer_exe, 'nn-trainer'),
            'sprintConfigStr'       : config_str,
        }
        if partition_epochs is not None:
            dataset["partitionEpoch"] = partition_epochs
        if estimated_num_seqs is not None:
            dataset["estimated_num_seqs"] = estimated_num_seqs
        return dataset
    
    def make_rasr_dataset(self, name, corpus, feature_flow, **kwargs):
        from i6_core.returnn import ReturnnRasrTrainingJob
        from i6_experiments.users.mann.experimental.write import WriteFlowNetworkJob
        feature_flow = self.system.feature_flows[corpus][feature_flow]
        feature_flow = ReturnnRasrTrainingJob.create_flow(feature_flow=feature_flow, **kwargs)
        write_feature_flow = WriteFlowNetworkJob(flow=feature_flow)
        rasr_config_file = self.write_rasr_train_config(self.system.crp[corpus], write_feature_flow.out_network_file, **kwargs)
        return self.get_rasr_dataset_config(self.system.crp[corpus], name, rasr_config_file, **kwargs)

    def make_hdf_dataset(self, hdf_alignment):
        return {
            "class": "HDFDataset",
            "files": [hdf_alignment],
            "use_cache_manager": True
        }

    def make_combined_ds(self,
        name,
        corpus,
        hdf_alignment,
        partition_epoch,
        rasr_args,
        hdf_features,
        hdf_classes_key="data",
    ):
        datasets = {}
        if hdf_features is None:
            datasets["features"] = self.make_rasr_dataset(name, corpus, **rasr_args)
        else:
            datasets["features"] = self.make_hdf_dataset(hdf_features)
        datasets["hdf_align"] = self.make_hdf_dataset(hdf_alignment)
        return {
            "class": "MetaDataset",
            "datasets": datasets,
            "data_map": {
                "classes": ("hdf_align", hdf_classes_key),
                "data": ("features", "data")
            },
            "seq_list_file": self.get_segments_pkl(corpus),
            "partition_epoch": partition_epoch,
            "seq_ordering": "default",
        }

    def train(self, name, hdf_alignment, partition_epochs, returnn_config, feature_corpus, num_classes, hdf_features=None, **kwargs):
        num_classes = self.system.functor_value(num_classes)
        returnn_config = copy.deepcopy(returnn_config)
        training_args = ChainMap(locals().copy(), kwargs)
        del training_args["self"], training_args["kwargs"]
        returnn_config.config["num_outputs"]["classes"] = [self.system.functor_value(num_classes), 1]
        for key in ["train", "dev"]:
            returnn_config.config[key] = self.make_combined_ds(
                key,
                "crnn_" + key,
                hdf_alignment,
                partition_epochs[key],
                rasr_args=dict(feature_corpus=feature_corpus, **kwargs),
                hdf_features=hdf_features,
            )
        # training_args.maps.insert(0, data)
        j = self.train_helper(**training_args)
        self.configs[name] = returnn_config
        self.save_job(feature_corpus, name, j)

class SoftAlignTrainer(BaseTrainer):

    def get_segments_pkl(self, overlay_name):
        from recipe.crnn.multi_sprint_training import PickleSegments
        return PickleSegments(self.system.csp[overlay_name].segment_path).segment_pickle

    def make_hdf_dataset(self, soft_alignment):
        return {
            "class": "HDFDataset",
            "files": [soft_alignment],
            "use_cache_manager": True
        }

    def make_combined_ds(self, name, corpus, soft_alignment, partition_epoch, arg_mapping):
        # acc_num_seqs = sum(args["estimated_num_seqs"] for args in arg_mapping.values())
        # TODO:
        # * class: hdf
        # * data_map: map soft_align to data
        # * add entry in num_outputs
        datasets = {}
        datasets["sprint"] = self.make_sprint_dataset(name, corpus, **arg_mapping)
        datasets["soft_align"] = self.make_hdf_dataset(soft_alignment)
        return {
            "class": "MetaDataset",
            "datasets": datasets,
            "data_map": {
                "classes_soft_align": ("soft_align", "data"),
                "classes": ("sprint", "classes"),
                "data": ("sprint", "data")
            },
            "seq_list_file": self.get_segments_pkl(corpus),
            "partition_epoch": partition_epoch,
            "seq_ordering": "default"
        }

    def train(self, name, soft_alignment, partition_epochs, crnn_config, feature_corpus, num_classes, **kwargs):
        training_args = ChainMap(locals().copy(), kwargs)
        del training_args["self"], training_args["kwargs"]
        crnn_config["num_outputs"]["classes_soft_align"] = [
            self.system.functor_value(num_classes), 2
        ]
        data = {
            key + "_data": self.make_combined_ds(
                key,
                "crnn_" + key,
                soft_alignment,
                partition_epochs[key],
                dict(num_classes=num_classes, feature_corpus=feature_corpus, **kwargs)
            )
            for key in ["train", "dev"]
        }
        # training_args.maps.insert(0, data)
        j = self.train_helper(**data, **training_args)
        # feature_corpus = "train"
        self.system.jobs[feature_corpus]['train_nn_%s' % name] = j
        self.system.nn_models[feature_corpus][name] = j.out_models
        self.system.nn_checkpoints[feature_corpus][name] = j.out_checkpoints
        self.system.nn_configs[feature_corpus][name] = j.out_returnn_config_file


class SprintCacheTrainer(BaseTrainer):
    def make_ds(self, feature_corpus, alignment_path, feature_path, partition_epoch=1, seq_ordering=None, cached=True, **kwargs):
        assert feature_corpus.startswith("crnn"), "Maybe something about the format went wrong"
        maybe_cache = lambda path: path if not cached else DelayedCodeWrapper("cf('{}')", path)
        dataset = {
            "class": "SprintCacheDataset",
            "data": {
                "data": {"filename": maybe_cache(feature_path)},
                "classes": {
                    "filename": maybe_cache(alignment_path),
                    "allophone_labeling": {
                        "silence_phone": "[SILENCE]",
                        "allophone_file": maybe_cache(self.system.get_allophone_file()),
                        "state_tying_file": maybe_cache(self.system.get_state_tying_file()),
                    }
                }
            },
            "partition_epoch": partition_epoch,
            "seq_list_filter_file": self.system.csp[feature_corpus].segment_path,
        }
        if seq_ordering:
            dataset["seq_ordering"] = seq_ordering
        return dataset
    
    def train(self, name, returnn_config, alignment, partition_epochs, feature_corpus, feature_flow, num_classes, seq_ordering=None, cached=True, **kwargs):
        returnn_config = copy.deepcopy(returnn_config)
        if cached:
            maybe_add_dependencies(returnn_config, "from returnn.util.basic import cf")
        returnn_config.config["extern_data"] = {
            "classes": {"dim": self.system.num_classes(), "dtype": "int16", "shape": (None,), "sparse": True},
            "data": {"dim": self.system.num_input, "shape": (None, self.system.num_input)},
        }
        training_args = ChainMap(locals().copy(), kwargs)
        del training_args["self"], training_args["kwargs"]

        feature_path = self.system.feature_bundles[feature_corpus][feature_flow]
        alignment_path = select_element(self.system.alignments, feature_corpus, alignment).alternatives["bundle"]
        returnn_config.config.update({
            key: self.make_ds(
                "crnn_" + key,
                alignment_path,
                feature_path,
                partition_epochs.get(key, 1),
                cached=cached,
                seq_ordering=None if key != "train" else seq_ordering
            )
            for key in ["train", "dev"]
        })
        j = self.train_helper(
            **training_args,
        )
        self.configs[name] = returnn_config
        self.save_job(feature_corpus, name, j)
