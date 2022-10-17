layernorm_ft_net = {
    "stft": {
        "class": "stft",
        "frame_shift": 200,
        "frame_size": 800,
        "fft_size": 1024,
        "from": "data:audio_features",
    },
    "abs": {
        "class": "activation",
        "from": "stft",
        "activation": "abs",
    },
    "power": {
        "class": "eval",
        "from": "abs",
        "eval": "source(0) ** 2",
    },
    "mel_filterbank": {
        "class": "mel_filterbank",
        "from": "power",
        "fft_size": 1024,
        "nr_of_filters": 80,
        "n_out": 80,
    },
    "log": {
        "from": "mel_filterbank",
        "class": "activation",
        "activation": "safe_log",
    },
    "log_mel_features": {
        "class": "layer_norm",
        "from": "log",
    }
}


log10_net = {
    "stft": {
        "class": "stft",
        "frame_shift": 200,
        "frame_size": 800,
        "fft_size": 1024,
        "from": "data:audio_features",
    },
    "abs": {
        "class": "activation",
        "from": "stft",
        "activation": "abs",
    },
    "power": {
        "class": "eval",
        "from": "abs",
        "eval": "source(0) ** 2",
    },
    "mel_filterbank": {
        "class": "mel_filterbank",
        "from": "power",
        "fft_size": 1024,
        "nr_of_filters": 80,
        "n_out": 80,
    },
    "log": {
        "from": "mel_filterbank",
        "class": "activation",
        "activation": "safe_log",
        "opts": {"eps": 1e-10},
    },
    "log10": {
        "from": "log",
        "class": "eval",
        "eval": "source(0) / 2.3026"
    },
    "log_mel_features": {
        "class": "copy",
        "from": "log10",
    }
}

log10_halved_net = {
    "stft": {
        "class": "stft",
        "frame_shift": 200,
        "frame_size": 800,
        "fft_size": 1024,
        "from": "data:audio_features",
    },
    "abs": {
        "class": "activation",
        "from": "stft",
        "activation": "abs",
    },
    "power": {
        "class": "eval",
        "from": "abs",
        "eval": "source(0) ** 2",
    },
    "mel_filterbank": {
        "class": "mel_filterbank",
        "from": "power",
        "fft_size": 1024,
        "nr_of_filters": 80,
        "n_out": 80,
    },
    "log": {
        "from": "mel_filterbank",
        "class": "activation",
        "activation": "safe_log",
        "opts": {"eps": 1e-10},
    },
    "log10": {
        "from": "log",
        "class": "eval",
        "eval": "source(0) / 4.6"
    },
    "log_mel_features": {
        "class": "copy",
        "from": "log10",
    }
}