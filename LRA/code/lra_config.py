# random stringsssssssssssssssssssssssss
config = {
    "listops": {
        "dataset": {
            "train": 96000,
            "dev": 2000,
            "test": 2000,
        },
        "model": {
            "learn_pos_emb": True,
            "tied_weights": False,
            "embedding_dim": 64,
            "transformer_dim": 64,
            "transformer_hidden_dim": 128,
            "head_dim": 32,
            "num_head": 2,
            "num_layers": 2,
            "vocab_size": 32,
            "max_seq_len": 2000,
            "dropout_prob": 0.1,
            "attention_dropout": 0.1,
            "pooling_mode": "MEAN",
            "num_classes": 10,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.0001,
            "warmup": 1000,
            "lr_decay": "linear",
            "weight_decay": 0,
            "eval_frequency": 50,
            "num_train_steps": 5000,
            "num_eval_steps": 62,
        },
        "gpu_memory": {
            "softmax": 32,
            "nystrom-32": 32,
            "nystrom-64": 32,
            "nystrom-128": 32,
            "nystrom-256": 32,
            "linformer-256": 32,
            "reformer-2": 32,
            "performer-256": 32,
            "linear": 32,
        },
        "extra_attn_config": {
            "softmax": {"attention_grad_checkpointing": True},
            "nystrom-32": {"attention_grad_checkpointing": False, "num_landmarks": 32, "conv_kernel_size": 35},
            "nystrom-64": {"attention_grad_checkpointing": False, "num_landmarks": 64, "conv_kernel_size": 35},
            "nystrom-128": {"attention_grad_checkpointing": False, "num_landmarks": 128, "conv_kernel_size": 35},
            "nystrom-256": {"attention_grad_checkpointing": False, "num_landmarks": 256, "conv_kernel_size": 35},
            "linformer-256": {"attention_grad_checkpointing": False, "linformer_k": 256},
            "reformer-2": {"attention_grad_checkpointing": False, "num_hash": 2},
            "performer-256": {"attention_grad_checkpointing": False, "rp_dim": 256, "kernel_type": "relu"},
            "linear": {"attention_grad_checkpointing": False},
        }
    },
    "image": {
        "dataset": {
            "train": 45000,
            "dev": 5000,
            "test": 10000,
        },
        "model": {
            "learn_pos_emb": True,
            "tied_weights": False,
            "embedding_dim": 64,
            "transformer_dim": 64,
            "transformer_hidden_dim": 128,
            "head_dim": 32,
            "num_head": 2,
            "num_layers": 2,
            "vocab_size": 512,
            "max_seq_len": 1024,
            "dropout_prob": 0.1,
            "attention_dropout": 0.1,
            "pooling_mode": "MEAN",
            "num_classes": 10,
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 0.0001,
            "warmup": 175,
            "lr_decay": "linear",
            "weight_decay": 0,
            "eval_frequency": 175,
            "num_train_steps": 35000,
            "num_eval_steps": 20,
        },
        "gpu_memory": {
            "softmax": 128,
            "nystrom-32": 128,
            "nystrom-64": 128,
            "nystrom-128": 128,
            "nystrom-256": 128,
            "linformer-256": 128,
            "reformer-2": 128,
            "performer-256": 128,
            "linear": 128,
        },
        "extra_attn_config": {
            "softmax": {"attention_grad_checkpointing": True},
            "nystrom-32": {"attention_grad_checkpointing": False, "num_landmarks": 32, "conv_kernel_size": 35},
            "nystrom-64": {"attention_grad_checkpointing": False, "num_landmarks": 64, "conv_kernel_size": 35},
            "nystrom-128": {"attention_grad_checkpointing": False, "num_landmarks": 128, "conv_kernel_size": 35},
            "nystrom-256": {"attention_grad_checkpointing": False, "num_landmarks": 256, "conv_kernel_size": 35},
            "linformer-256": {"attention_grad_checkpointing": False, "linformer_k": 256},
            "reformer-2": {"attention_grad_checkpointing": False, "num_hash": 2},
            "performer-256": {"attention_grad_checkpointing": False, "rp_dim": 256, "kernel_type": "relu"},
            "linear": {"attention_grad_checkpointing": False},
        }
    },
    "pathfinder32": {
        "model": {
            "learn_pos_emb": True,
            "tied_weights": False,
            "embedding_dim": 64,
            "transformer_dim": 64,
            "transformer_hidden_dim": 128,
            "head_dim": 32,
            "num_head": 2,
            "num_layers": 2,
            "vocab_size": 512,
            "max_seq_len": 1024,
            "dropout_prob": 0.1,
            "attention_dropout": 0.1,
            "pooling_mode": "MEAN",
            "num_classes": 2,
        },
        "training": {
            "batch_size": 256,
            "learning_rate": 0.0001,
            "warmup": 312,
            "lr_decay": "linear",
            "weight_decay": 0,
            "eval_frequency": 312,
            "num_train_steps": 62400,
            "num_eval_steps": 312,
        },
        "gpu_memory": {
            "softmax": 128,
            "nystrom-32": 128,
            "nystrom-64": 128,
            "nystrom-128": 128,
            "nystrom-256": 128,
            "linformer-256": 128,
            "reformer-2": 128,
            "performer-256": 128,
            "linear": 128,
            "curformer": 128,
        },
        "extra_attn_config": {
            "softmax": {"attention_grad_checkpointing": True},
            "nystrom-32": {"attention_grad_checkpointing": False, "num_landmarks": 32, "conv_kernel_size": 35},
            "nystrom-64": {"attention_grad_checkpointing": False, "num_landmarks": 64, "conv_kernel_size": 35},
            "nystrom-128": {"attention_grad_checkpointing": False, "num_landmarks": 128, "conv_kernel_size": 35},
            "nystrom-256": {"attention_grad_checkpointing": False, "num_landmarks": 256, "conv_kernel_size": 35},
            "linformer-256": {"attention_grad_checkpointing": False, "linformer_k": 256},
            "reformer-2": {"attention_grad_checkpointing": False, "num_hash": 2},
            "performer-256": {"attention_grad_checkpointing": False, "rp_dim": 256, "kernel_type": "relu"},
            "linear": {"attention_grad_checkpointing": False},
            "curformer": {"attention_grad_checkpointing": False, "select_number": 64, "select_type": "random"},
        }
    },
    "retrieval": {
        "dataset": {
            "train": 147086,
            "dev": 18090,
            "test": 17437,
        },
        "model": {
            "learn_pos_emb": True,
            "tied_weights": False,
            "embedding_dim": 64,
            "transformer_dim": 64,
            "transformer_hidden_dim": 128,
            "head_dim": 32,
            "num_head": 2,
            "num_layers": 2,
            "vocab_size": 512,
            "max_seq_len": 4000,
            "dropout_prob": 0.1,
            "attention_dropout": 0.1,
            "pooling_mode": "MEAN",
            "num_classes": 2,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.0001,
            "warmup": 800,
            "lr_decay": "linear",
            "weight_decay": 0,
            "eval_frequency": 300,
            "num_train_steps": 30000,
            "num_eval_steps": 565,
        },
        "gpu_memory": {
            "softmax": 32,
            "nystrom-32": 32,
            "nystrom-64": 32,
            "nystrom-128": 32,
            "nystrom-256": 32,
            "linformer-256": 32,
            "reformer-2": 32,
            "performer-256": 32,
            "linear": 32,
        },
        "extra_attn_config": {
            "softmax": {"attention_grad_checkpointing": True},
            "nystrom-32": {"attention_grad_checkpointing": False, "num_landmarks": 32, "conv_kernel_size": 35},
            "nystrom-64": {"attention_grad_checkpointing": False, "num_landmarks": 64, "conv_kernel_size": 35},
            "nystrom-128": {"attention_grad_checkpointing": False, "num_landmarks": 128, "conv_kernel_size": 35},
            "nystrom-256": {"attention_grad_checkpointing": False, "num_landmarks": 256, "conv_kernel_size": 35},
            "linformer-256": {"attention_grad_checkpointing": False, "linformer_k": 256},
            "reformer-2": {"attention_grad_checkpointing": False, "num_hash": 2},
            "performer-256": {"attention_grad_checkpointing": False, "rp_dim": 256, "kernel_type": "relu"},
            "linear": {"attention_grad_checkpointing": False},
        }
    },
    "text": {
        "dataset": {
            "train": 25000,
            "dev": 25000,
            "test": 25000,
        },
        "model": {
            "learn_pos_emb": True,
            "tied_weights": False,
            "embedding_dim": 64,
            "transformer_dim": 64,
            "transformer_hidden_dim": 128,
            "head_dim": 32,
            "num_head": 2,
            "num_layers": 2,
            "vocab_size": 512,
            "max_seq_len": 4000,
            "dropout_prob": 0.1,
            "attention_dropout": 0.1,
            "pooling_mode": "MEAN",
            "num_classes": 2,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.0001,
            "warmup": 8000,
            "lr_decay": "linear",
            "weight_decay": 0,
            "eval_frequency": 500,
            "num_train_steps": 20000,
            "num_eval_steps": 781,
        },
        "gpu_memory": {
            "softmax": 32,
            "nystrom-32": 32,
            "nystrom-64": 32,
            "nystrom-128": 32,
            "nystrom-256": 32,
            "linformer-256": 32,
            "reformer-2": 32,
            "performer-256": 32,
            "linear": 32,
        },
        "extra_attn_config": {
            "softmax": {"attention_grad_checkpointing": True},
            "nystrom-32": {"attention_grad_checkpointing": False, "num_landmarks": 32, "conv_kernel_size": 35},
            "nystrom-64": {"attention_grad_checkpointing": False, "num_landmarks": 64, "conv_kernel_size": 35},
            "nystrom-128": {"attention_grad_checkpointing": False, "num_landmarks": 128, "conv_kernel_size": 35},
            "nystrom-256": {"attention_grad_checkpointing": False, "num_landmarks": 256, "conv_kernel_size": 35},
            "linformer-256": {"attention_grad_checkpointing": False, "linformer_k": 256},
            "reformer-2": {"attention_grad_checkpointing": False, "num_hash": 2},
            "performer-256": {"attention_grad_checkpointing": False, "rp_dim": 256, "kernel_type": "relu"},
            "linear": {"attention_grad_checkpointing": False},
        }
    }
}

config["pathfinder32-curv_baseline"] = config["pathfinder32"]
config["pathfinder32-curv_contour_length_9"] = config["pathfinder32"]
config["pathfinder32-curv_contour_length_14"] = config["pathfinder32"]
