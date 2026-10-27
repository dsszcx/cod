"""Restore the numerical settings used by the fixed candidate training runs."""
import os
from pathlib import Path

import torch


def configure_evaluation(cfg):
    deterministic = bool(cfg.get('deterministic_algorithms', False))
    if deterministic:
        # Set before build_model/seed_all initializes CUDA, as in the launcher.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # same as training_core.seed_all
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def prepare_export_directory(path):
    """Permit the empty directory left by the failed first replay; overwrite nothing."""
    path = Path(path)
    path.mkdir(exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError('Probability export is nonempty: ' + str(path))
    return path
