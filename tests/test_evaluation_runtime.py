"""Regression for missing deterministic flag and empty first-replay directory."""
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.semantic_prior.evaluation_runtime import configure_evaluation, prepare_export_directory


def main():
    torch.use_deterministic_algorithms(False)
    first = configure_evaluation({'deterministic_algorithms': True})
    assert first['deterministic_algorithms'] and first['deterministic_warn_only']
    assert first['cublas_workspace_config'] == ':4096:8'
    assert first['cudnn_deterministic'] and not first['cudnn_benchmark']
    assert not first['matmul_tf32'] and not first['cudnn_tf32']
    assert not configure_evaluation({'deterministic_algorithms': False})['deterministic_algorithms']
    assert configure_evaluation({'deterministic_algorithms': True}) == first
    parent = ROOT / '.test_tmp'
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='evaluation_runtime_', dir=parent) as directory:
        export = Path(directory) / 'metric_inputs'
        prepare_export_directory(export)
        prepare_export_directory(export)  # empty directory from failed replay
        marker = export / 'existing.bin'
        marker.write_bytes(b'keep')
        try:
            prepare_export_directory(export)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Nonempty export accepted')
        assert marker.read_bytes() == b'keep'
    print('EVALUATION_RUNTIME_TESTS_PASSED')


if __name__ == '__main__':
    main()
