"""Synthetic resume/scheduler tests, not a COD performance experiment."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.semantic_prior.full_training_state import make_scheduler, save_checkpoint, latest_checkpoint, restore

TEST_ROOT = Path(__file__).resolve().parents[1]/'.test_tmp'
TEST_ROOT.mkdir(exist_ok=True)

class TestState(unittest.TestCase):
    def test_exact_epoch_resume_and_lr_boundaries(self):
        torch.set_num_threads(2)
        config = dict(lr_milestones=[2, 4], lr_gamma=.1)

        def setup():
            torch.manual_seed(12)
            model = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.BatchNorm1d(8),
                                        torch.nn.Dropout(.3), torch.nn.Linear(8, 1))
            opt = torch.optim.AdamW(model.parameters(), lr=.001)
            return model, opt, make_scheduler(opt, config)

        def step(model, opt, scheduler, epoch):
            torch.manual_seed(25+epoch)
            model.train()
            used_lr = opt.param_groups[0]['lr']
            for _ in range(3):
                x, gt = torch.randn(4, 5), torch.randn(4, 1)
                opt.zero_grad(set_to_none=True)
                (model(x)-gt).square().mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
            scheduler.step()
            return used_lr

        full, opt, sched = setup()
        lrs = [step(full, opt, sched, e) for e in range(6)]
        model, optimizer, scheduler = setup()
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temp:
            for epoch in range(3):
                step(model, optimizer, scheduler, epoch)
            save_checkpoint(Path(temp)/'last.pth', model, optimizer, scheduler, epoch=3, run_signature='fixed')
            resumed, ropt, rsched = setup()
            _, state = latest_checkpoint(temp)
            restore(state, resumed, ropt, rsched, 'fixed')
            for epoch in range(state['epoch'], 6):
                step(resumed, ropt, rsched, epoch)
            for key, value in full.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
            self.assertEqual(sched.state_dict(), rsched.state_dict())
            for a, b in zip(opt.state.values(), ropt.state.values()):
                for k in a:
                    self.assertTrue(torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k])
            for a, b in zip(lrs, [.001, .001, .0001, .0001, .00001, .00001]):
                self.assertAlmostEqual(a, b)
            with self.assertRaises(ValueError):
                restore(state, resumed, ropt, rsched, 'changed')

    def test_newer_best_is_valid_resume_boundary(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temp:
            model = torch.nn.Linear(1, 1)
            opt = torch.optim.AdamW(model.parameters())
            sched = make_scheduler(opt, dict(lr_milestones=[2], lr_gamma=.1))
            save_checkpoint(Path(temp)/'last.pth', model, opt, sched, epoch=2)
            save_checkpoint(Path(temp)/'best.pth', model, opt, sched, epoch=3)
            path, state = latest_checkpoint(temp)
            self.assertEqual(path.name, 'best.pth')
            self.assertEqual(state['epoch'], 3)
            self.assertFalse(list(Path(temp).glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
