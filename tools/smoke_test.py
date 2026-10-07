"""CPU checks for contact pretraining, policy gradients, and checkpoint portability."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from models.policy.contact_policy import ContactPolicy
from models.policy.contact_autoencoder import load_force_vq
from trainers.pretrain_trainer import build_vq_model
from trainers.policy_trainer import get_checkpoint_state
from utils.checkpoint_util import load_policy_from_checkpoint
from utils.normalizer import FieldNormalizer, MultiFieldNormalizer
from utils.train_utils import build_canonical_config, validate_training_paths


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(7)
    pre = build_canonical_config('config/pretrain.yaml')
    cfg = build_canonical_config('config/policy.yaml')
    for config in (pre, cfg):
        try:
            validate_training_paths(config)
        except ValueError as exc:
            assert 'data.root_dir' in str(exc)
        else:
            raise AssertionError('Empty training paths must fail before model construction.')

    vq = build_vq_model(pre, torch.device('cpu'))
    pos = torch.randn(1, 128, 3)
    force = torch.randn(1, 128, 12)
    tactile = torch.randn(1, 128, 35, 20, 6)
    out = vq(pos=pos, force=force, tactile=tactile)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in vq.parameters())
    vq.zero_grad(set_to_none=True)
    norm = MultiFieldNormalizer()
    for name, dim in [('mode_pos', 3), ('force', 12), ('tactile', 6)]:
        norm[name] = FieldNormalizer(torch.ones(dim), torch.zeros(dim))
    bundle = {
        'config': pre,
        'force_vq_state_dict': vq.state_dict(),
        'force_normalizer_state_dict': norm.state_dict(),
    }
    restored_vq, _, _ = load_force_vq(bundle, 'cpu')
    vq.eval()
    with torch.no_grad():
        a = vq(pos=pos, force=force, tactile=tactile)['loss']
        b = restored_vq(pos=pos, force=force, tactile=tactile)['loss']
    torch.testing.assert_close(a, b, rtol=0, atol=0)

    policy_cfg = cfg['model']['policy']
    policy_cfg['image_pretrained'] = False
    policy_cfg['down_dims'] = [32, 64]
    policy_cfg['diffusion_step_embed_dim'] = 32
    policy_cfg['inference']['num_inference_steps'] = 2
    policy = ContactPolicy.from_cfg(cfg)
    policy.set_trainable_force_vq(vq, norm)
    batch = {
        'obs': {
            'image': torch.randint(0, 256, (1, 1, 2, 3, 224, 224), dtype=torch.uint8),
            'state': torch.randn(1, 8, 10),
            'force': torch.randn(1, 8, 12),
            'tactile': torch.randn(1, 8, 35, 20, 6),
        },
        'action': torch.randn(1, 32, 10),
        'future_force': torch.randn(1, 8, 12),
        'force_vq_reference_pos': pos,
        'force_vq_reference_force': force,
        'force_vq_reference_tactile': tactile,
        'force_vq_reference_phase': torch.linspace(0, 1, 128).unsqueeze(0),
    }
    result = policy.forward_train(batch)
    assert torch.isfinite(result['loss'])
    assert result['metrics']['force_consistency_loss'] > 0
    result['loss'].backward()
    for name, module in [('reference', policy.trainable_force_vq),
                         ('action', policy.velocity_net),
                         ('force', policy.force_consistency_head)]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()), name

    obs = dict(batch['obs'])
    obs.update({k: v for k, v in batch.items() if k.startswith('force_vq_')})
    policy.eval()
    with torch.no_grad():
        torch.manual_seed(11)
        expected = policy.predict_action(obs)['action']
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'policy.pt'
        torch.save(get_checkpoint_state(policy, None, 0, cfg, bundle), checkpoint)
        loaded, _ = load_policy_from_checkpoint(checkpoint, torch.device('cpu'))
        with torch.no_grad():
            torch.manual_seed(11)
            actual = loaded.predict_action(obs)['action']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (1, 32, 10)
    print('PASS: FPT pretraining, continuous policy gradients, force prediction, and checkpoint round trips.')


if __name__ == '__main__':
    main()
