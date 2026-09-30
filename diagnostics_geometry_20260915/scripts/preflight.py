import bootstrap
from bootstrap import ROOT, DIAG
import hashlib
import json
import platform
import torch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


if __name__ == '__main__':
    out = DIAG/'runs/preflight'
    out.mkdir(parents=True, exist_ok=True)
    audit = {str(p.relative_to(ROOT)): sha(p) for folder in ['wCoefNet', 'woCoefNet']
             for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and p.suffix in ['.py', '.pth']}
    report = dict(python=platform.python_version(), torch=torch.__version__,
                  cuda_available=torch.cuda.is_available(), source_sha256=audit)
    for name in ['roma_outdoor.pth', 'dinov2_vitl14_pretrain.pth']:
        p = DIAG/'cache/torch/hub/checkpoints'/name
        sd = torch.load(p, map_location='cpu', weights_only=True)
        report[name] = dict(bytes=p.stat().st_size, sha256=sha(p), entries=len(sd))
        del sd
        print('WEIGHT_READ_OK', name, flush=True)
    from romatch import roma_outdoor
    report['roma_import'] = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not accessible; do not replace original CUDA pipeline with a CPU variant')
    torch.cuda.set_device(0)
    report['gpu'] = torch.cuda.get_device_name(0)
    from unittest.mock import patch
    import torchvision.models as models
    import network
    ctor = models.resnet.resnet18
    with patch.object(models.resnet, 'resnet18', side_effect=lambda *a, **k: ctor(weights=None)), patch.object(torch.cuda, 'is_available', return_value=False):
        net, coef = network.Network(), network.CoefNetwork()
    for module, path in [(net, ROOT/'woCoefNet/model_homo/epoch100_model.pth'),
                         (coef, ROOT/'wCoefNet/model_coef/epoch050_coefmodel.pth')]:
        module.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'], strict=True)
    report['rop_strict_load'] = True
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print('PREFLIGHT_OK', report['gpu'], flush=True)
