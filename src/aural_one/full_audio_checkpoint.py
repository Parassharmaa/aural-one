"""Resumable checkpoints for full-acoustic Gemma updates."""
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(directory,step,optimizer,metadata):
    directory=Path(directory)
    target=directory/f'step-{step:04d}'
    partial=directory/f'.step-{step:04d}.partial'
    if target.exists():
        raise FileExistsError(target)
    directory.mkdir(parents=True,exist_ok=True)
    if partial.exists():
        quarantine=directory/'quarantine'
        quarantine.mkdir(exist_ok=True)
        partial.rename(quarantine/f'step-{step:04d}-{time.time_ns()}.partial')
    partial.mkdir()
    weights={name:parameter.detach().cpu().contiguous()
             for name,parameter in optimizer.items}
    save_file(weights,partial/'acoustic_weights.safetensors')
    state={'optimizer':optimizer.state_dict(),
           'torch_rng':torch.get_rng_state(),
           'cuda_rng':torch.cuda.get_rng_state_all()}
    torch.save(state,partial/'training_state.pt')
    manifest={'step':step,'schema':'aural-one-full-audio-v1',
              'weights_sha256':sha256(partial/'acoustic_weights.safetensors'),
              'training_state_sha256':sha256(partial/'training_state.pt'),
              **metadata}
    (partial/'metadata.json').write_text(json.dumps(manifest,indent=2)+'\n')
    partial.rename(target)
    return target


def load_checkpoint(path,optimizer,expected_identity):
    path=Path(path)
    meta=json.loads((path/'metadata.json').read_text())
    if meta['schema']!='aural-one-full-audio-v1':
        raise RuntimeError('Wrong checkpoint schema')
    if meta['identity']!=expected_identity:
        raise RuntimeError('Base/adapter/code/config/schedule/data identity differs from checkpoint')
    if meta.get('screen_retention_reasons'):
        raise RuntimeError('Rejected checkpoint cannot be resumed as this run')
    if sha256(path/'acoustic_weights.safetensors')!=meta['weights_sha256'] or \
       sha256(path/'training_state.pt')!=meta['training_state_sha256']:
        raise RuntimeError('Checkpoint file hash mismatch')
    saved=load_file(str(path/'acoustic_weights.safetensors'))
    if set(saved)!=set(optimizer.names):
        raise RuntimeError('Acoustic weight names differ from checkpoint')
    state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    for name,parameter in optimizer.items:
        tensor=saved[name]
        if tensor.shape!=parameter.shape or not torch.equal(parameter.detach().cpu(),tensor):
            raise RuntimeError('BF16 weight/master restore mismatch: '+name)
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    return meta
