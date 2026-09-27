"""Merge a frozen LoRA reference, then select real acoustic weights to train."""

from peft.tuners.lora import LoraLayer


def merge_loaded_lora(model):
    """Merge the already loaded adapter into base weights, then freeze wrappers.

    Run a matched answer-level parity check before using the returned model for training.
    """
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, LoraLayer)]
    if not layers:
        raise RuntimeError('No loaded LoRA layers found')
    for name, module in layers:
        if module.merged:
            raise RuntimeError(f'LoRA already merged: {name}')
        if len(module.active_adapters) != 1:
            raise RuntimeError(f'Expected one active adapter at {name}: {module.active_adapters}')
        module.merge(safe_merge=True)
        if not module.merged:
            raise RuntimeError(f'LoRA merge did not complete: {name}')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {'merged_modules': len(layers), 'module_names': [name for name, _ in layers]}


def select_full_audio_parameters(model, last_layers=2):
    """Select both projections and optionally the final Conformer blocks."""
    tower = model.model.audio_tower
    if tower is None or len(tower.layers) != 12:
        raise RuntimeError('Expected the pinned 12-layer Gemma 4 E2B audio tower')
    if not 0 <= last_layers <= 12:
        raise ValueError(last_layers)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups = {
        'conformer': (list(tower.layers[-last_layers:].named_parameters())
                      if last_layers else []),
        'audio_output_projection': list(tower.output_proj.named_parameters()),
        'audio_language_projection': list(model.model.embed_audio.embedding_projection.named_parameters()),
    }
    selected = {}
    for group_name, entries in groups.items():
        active = []
        for name, parameter in entries:
            if 'lora_' in name:
                continue
            parameter.requires_grad_(True)
            active.append(parameter)
        if not active and group_name != 'conformer':
            raise RuntimeError(f'No full weights selected for {group_name}')
        selected[group_name] = active
    ids = [id(p) for parameters in selected.values() for p in parameters]
    if len(ids) != len(set(ids)):
        raise RuntimeError('A full-weight parameter appears in two optimizer groups')
    active_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if active_ids != set(ids):
        raise RuntimeError('Unexpected trainable weight outside selected acoustic groups')
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and ('lora_' in name or 'language_model' in name):
            raise RuntimeError(f'Unexpected trainable LoRA/decoder weight: {name}')
    return {
        'groups': selected,
        'counts': {name: sum(p.numel() for p in parameters)
                   for name, parameters in selected.items()},
        'trainable_total': sum(p.numel() for parameters in selected.values() for p in parameters),
        'last_layers': last_layers,
    }


def keep_frozen_lora_deterministic(model):
    """Disable dropout in the frozen reference adapter during acoustic training."""
    if any('lora_' in name and parameter.requires_grad
           for name, parameter in model.named_parameters()):
        raise RuntimeError('LoRA parameter unexpectedly trainable')
    count = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            for dropout in module.lora_dropout.values():
                dropout.eval()
                count += 1
    if count == 0:
        raise RuntimeError('No frozen LoRA dropout module found')
    return count
