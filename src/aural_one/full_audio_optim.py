"""FP32 master-weight AdamW for BF16 full-acoustic Gemma parameters.

The model's forward weights stay BF16. Each microbatch gradient is copied into
FP32 master gradients, then model grads are cleared to avoid BF16 accumulation.
The optimizer and its moment tensors operate on the FP32 master parameters.
"""

import math

import torch


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


class MasterWeightAdamW:
    def __init__(self, model, selected, block_lr, projection_lr, weight_decay=0.01):
        active = {id(p) for group in selected['groups'].values() for p in group}
        self.items = [(name, parameter) for name, parameter in model.named_parameters()
                      if id(parameter) in active]
        if len(self.items) != len(active):
            raise RuntimeError('Selected parameter names are not unique')
        if any(parameter.dtype != torch.bfloat16 for _, parameter in self.items):
            raise RuntimeError('Expected BF16 model weights for the FP32-master optimizer')
        self.names = [name for name, _ in self.items]
        self.master = {name: torch.nn.Parameter(parameter.detach().float().clone())
                       for name, parameter in self.items}
        conformer_ids = {id(p) for p in selected['groups']['conformer']}
        blocks, projections = [], []
        for name, parameter in self.items:
            (blocks if id(parameter) in conformer_ids else projections).append(self.master[name])
        if not projections:
            raise RuntimeError('Missing the audio projection optimizer group')
        groups = ([{'params': blocks, 'lr': block_lr, 'name': 'conformer'}]
                  if blocks else [])
        groups.append({'params': projections, 'lr': projection_lr, 'name': 'projections'})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
        self.begin_step()

    def begin_step(self):
        self.optimizer.zero_grad(set_to_none=True)
        for _, parameter in self.items:
            parameter.grad = None
        self.microbatches = 0

    def accumulate(self):
        """Call immediately after each scaled microbatch loss.backward()."""
        received = 0
        for name, parameter in self.items:
            gradient = parameter.grad
            if gradient is None:
                continue
            converted = gradient.detach().float()
            master = self.master[name]
            if master.grad is None:
                master.grad = converted.clone()
            else:
                master.grad.add_(converted)
            parameter.grad = None
            received += 1
        if received == 0:
            raise RuntimeError('No full-acoustic gradient reached the optimizer')
        self.microbatches += 1
        return received

    def step(self, max_grad_norm=1.0):
        if self.microbatches <= 0:
            raise RuntimeError('No microbatch gradients accumulated')
        gradients = [p.grad for p in self.master.values() if p.grad is not None]
        if len(gradients) != len(self.items):
            raise RuntimeError('Some selected full weights received no gradient')
        norm = torch.nn.utils.clip_grad_norm_(list(self.master.values()), max_grad_norm)
        if not math.isfinite(float(norm)):
            raise RuntimeError('Nonfinite FP32 master gradient norm')
        self.optimizer.step()
        with torch.no_grad():
            for name, parameter in self.items:
                parameter.copy_(self.master[name].to(parameter.dtype))
        return float(norm)

    def state_dict(self):
        return {'names': self.names,
                'master': {name: parameter.detach().cpu().clone()
                           for name, parameter in self.master.items()},
                'optimizer': _cpu_tree(self.optimizer.state_dict())}

    def load_state_dict(self, state):
        if state['names'] != self.names:
            raise RuntimeError('FP32 master names/order differ from checkpoint')
        with torch.no_grad():
            for name, parameter in self.items:
                self.master[name].copy_(state['master'][name].to(parameter.device))
                parameter.copy_(self.master[name].to(parameter.dtype))
        self.optimizer.load_state_dict(state['optimizer'])
        self.begin_step()

    def bytes_by_dtype(self):
        tensors = [p for p in self.master.values()]
        tensors += [value for state in self.optimizer.state.values()
                    for value in state.values() if isinstance(value, torch.Tensor)]
        counts = {}
        for tensor in tensors:
            key = str(tensor.dtype)
            counts[key] = counts.get(key, 0) + tensor.numel() * tensor.element_size()
        return counts
