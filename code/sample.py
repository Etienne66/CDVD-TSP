import torch
from pathlib import Path

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

kwargs = {'lr':             5.0e-6}
optimizer = torch.optim.AdamW(model.parameters(), **kwargs)

kwargs = {'base_lr':        5.0e-6,
          'max_lr':         2.5e-4,
          'cycle_momentum': False,
          'step_size_up':   100_000,
          'mode':           'exp_range',
          'gamma':          0.99999979673105275299293754990642}
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
scheduler_dict = scheduler.state_dict()

torch.save(scheduler_dict , Path('./checkpoint.pt'))
