import numpy as np

class LinearDecay:
    def __init__(self,
                 lr_init: float,
                 lr_end: float,
                 max_epochs: int):
        self.lr_init = lr_init
        self.lr_end = lr_end
        self.max_epochs = max_epochs

    def caculate_lr(self, epoch, batch_idx):
        return self.lr_init - epoch*(self.lr_init - self.lr_end)/self.max_epochs

class CosineAnnealing:
    def __init__(self,
                 base_lr: float,
                 warmup_epochs: int,
                 steps_per_epoch: int,
                 max_epochs: int):
        self.lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
    
    def calculate_lr(self, epoch, batch_idx):
        steps = epoch*self.steps_per_epoch + batch_idx
        if steps < self.warmup_epochs*self.steps_per_epoch:
            lr = self.lr*steps/(self.warmup_epochs*self.steps_per_epoch)
        else:
            lr = 0.5*self.lr*(1 + np.cos(np.pi*(steps - self.warmup_epochs*self.steps_per_epoch)/(self.max_epochs*self.steps_per_epoch - self.warmup_epochs*self.steps_per_epoch)))
        return lr
