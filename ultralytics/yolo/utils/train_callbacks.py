import os
import torch
from torch.optim import lr_scheduler


class CallbacksManager:
    def __init__(self, list_of_callbacks=None):
        self.callbacks_list = [] if list_of_callbacks is None else list_of_callbacks

    def on_epoch_end(self, **kwargs):
        [c.on_epoch_end(**kwargs) for c in self.callbacks_list]


class Callback:
    def on_epoch_end(self, *args, **kwargs):
        pass


class ReduceLROnPlateau(Callback):
    def __init__(self, optimizer, factor=0.1, patience=5, verbose=True, min_lr=1e-7, monitor="val/dfl_loss"):
        self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=factor,
                                                        patience=patience,
                                                        verbose=verbose,
                                                        min_lr=min_lr,
                                                        threshold=0.002)
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get(self.monitor)
        self.scheduler.step(val_loss)


class LRCallback(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.step()

