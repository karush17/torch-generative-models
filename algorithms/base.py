"""Implements the base learner."""

from typing import Any

import abc

class BaseTrainer(object, metaclass=abc.ABCMeta):
    """Abstract base traininer class."""
    def __init__(
            self,
            args: Any,
            env: str,
            buffer: Any,
            filename: str
    ):
        """Initializes the base object."""
        self.args = args
        self.buffer = buffer
        self.filename = 0
        self.step = 0

    def train(self, batch: Any):
        """Trains the learner."""
        raise NotImplementedError('train must implemented by inherited class')

    def save(self):
        """Saves the model."""
        raise NotImplementedError('save must implemented by inherited class')

    def load(self):
        """Loads the model."""
        raise NotImplementedError('save must implemented by inherited class')
