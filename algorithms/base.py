import abc

class BaseTrainer(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            args,
            env,
            buffer,
            filename
    ):
        self.args = args
        self.buffer = buffer
        self.filename = 0
        self.step = 0

    def train(self, batch):
        raise NotImplementedError('train must implemented by inherited class')
    
    def save(self):
        raise NotImplementedError('save must implemented by inherited class')

    def load(self):
        raise NotImplementedError('save must implemented by inherited class')

