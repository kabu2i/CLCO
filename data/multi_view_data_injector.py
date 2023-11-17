class MultiViewDataInjector(object):
    """Create multi views of the same signal"""
    def __init__(self, *args):
        self.transforms = args[0]

    def __call__(self, sample):
        return [transform(sample) for transform in self.transforms]
