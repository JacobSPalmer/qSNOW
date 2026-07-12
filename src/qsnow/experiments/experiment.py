from typing import Dict


class Experiment:
    config: Dict = {}

    def __init__(self, **kwargs):
        self.config = kwargs
