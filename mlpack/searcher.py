import numpy as np


def simple_generator(values: list, key: str):
    def generator():
        d = {}
        for v in values:
            d[key] = v
            yield d
    return generator


class ParamSearcher:

    def __init__(self, model, generator, callbacks: dict):
        self.model, self.generator = model, generator
        self.callbacks = callbacks

    def run(self, x_train, y_train, x_valid, y_valid):
        metrics = {
            key: [] for key in self.callbacks.keys()
        }
        for variables in self.generator():
            self.model.fit(x_train, y_train, **variables)
            pred = self.model(x_valid)
            for key in metrics.keys():
                metrics[key].append(self.callbacks[key](pred, y_valid))

        self.metrics = metrics.copy()

    def best(self, key: str, order: str = 'asc'):
        assert order in ['asc', 'desc']
        assert key in self.metrics.keys()
        m = np.array(self.metrics[key])
        if order == 'asc':
            return m.max(), m.argmax()
        elif order == 'desc':
            return m.min(), m.argmin()
