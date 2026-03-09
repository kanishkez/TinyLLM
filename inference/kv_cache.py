import torch


class KVCache:

    def __init__(self):
        self.cache = []

    def update(self, layer, k, v):

        if len(self.cache) <= layer:
            self.cache.append((k, v))
        else:
            pk, pv = self.cache[layer]
            self.cache[layer] = (
                torch.cat([pk, k], dim=2),
                torch.cat([pv, v], dim=2),
            )

        return self.cache[layer]

    def get(self, layer):
        return self.cache[layer] if layer < len(self.cache) else None

    def to_list(self):
        return self.cache

    def clear(self):
        self.cache = []
