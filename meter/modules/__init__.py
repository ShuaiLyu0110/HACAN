"""HACAN model package with lazy model loading."""

__all__ = ["HACAN", "METERTransformerSS"]


def __getattr__(name):
    if name in __all__:
        from .meter_module import METERTransformerSS

        return METERTransformerSS
    raise AttributeError(name)
