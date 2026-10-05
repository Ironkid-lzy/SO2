"""T001 compatibility shim for the `so2-repro` environment.

Python 3.10 removed the aliases in the `collections` top-level namespace that were
deprecated since 3.3 (PEP 585 / bpo-37324). The 2020-era dependency `namedlist==1.8`
(used by DI-engine's `ding/envs/env/base_env.py` and `ding/torch_utils/pytorch_util.py`)
still uses `collections.Mapping` / `collections.Sequence`, so importing `ding` fails.

This shim re-exposes the ABCs on `collections`. It changes no algorithm semantics: it
only restores names that the standard library itself removed.
"""
import collections
import collections.abc as _abc

for _name in (
    'Mapping', 'MutableMapping', 'Sequence', 'MutableSequence', 'Set', 'MutableSet',
    'Iterable', 'Iterator', 'Callable', 'Hashable', 'Sized', 'Container',
    'Collection', 'ByteString', 'Coroutine', 'Generator', 'Awaitable', 'AsyncIterable',
    'AsyncIterator', 'AsyncGenerator', 'Reversible', 'KeysView', 'ItemsView', 'ValuesView',
    'MappingView',
):
    if not hasattr(collections, _name) and hasattr(_abc, _name):
        setattr(collections, _name, getattr(_abc, _name))
