"""Reuse pad-contact reads only inside controlled environment step calls."""
from contextlib import contextmanager
from functools import wraps


@contextmanager
def contact_read_scope(base):
    outer = not base._pad_contact_cache_active
    if outer:
        base._invalidate_contact_cache()
        base._pad_contact_cache_active = True
    try:
        yield
    finally:
        if outer:
            base._pad_contact_cache_active = False
            base._invalidate_contact_cache()


def cached_contact_reads(function):
    @wraps(function)
    def step(self, *args, **kwargs):
        base = getattr(self, 'base_env', self)
        if not base.config.contact_read_cache_enabled:
            return function(self, *args, **kwargs)
        with contact_read_scope(base):
            return function(self, *args, **kwargs)
    return step
