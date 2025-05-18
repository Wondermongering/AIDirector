"""Simple plugin used for testing."""

called = {}

def sample_hook(arg, *, kw=None):
    called['arg'] = arg
    called['kw'] = kw
    return 'ok'
