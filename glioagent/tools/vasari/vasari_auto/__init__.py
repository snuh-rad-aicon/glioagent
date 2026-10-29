# Lazy import to avoid loading heavy dependencies (ANTs, scipy, sklearn)
# at package-import time.  Downstream code should import explicitly:
#     from vasari_auto.vasari_auto import get_vasari_features

def __getattr__(name):
    if name == 'vasari_auto':
        from vasari_auto import vasari_auto as _mod  # noqa: F811
        return _mod
    raise AttributeError(f"module 'vasari_auto' has no attribute {name!r}")