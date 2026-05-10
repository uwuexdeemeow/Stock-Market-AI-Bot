from _compat_import import export_public, load_reorganized_module

_module = load_reorganized_module(__name__, "labels.py")
export_public(_module, globals())
