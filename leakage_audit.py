from _compat_import import export_public, load_reorganized_module

_module = load_reorganized_module(__name__, "leakage_audit.py")
export_public(_module, globals())

if __name__ == "__main__":
    import sys

    sys.exit(_module.main())
