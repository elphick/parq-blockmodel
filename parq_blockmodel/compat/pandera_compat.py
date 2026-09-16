import os

# Disable pandera import warning if pandera supports that env var; set before importing pandera
os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

try:
    # Prefer the explicit pandas-specific API when available (newer pandera)
    from pandera.pandas import DataFrameSchema, Column, Check  # type: ignore[import]
    import pandera.pandas as _pandera_module  # type: ignore[import]
except Exception:
    # Fall back to the top-level pandera for older releases
    from pandera import DataFrameSchema, Column, Check  # type: ignore[import]
    import pandera as _pandera_module  # type: ignore[import]

# Public exports for convenience
__all__ = ["DataFrameSchema", "Column", "Check", "pa"]

# Expose the underlying pandera module as `pa` for callers that expect a module
pa = _pandera_module
