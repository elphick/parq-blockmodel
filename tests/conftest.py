"""Root conftest for parq-blockmodel tests.

Marker conventions (also registered in pyproject.toml):
- integration  : tests that cross module boundaries (e.g., file I/O + mesh export)
- slow         : expensive tests excluded from the default fast lane
- gui          : tests that require an interactive display or rendering backend

Default fast lane (excludes slow + gui)::

    pytest -m "not slow and not gui"

Include integration tests::

    pytest -m "integration and not gui"
"""

