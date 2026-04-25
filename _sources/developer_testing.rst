Developer Testing Guide
=======================

This page describes how we use pytest markers and how to quarantine legacy tests
while keeping visibility.

Marker conventions
------------------

Use the following markers consistently:

- ``@pytest.mark.integration``
  For tests that exercise multiple layers together (for example writing/reading
  ``.pbm`` files plus mesh export).

- ``@pytest.mark.slow``
  For expensive tests that are valuable but too costly for the default unit test lane.

- ``@pytest.mark.gui``
  For tests that require an interactive display, browser, or rendering backend.

Keep unit tests (fast, deterministic, no external dependencies) unmarked.

Example usage::

   import pytest

   @pytest.mark.integration
   def test_pbm_mesh_roundtrip(...):
       ...

   @pytest.mark.slow
   def test_large_spatial_encoding_stress(...):
       ...

   @pytest.mark.gui
   def test_interactive_plot(...):
       ...

How to run subsets
------------------

Default fast lane::

   pytest -m "not slow and not gui"

Include integration tests::

   pytest -m "integration and not gui"

Run only GUI tests locally::

   pytest -m "gui"

Run everything::

   pytest

Quarantining legacy tests
-------------------------

If a test is currently valuable for visibility but not ready for the default lane,
quarantine it rather than deleting it.

Recommended approach:

1. Keep the test under ``tests/`` but mark it clearly (for example ``skip`` or a
   dedicated marker when introduced).
2. Add a short reason and an issue/work item reference in the decorator or comment.
3. Define an explicit exit condition (what must be implemented to unquarantine it).

Current quarantine candidates include:

- ``tests/blockmodel/test_dxf_export.py`` (pending DXF workflow completion)
- GUI-heavy cases in ``tests/blockmodel/test_toy_blockmodel.py``
- skipped structured-grid accessor case in ``tests/pyvista/test_accessor.py``

Guideline
---------

Prefer a small number of focused test files by concern (ingest, indexing,
visualization, integration) and avoid single monolithic files.


