# Changelog

All notable changes to this project will be documented in this file.

## 0.15.0 (2026-09-02)

### Added
- Introduce `CRSDef` structured type for CRS metadata: `parq_blockmodel.crs.CRSDef(authority, code, name=None, wkt=None)`.
- Add `ParquetBlockModel` helpers: `bm.crs`, `bm.crs_key`, and `bm.pyproj_crs`.
- Add migration guide and docs for `crs` metadata.

### Changed
- Migrate geometry constructors from `srs` -> `crs` keyword (breaking): accept string `"AUTH:code"`, dict `{"authority":..., "code":...}` or `CRSDef`.
- Geometry metadata now serializes structured `crs` and preserves legacy `srs` string for backward compatibility.
- WorldFrame exposes `srs` property (returns structured string or preserved legacy value).

### Fixed
- Preserve legacy arbitrary `srs` strings when parsing into structured CRS fails.



### Recent commits (from GitHub main branch)
- 87f0c22  Closes #135 extent footprints (#139) — added extent footprint support and missing files.
- bb8422d  Closes #134 - categorical footprints — categorical footprint feature and dependency pinning adjustments.
- 180756c  Closes #131 - removed imports related to extra deps — reduce optional import surface and fix tests.
- 54b09fd  Update coverage badge [skip ci]
- 5ad0d6b  Update coverage badge [skip ci]

*Generated from the repository's recent commits. For a full chronological changelog, enable Git in this working directory or provide repository access.*