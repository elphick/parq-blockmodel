from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class PbmAsset:
    path: Path
    name: str
    levels: tuple[tuple[str, str], ...]

    @property
    def level_map(self) -> dict[str, str]:
        return dict(self.levels)


class HivePbmCatalog:
    """Read-only hive-style PBM asset catalog."""

    def __init__(self, assets: Iterable[PbmAsset]) -> None:
        normalized_assets = [
            PbmAsset(
                path=asset.path.resolve(),
                name=asset.name,
                levels=tuple((str(k), str(v)) for k, v in asset.levels),
            )
            for asset in assets
        ]
        if not normalized_assets:
            raise ValueError("HivePbmCatalog requires at least one PBM asset.")
        self.assets = tuple(
            sorted(
                normalized_assets,
                key=lambda asset: (
                    tuple(asset.levels),
                    asset.name,
                    str(asset.path),
                ),
            )
        )
        seen_keys: list[str] = []
        for asset in self.assets:
            for key, _ in asset.levels:
                if key not in seen_keys:
                    seen_keys.append(key)
        self.level_keys = tuple(seen_keys)
        self._assets_by_path = {str(asset.path): asset for asset in self.assets}

    @classmethod
    def discover(cls, root_path: str | Path) -> "HivePbmCatalog":
        root = Path(root_path).resolve()
        assets: list[PbmAsset] = []
        for pbm_path in sorted(root.rglob("*.pbm")):
            relative_parent = pbm_path.parent.relative_to(root)
            levels: list[tuple[str, str]] = []
            for part in relative_parent.parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                if not key or not value:
                    continue
                levels.append((key, value))
            assets.append(PbmAsset(path=pbm_path, name=pbm_path.stem, levels=tuple(levels)))
        if not assets:
            raise FileNotFoundError(f"No .pbm files found under: {root}")
        return cls(assets)

    def find_by_path(self, pbm_path: str | Path) -> PbmAsset | None:
        resolved = Path(pbm_path).resolve()
        return self._assets_by_path.get(str(resolved))

    def level_options(self, key: str, selections: Mapping[str, str] | None = None) -> list[str]:
        if key not in self.level_keys:
            return []
        level_index = self.level_keys.index(key)
        constrained_keys = self.level_keys[:level_index]
        filtered_assets = self._filter_assets(selections=selections, keys=constrained_keys)
        values = []
        for asset in filtered_assets:
            level_map = asset.level_map
            if key in level_map:
                values.append(level_map[key])
        return sorted(set(values))

    def pbm_name_options(self, selections: Mapping[str, str] | None = None) -> list[str]:
        names = [asset.name for asset in self._filter_assets(selections=selections)]
        return sorted(set(names))

    def select_asset(self, selections: Mapping[str, str], name: str) -> PbmAsset:
        candidates = [
            asset
            for asset in self._filter_assets(selections=selections)
            if asset.name == name
        ]
        if not candidates:
            raise LookupError(
                "No PBM asset matches the provided hive selection and name "
                f"(name={name}, selections={dict(selections)})."
            )
        if len(candidates) > 1:
            paths = ", ".join(str(asset.path) for asset in candidates)
            raise LookupError(
                f"PBM name '{name}' is not unique for selections {dict(selections)}. "
                f"Candidates: {paths}"
            )
        return candidates[0]

    def _filter_assets(
        self,
        *,
        selections: Mapping[str, str] | None = None,
        keys: tuple[str, ...] | None = None,
    ) -> list[PbmAsset]:
        if not selections:
            return list(self.assets)
        constrained_keys = keys if keys is not None else tuple(selections.keys())
        filtered: list[PbmAsset] = []
        for asset in self.assets:
            level_map = asset.level_map
            is_match = True
            for key in constrained_keys:
                expected = selections.get(key)
                if expected in (None, ""):
                    continue
                if level_map.get(key) != expected:
                    is_match = False
                    break
            if is_match:
                filtered.append(asset)
        return filtered
