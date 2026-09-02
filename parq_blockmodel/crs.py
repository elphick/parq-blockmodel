from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Dict


@dataclass(frozen=True)
class CRSDef:
    """Structured CRS definition.

    Identity is (authority, code). Name and WKT are optional and are not used
    for equality or hashing.
    """

    authority: str
    code: Any
    name: Optional[str] = None
    wkt: Optional[str] = None

    def __post_init__(self):
        # Normalize authority to upper-case
        object.__setattr__(self, "authority", str(self.authority).upper())
        # Keep code as int when possible, else string
        try:
            c = int(self.code)
        except Exception:
            c = str(self.code)
        object.__setattr__(self, "code", c)

    def to_string(self) -> str:
        return f"{self.authority}:{self.code}"

    def to_metadata(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"authority": self.authority, "code": self.code}
        if self.name is not None:
            payload["name"] = self.name
        if self.wkt is not None:
            payload["wkt"] = self.wkt
        return payload

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "CRSDef":
        if not isinstance(meta, dict):
            raise TypeError("CRS metadata must be a dict")
        authority = meta.get("authority")
        code = meta.get("code")
        name = meta.get("name")
        wkt = meta.get("wkt")
        if authority is None or code is None:
            raise ValueError("CRS metadata requires 'authority' and 'code'")
        return cls(authority=authority, code=code, name=name, wkt=wkt)

    @classmethod
    def from_string(cls, s: str) -> "CRSDef":
        if not isinstance(s, str):
            raise TypeError("CRS string must be a str")
        parts = s.split(":", 1)
        if len(parts) != 2:
            raise ValueError("CRS string must be of the form 'AUTH:code'")
        auth, code = parts[0].strip().upper(), parts[1].strip()
        try:
            code_val = int(code)
        except Exception:
            code_val = code
        return cls(authority=auth, code=code_val)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CRSDef):
            return NotImplemented
        return (self.authority.upper(), str(self.code)) == (other.authority.upper(), str(other.code))

    def __hash__(self) -> int:
        return hash((self.authority.upper(), str(self.code)))

    def __repr__(self) -> str:
        return f"CRSDef(authority={self.authority!r}, code={self.code!r}, name={self.name!r})"