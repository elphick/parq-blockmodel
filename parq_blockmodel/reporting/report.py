"""Reporting helpers for Parquet block models."""

from pathlib import Path
from typing import Optional, Union

from parq_tools import ParquetProfileReport


class BlockModelReport:
    """Thin wrapper around :class:`parq_tools.ParquetProfileReport`.

    The wrapper keeps track of the default output path used by
    :meth:`parq_blockmodel.blockmodel.ParquetBlockModel.create_report`
    and exposes a convenient :meth:`save` alias that writes HTML via
    ``save_html``.
    """

    def __init__(
        self,
        report: ParquetProfileReport,
        output_path: Path,
        columns: list[str],
        columns_per_batch: Optional[int],
        memory_budget_bytes: Optional[int] = None,
    ) -> None:
        self._report = report
        self.output_path = Path(output_path)
        self.columns = list(columns)
        self.columns_per_batch = columns_per_batch
        self.memory_budget_bytes = memory_budget_bytes

    def save(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Save the rendered report HTML and return the resolved path."""
        if output_path is None:
            path = self.output_path
        else:
            path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._report.save_html(path)
        self.output_path = path
        return path

    def save_html(self, output_path: Union[str, Path]) -> Path:
        """Compatibility wrapper mirroring the underlying report API."""
        return self.save(output_path)

    def show(self, notebook: bool = False):
        """Display the rendered report."""
        return self._report.show(notebook=notebook)

    def to_html(self) -> str:
        """Return the rendered report as HTML."""
        return self._report.to_html()

    @property
    def raw_report(self) -> ParquetProfileReport:
        """Access the underlying :class:`parq_tools.ParquetProfileReport`."""
        return self._report

    def __getattr__(self, item):
        return getattr(self._report, item)


