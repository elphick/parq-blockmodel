import subprocess
import sys
from pathlib import Path

_CHROME_READY = False

def _get_plotly_chrome_dir() -> Path:
    import plotly.io as pio
    chrome_dir = Path(pio.__file__).parent / "_kaleido" / "chrome-bin"
    print(f"[plotly_chrome] Using chrome dir: {chrome_dir}")
    return chrome_dir


def _get_plotly_chrome_exe() -> Path | None:
    """
    Return the full path to plotly's managed chrome.exe, if present.
    """
    chrome_dir = _get_plotly_chrome_dir()
    if not chrome_dir.exists():
        print(f"[plotly_chrome] Chrome dir does not exist yet: {chrome_dir}")
        return None

    # Subdir normally `chrome-win64/chrome.exe` on Windows
    exe = chrome_dir / "chrome-win64" / "chrome.exe"
    if exe.exists():
        print(f"[plotly_chrome] chrome.exe found at: {exe}")
        return exe

    # Fallback: search, as before
    candidates = list(chrome_dir.rglob("chrome.exe"))
    print(f"[plotly_chrome] Fallback chrome.exe candidates: {candidates}")
    return candidates[0] if candidates else None


def _install_plotly_chrome(chrome_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "plotly.io._sg_cli_get_chrome",
        "--path",
        str(chrome_dir),
        "-y",
    ]
    print(f"[plotly_chrome] Running installer: {cmd}")
    subprocess.run(cmd, check=True)


def ensure_chrome_installed() -> None:
    global _CHROME_READY
    if _CHROME_READY:
        print("[plotly_chrome] Chrome already marked as ready in this process")
        return

    print("[plotly_chrome] ensure_chrome_installed() called")
    exe = _get_plotly_chrome_exe()
    if exe and exe.exists():
        print(f"[plotly_chrome] Existing Chrome found at: {exe}")
        _CHROME_READY = True
        return

    chrome_dir = _get_plotly_chrome_dir()
    chrome_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plotly_chrome] Created/ensured chrome dir: {chrome_dir}")

    _install_plotly_chrome(chrome_dir)

    exe = _get_plotly_chrome_exe()
    if not exe or not exe.exists():
        raise RuntimeError(
            f"[plotly_chrome] plotly_get_chrome ran but Chrome was not found in {chrome_dir}"
        )

    # set the BROWSER_PATH env var to try to use edge/chrome from plotly
    # import os
    # os.environ["BROWSER_PATH"] = str(exe.parent)
    # print(f"[BROWSER_PATH] env var set to: {str(exe.parent)}")

    _CHROME_READY = True
    print(f"[plotly_chrome] Chrome successfully installed at: {exe}")


def get_chrome_exe_path() -> Path | None:
    """
    Public helper: ensure Chrome is installed and return the exe path.
    """
    ensure_chrome_installed()
    return _get_plotly_chrome_exe()


if __name__ == "__main__":
    ensure_chrome_installed()
    print("Chrome path:", get_chrome_exe_path())
