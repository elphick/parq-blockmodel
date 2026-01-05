"""
Helper to manager the browser used for scraping plotly to get thumbnails
"""
from parq_blockmodel.utils.cicd import is_github_runner


def autoset_plotly_browser() -> None:
    """
    Set the browser to be used by plotly for thumbnail generation.
    """
    if is_github_runner():  # (linux)
        import plotly.io as pio
        pio.get_chrome()
    else:  # default to edge (windows)
        import os
        os.environ["BROWSER_PATH"] = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
