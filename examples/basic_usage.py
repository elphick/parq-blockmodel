"""
Basic usage
===========

A simple example demonstrating how to use mass-composition.

Design notes:
Once data is loaded chemical analyte names and H2O will conform to the internal standard.

"""
import numpy as np
import plotly
import plotly.graph_objects as go

# %%
#
# This is a demo
# --------------
#
# We create a simple interactive plot

fig = go.Figure(data=go.Scatter(
    y=np.random.randn(500),
    mode='markers',
    marker=dict(
        size=16,
        color=np.random.randn(500),  # set color equal to a variable
        colorscale='Viridis',  # one of plotly colorscales
        showscale=True
    )
))

# noinspection PyTypeChecker
plotly.io.show(fig)  # this call to show will set the thumbnail for use in the gallery
