from __future__ import annotations

import math

import numpy as np
import pyvista as pv


class CustomPlotter(pv.Plotter):
    """
    A custom PyVista Plotter with Z-up enforcement, picking, and directional camera view.

    Examples
    --------
    >>> grid = pv.ImageData(dimensions=(4, 4, 4), spacing=(1, 1, 1), origin=(0, 0, 0))
    >>> grid.cell_data["block_id"] = np.arange(grid.n_cells)
    >>>
    >>> plotter = CustomPlotter()
    >>> plotter.add_mesh(grid, show_edges=True)
    >>> plotter.set_directional_view(direction='WSW', elevation_deg=30)
    >>> plotter.add_axes()
    >>> plotter.show()
    """

    HELP_TEXT_NAME = "help_overlay"

    HOTKEYS = {
        "h": "Show/hide this help",
        "p": "Toggle cell picking",
        "z": "Z-up rotation (hold)",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hotkey_pressed = {'z': False}
        self.picking_enabled = False
        self.slicer_enabled = False
        self.help_visible = False
        self._show_help_overlay()
        self._setup_callbacks()

    def _key_press_callback(self, obj, event):
        key = obj.GetKeySym()
        if key == 'z':
            self.hotkey_pressed['z'] = True
        if key == 'p':
            if not self.picking_enabled:
                self.enable_general_picking()
                self.picking_enabled = True
            else:
                self.disable_picking()
                self.remove_actor("cell_info_text")
                self.picking_enabled = False
        if key == 'h':
            if not self.help_visible:
                self._show_help_overlay()
                self.help_visible = True
            else:
                self.remove_actor(self.HELP_TEXT_NAME)
                self.help_visible = False

    def _show_help_overlay(self):
        lines = []
        for k, v in self.HOTKEYS.items():
            lines.append(f"[{k.upper()}]  {v}")
        help_text = "\n".join(lines)
        self.add_text(
            help_text,
            position="right_edge",
            font_size=10,
            name=self.HELP_TEXT_NAME,
            color="white",
            shadow=True,
            viewport=True,
        )

    def set_directional_view(
            self,
            direction='WSW',
            radius_factor=4.0,
            elevation_deg=30,
            azimuth_deg=None
    ):
        # Map compass directions to azimuth angles (degrees)
        direction_azimuth = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        if azimuth_deg is None:
            azimuth_deg = direction_azimuth.get(direction.upper(), 247.5)  # Default to WSW

        bounds = self.bounds
        center = [
            (bounds[1] + bounds[0]) / 2,
            (bounds[3] + bounds[2]) / 2,
            (bounds[5] + bounds[4]) / 2
        ]
        max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        r = max_dim * radius_factor

        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        x = center[0] + r * math.cos(elevation) * math.cos(azimuth)
        y = center[1] + r * math.cos(elevation) * math.sin(azimuth)
        z = center[2] + r * math.sin(elevation)
        self.camera_position = [(x, y, z), center, (0, 0, 1)]

    def _setup_callbacks(self):
        iren = self.iren
        iren.add_observer("KeyPressEvent", self._key_press_callback)
        iren.add_observer("KeyReleaseEvent", self._key_release_callback)
        iren.add_observer("InteractionEvent", self._enforce_z_up_during_interaction)

    def _key_release_callback(self, obj, event):
        key = obj.GetKeySym()
        if key == 'z':
            self.hotkey_pressed['z'] = False

    def _enforce_z_up_during_interaction(self, obj, event):
        if self.hotkey_pressed['z']:
            self.camera.SetViewUp(0, 0, 1)
            self.render()

    def enable_general_picking(self):
        def cell_callback(picked_cell):
            text_name = "cell_info_text"
            if text_name in self.actors:
                self.remove_actor(text_name)
            if hasattr(picked_cell, "n_cells") and picked_cell.n_cells == 1:
                for mesh in self.meshes:  # self.meshes is a list
                    if "vtkOriginalCellIds" in picked_cell.cell_data:
                        cell_id = int(picked_cell.cell_data["vtkOriginalCellIds"][0])
                        centroid = mesh.cell_centers().points[cell_id]
                        centroid_str = f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"
                        values = {attr: mesh.cell_data[attr][cell_id] for attr in mesh.cell_data}
                        msg = f"Cell ID: {cell_id}, {centroid_str}, " + ", ".join(
                            f"{k}: {v}" for k, v in values.items())
                        break
                else:
                    msg = "Picked cell, but could not resolve mesh/cell data."
                self.add_text(msg, position="upper_left", font_size=12, name=text_name)
            else:
                self.add_text("No valid cell picked.", position="upper_left", font_size=12, name=text_name)

        self.disable_picking()  # Always disable before enabling
        self.enable_cell_picking(callback=cell_callback, show_message=False, through=False)
