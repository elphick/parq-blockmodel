import json
import numpy as np
import pyvista as pv

def store_mapping_dict(mesh: pv.DataSet, name: str, mapping: dict):
    """
    Store a mapping dictionary in the field_data of a PyVista mesh as a JSON string.

    Parameters:
    - mesh: pyvista.DataSet (e.g., PolyData, UnstructuredGrid)
    - name: str, the key under which to store the mapping (e.g., "rock_type_map")
    - mapping: dict, the dictionary to store (must be JSON-serializable)
    """
    json_str = json.dumps(mapping)
    mesh.field_data[name + "_json"] = np.array([json_str])

def load_mapping_dict(mesh: pv.DataSet, name: str) -> dict:
    """
    Load a mapping dictionary from a PyVista mesh's field_data.

    Parameters:
    - mesh: pyvista.DataSet
    - name: str, the key under which the mapping was stored

    Returns:
    - dict, the decoded mapping
    """
    return json.loads(mesh.field_data[name + "_json"][0])
