"""
Upsample Example
================

A basic example of the upsample function
"""

import numpy as np
import pandas as pd

from parq_blockmodel.reblocking.upsample import upsample_attributes

# Create synthetic 3D arrays
np.random.seed(42)
shape = (4, 4, 4)
grade = np.random.rand(*shape)
density = np.random.rand(*shape)
dry_mass = np.random.rand(*shape) * 10
volume = np.full(shape, 5.0)  # scalar volume
# Convert rock_types to categorical codes
rock_types_str = np.random.choice(['shale', 'sandstone', 'limestone'], size=np.prod(shape))
rock_types_cat = pd.Categorical(rock_types_str)
rock_types = rock_types_cat.codes.reshape(shape)
rock_type_categories = rock_types_cat.categories

# Define attributes and config
attributes = {
    'grade': grade,
    'density': density,
    'dry_mass': dry_mass,
    'volume': volume,
    'rock_type': rock_types
}

interpolation_config = {
    'grade': 'linear',
    'density': 'linear',
    'dry_mass': 'linear',
    'volume': 'linear',
    'rock_type': 'nearest'
}

# Upsampling factors
fx, fy, fz = 2, 2, 2

# Run upsampling
upsampled = upsample_attributes(attributes, fx, fy, fz, interpolation_config)

# Restore rock_type codes to categories for display
upsampled['rock_type'] = pd.Categorical.from_codes(
    list(upsampled['rock_type'].ravel()), rock_type_categories
).reshape(upsampled['rock_type'].shape)

# Display results
for key, value in upsampled.items():
    print(f"\n{key} (shape: {value.shape}):\n{value}")