from parq_blockmodel.utils.orientation_utils import generate_block_model_with_ellipse, calculate_orientation


def test_orientation_with_ellipse():
    # Parameters for the block model
    x_range = (0, 100)
    y_range = (0, 100)
    z_range = (0, 50)
    spacing = 10
    ellipse_center = (50, 50)
    semi_major_axis = 80
    semi_minor_axis = 30
    orientation_angle = 45  # degrees
    grade_min = 50
    grade_max = 70

    # Generate the block model
    block_model = generate_block_model_with_ellipse(
        x_range, y_range, z_range, spacing,
        ellipse_center, semi_major_axis, semi_minor_axis,
        orientation_angle, grade_min, grade_max
    )

    # Calculate orientation
    orientation_df = calculate_orientation(block_model)

    # Validate results
    assert not orientation_df.empty, "Orientation calculation returned an empty DataFrame."
    assert 'bearing' in orientation_df.columns, "Missing 'bearing' in orientation results."
    assert 'dip' in orientation_df.columns, "Missing 'dip' in orientation results."
    assert 'plunge' in orientation_df.columns, "Missing 'plunge' in orientation results."

    # Print results for debugging
    print(orientation_df.head())