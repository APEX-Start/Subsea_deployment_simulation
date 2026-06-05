"""Minimal smoke tests for package imports."""


def test_core_imports():
    """Verify that the core simulation classes are importable."""
    from subsea_deployment_simulation.high_fidelity_sim import (
        DEFAULT_CONFIG,
        DynamicWireRopeSystem3D,
        VesselMotion,
        WireRopeSystem3D,
    )

    assert isinstance(DEFAULT_CONFIG, dict)
    assert DynamicWireRopeSystem3D is not None
    assert WireRopeSystem3D is not None
    assert VesselMotion is not None


def test_plotting_style_import():
    """Verify that the plotting style module is importable."""
    from subsea_deployment_simulation.plotting_style import apply_times_new_roman_style

    assert callable(apply_times_new_roman_style)


def test_package_version():
    """Verify the package has a version attribute or can be imported."""
    import subsea_deployment_simulation

    assert subsea_deployment_simulation is not None
