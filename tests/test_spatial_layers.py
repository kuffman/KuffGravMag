"""Independent CRS and provenance checks for local XYZ layers."""

import hashlib

import numpy as np
import pytest

from engine.spatial import load_xyz_layer, spatial_layer_csv, transform_coordinates


def test_crs_transform_uses_xy_order_for_synthetic_ghana_utm_coordinates():
    longitude, latitude = transform_coordinates(
        [500_000.0], [663_204.567], "EPSG:32630", "EPSG:4326"
    )

    assert longitude[0] == pytest.approx(-3.0, abs=2e-6)
    assert latitude[0] == pytest.approx(6.0, abs=2e-6)


def test_local_xyz_layer_transforms_coordinates_and_records_source_provenance():
    payload = (
        b"# synthetic teaching points\n"
        b"easting,northing,gravity_mgal\n"
        b"500000,663204.567,0.25\n"
        b"500100,663204.567,0.50\n"
        b"500000,663304.567,0.75\n"
    )
    layer = load_xyz_layer(
        payload,
        source_crs="EPSG:32630",
        target_crs="EPSG:4326",
        units="mGal",
        name="synthetic Ghana classroom grid",
    )

    assert layer.name == "synthetic Ghana classroom grid"
    assert layer.source_crs == "EPSG:32630"
    assert layer.target_crs == "EPSG:4326"
    assert layer.units == "mGal"
    assert layer.source_sha256 == hashlib.sha256(payload).hexdigest()
    assert layer.x[0] == pytest.approx(-3.0, abs=2e-6)
    assert layer.y[0] == pytest.approx(6.0, abs=2e-6)
    np.testing.assert_array_equal(layer.values, [0.25, 0.5, 0.75])
    assert not layer.x.flags.writeable
    assert b"# source_crs: EPSG:32630\n" in spatial_layer_csv(layer)


def test_spatial_layer_can_keep_declared_projected_coordinates_without_reprojection():
    payload = b"x,y,value\n100,200,1\n200,200,2\n100,300,3\n"
    layer = load_xyz_layer(payload, source_crs="EPSG:32630", units="nT")

    np.testing.assert_array_equal(layer.x, [100.0, 200.0, 100.0])
    np.testing.assert_array_equal(layer.y, [200.0, 200.0, 300.0])
    assert layer.source_crs == layer.target_crs == "EPSG:32630"


@pytest.mark.parametrize(
    "x,y,source,target,message",
    [
        ([1.0, 2.0], [3.0], "EPSG:4326", "EPSG:32630", "aligned"),
        ([float("nan")], [0.0], "EPSG:4326", "EPSG:32630", "finite"),
        ([0.0], [0.0], "not-a-crs", "EPSG:32630", "Could not transform"),
    ],
)
def test_coordinate_transform_rejects_bad_inputs(x, y, source, target, message):
    with pytest.raises(ValueError, match=message):
        transform_coordinates(x, y, source, target)


def test_xyz_layer_rejects_malformed_rows_after_data_instead_of_dropping_points():
    payload = b"x,y,value\n0,0,1\nnot,a,point\n1,0,2\n"
    with pytest.raises(ValueError, match="row 3 contains non-numeric"):
        load_xyz_layer(payload, source_crs="EPSG:32630")


def test_xyz_layer_rejects_missing_crs_and_non_finite_values():
    with pytest.raises(ValueError, match="source CRS"):
        load_xyz_layer(b"0,0,1\n1,0,2\n0,1,3\n", source_crs="")
    with pytest.raises(ValueError, match="non-finite"):
        load_xyz_layer(b"0,0,1\n1,0,nan\n0,1,3\n", source_crs="EPSG:32630")
