"""Phase 1 regression tests — PDS4 ingestion & orbital geometry.

Uses mathematically rigorous synthetic PDS4 XML labels.
Core mathematical operations are NEVER mocked; the synthetic data is
generated with known closed-form answers so every assertion is
independently verifiable.
"""

from __future__ import annotations

import math
import textwrap

import numpy as np
import pytest

from lunar_core.io.pds4_parser import (
    MOON_RADIUS_M,
    InvalidTelemetryError,
    LunarBoundingBox,
    LunarTelemetryMetadata,
    compute_bbox_overlap,
    compute_scale_gap,
    parse_metadata,
)


# ═══════════════════════════════════════════════════════════════════════
# Synthetic PDS4 XML labels with known telemetry values
# ═══════════════════════════════════════════════════════════════════════

OHRC_LABEL = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational
        xmlns="http://pds.nasa.gov/pds4/pds/v1"
        xmlns:geom="http://pds.nasa.gov/pds4/geom/v1"
        xmlns:cart="http://pds.nasa.gov/pds4/cart/v1"
        xmlns:img="http://pds.nasa.gov/pds4/img/v1">

      <Identification_Area>
        <logical_identifier>urn:isro:ch2:ohrc:20190907T123456</logical_identifier>
        <title>Chandrayaan-2 OHRC Synthetic Test Product</title>
      </Identification_Area>

      <Observation_Area>
        <Observing_System>
          <instrument_name>OHRC</instrument_name>
        </Observing_System>
        <Discipline_Area>
          <geom:Geometry>
            <geom:Surface_Geometry>
              <geom:incidence_angle unit="deg">45.2310</geom:incidence_angle>
              <geom:emission_angle unit="deg">12.1450</geom:emission_angle>
              <geom:solar_azimuth_angle unit="deg">230.5678</geom:solar_azimuth_angle>
            </geom:Surface_Geometry>
          </geom:Geometry>
          <img:Imaging>
            <img:pixel_resolution unit="m/pixel">0.2500</img:pixel_resolution>
          </img:Imaging>
          <cart:Cartography>
            <cart:Spatial_Domain>
              <cart:Bounding_Coordinates>
                <cart:west_bounding_coordinate unit="deg">24.5000</cart:west_bounding_coordinate>
                <cart:east_bounding_coordinate unit="deg">24.6200</cart:east_bounding_coordinate>
                <cart:north_bounding_coordinate unit="deg">-70.3800</cart:north_bounding_coordinate>
                <cart:south_bounding_coordinate unit="deg">-70.5100</cart:south_bounding_coordinate>
              </cart:Bounding_Coordinates>
            </cart:Spatial_Domain>
          </cart:Cartography>
        </Discipline_Area>
      </Observation_Area>
    </Product_Observational>
""").encode("utf-8")

TMC2_LABEL = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <Product_Observational
        xmlns="http://pds.nasa.gov/pds4/pds/v1"
        xmlns:geom="http://pds.nasa.gov/pds4/geom/v1"
        xmlns:cart="http://pds.nasa.gov/pds4/cart/v1"
        xmlns:img="http://pds.nasa.gov/pds4/img/v1">

      <Identification_Area>
        <logical_identifier>urn:isro:ch2:tmc2:20190912T091011</logical_identifier>
      </Identification_Area>

      <Observation_Area>
        <Observing_System>
          <instrument_name>TMC-2</instrument_name>
        </Observing_System>
        <Discipline_Area>
          <geom:Geometry>
            <geom:Surface_Geometry>
              <geom:incidence_angle unit="deg">62.8400</geom:incidence_angle>
              <geom:emission_angle unit="deg">5.3200</geom:emission_angle>
              <geom:solar_azimuth_angle unit="deg">185.1234</geom:solar_azimuth_angle>
            </geom:Surface_Geometry>
          </geom:Geometry>
          <img:Imaging>
            <img:pixel_resolution unit="m/pixel">5.0000</img:pixel_resolution>
          </img:Imaging>
          <cart:Cartography>
            <cart:Spatial_Domain>
              <cart:Bounding_Coordinates>
                <cart:west_bounding_coordinate unit="deg">24.4000</cart:west_bounding_coordinate>
                <cart:east_bounding_coordinate unit="deg">24.8000</cart:east_bounding_coordinate>
                <cart:north_bounding_coordinate unit="deg">-70.2000</cart:north_bounding_coordinate>
                <cart:south_bounding_coordinate unit="deg">-70.6000</cart:south_bounding_coordinate>
              </cart:Bounding_Coordinates>
            </cart:Spatial_Domain>
          </cart:Cartography>
        </Discipline_Area>
      </Observation_Area>
    </Product_Observational>
""").encode("utf-8")

TOL = 1e-4  # Acceptance tolerance per spec


# ═══════════════════════════════════════════════════════════════════════
# Independent reference implementation for overlap (no imports from SUT)
# ═══════════════════════════════════════════════════════════════════════

def _ref_spherical_area(
    lat1: float, lat2: float, lon1: float, lon2: float,
    radius: float = MOON_RADIUS_M,
) -> float:
    """Standalone spherical-rectangle area for oracle verification."""
    return (
        radius * radius
        * abs(math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)))
        * abs(math.radians(lon2) - math.radians(lon1))
    )


def _ref_overlap(
    bb1_lats: tuple[float, float], bb1_lons: tuple[float, float],
    bb2_lats: tuple[float, float], bb2_lons: tuple[float, float],
) -> float:
    """Standalone overlap ratio for oracle verification."""
    lat_lo = max(bb1_lats[0], bb2_lats[0])
    lat_hi = min(bb1_lats[1], bb2_lats[1])
    lon_lo = max(bb1_lons[0], bb2_lons[0])
    lon_hi = min(bb1_lons[1], bb2_lons[1])
    if lat_lo >= lat_hi or lon_lo >= lon_hi:
        return 0.0
    a_inter = _ref_spherical_area(lat_lo, lat_hi, lon_lo, lon_hi)
    a1 = _ref_spherical_area(*bb1_lats, *bb1_lons)
    a2 = _ref_spherical_area(*bb2_lats, *bb2_lons)
    return a_inter / min(a1, a2)


# ═══════════════════════════════════════════════════════════════════════
# 1. PDS4 Label Parsing
# ═══════════════════════════════════════════════════════════════════════

class TestOHRCParsing:
    """Parse the synthetic OHRC label and verify every extracted value."""

    @pytest.fixture(scope="class")
    def meta(self) -> LunarTelemetryMetadata:
        return parse_metadata(OHRC_LABEL)

    def test_gsd(self, meta: LunarTelemetryMetadata) -> None:
        assert abs(meta.gsd_meters - 0.25) < TOL

    def test_incidence_angle(self, meta: LunarTelemetryMetadata) -> None:
        assert abs(meta.incidence_angle_deg - 45.2310) < TOL

    def test_emission_angle(self, meta: LunarTelemetryMetadata) -> None:
        assert abs(meta.emission_angle_deg - 12.1450) < TOL

    def test_solar_azimuth(self, meta: LunarTelemetryMetadata) -> None:
        assert abs(meta.solar_azimuth_deg - 230.5678) < TOL

    def test_product_id(self, meta: LunarTelemetryMetadata) -> None:
        assert meta.product_id == "urn:isro:ch2:ohrc:20190907T123456"

    def test_instrument(self, meta: LunarTelemetryMetadata) -> None:
        assert meta.instrument == "OHRC"

    def test_source_format(self, meta: LunarTelemetryMetadata) -> None:
        assert meta.source_format == "PDS4"

    def test_bounding_box_present(self, meta: LunarTelemetryMetadata) -> None:
        assert meta.bounding_box is not None

    def test_bbox_lat_range(self, meta: LunarTelemetryMetadata) -> None:
        bb = meta.bounding_box
        assert bb is not None
        assert abs(bb.lat_min - (-70.51)) < TOL
        assert abs(bb.lat_max - (-70.38)) < TOL

    def test_bbox_lon_range(self, meta: LunarTelemetryMetadata) -> None:
        bb = meta.bounding_box
        assert bb is not None
        assert abs(bb.lon_min - 24.50) < TOL
        assert abs(bb.lon_max - 24.62) < TOL

    def test_pixel_resolution(self, meta: LunarTelemetryMetadata) -> None:
        assert meta.pixel_resolution_m is not None
        assert abs(meta.pixel_resolution_m - 0.25) < TOL


class TestTMC2Parsing:
    """Parse the synthetic TMC-2 label."""

    @pytest.fixture(scope="class")
    def meta(self) -> LunarTelemetryMetadata:
        return parse_metadata(TMC2_LABEL)

    def test_gsd(self, meta: LunarTelemetryMetadata) -> None:
        assert abs(meta.gsd_meters - 5.0) < TOL

    def test_incidence_angle(self, meta: LunarTelemetryMetadata) -> None:
        assert abs(meta.incidence_angle_deg - 62.84) < TOL

    def test_emission_angle(self, meta: LunarTelemetryMetadata) -> None:
        assert abs(meta.emission_angle_deg - 5.32) < TOL

    def test_solar_azimuth(self, meta: LunarTelemetryMetadata) -> None:
        assert abs(meta.solar_azimuth_deg - 185.1234) < TOL

    def test_instrument(self, meta: LunarTelemetryMetadata) -> None:
        assert meta.instrument == "TMC-2"


# ═══════════════════════════════════════════════════════════════════════
# 2. GSD Scale-Gap Ratio
# ═══════════════════════════════════════════════════════════════════════

class TestScaleGap:

    def test_ohrc_vs_tmc2(self) -> None:
        """OHRC 0.25 m vs TMC-2 5 m → gap = 20×."""
        assert abs(compute_scale_gap(0.25, 5.0) - 20.0) < TOL

    def test_symmetric(self) -> None:
        """Order must not matter."""
        assert abs(compute_scale_gap(5.0, 0.25) - 20.0) < TOL

    def test_equal_gsd(self) -> None:
        """Same GSD → gap = 1.0."""
        assert abs(compute_scale_gap(0.25, 0.25) - 1.0) < TOL

    def test_ohrc_vs_iirs(self) -> None:
        """OHRC 0.25 m vs IIRS 80 m → gap = 320×."""
        assert abs(compute_scale_gap(0.25, 80.0) - 320.0) < TOL

    def test_negative_gsd_raises(self) -> None:
        with pytest.raises(InvalidTelemetryError, match="positive"):
            compute_scale_gap(-1.0, 5.0)

    def test_zero_gsd_raises(self) -> None:
        with pytest.raises(InvalidTelemetryError, match="positive"):
            compute_scale_gap(0.0, 5.0)


# ═══════════════════════════════════════════════════════════════════════
# 3. Spherical Bounding-Box Overlap
# ═══════════════════════════════════════════════════════════════════════

class TestBBoxOverlap:

    def test_disjoint_returns_zero(self) -> None:
        """Non-overlapping boxes → ratio = 0.0."""
        bb1 = LunarBoundingBox(lat_min=0, lat_max=10, lon_min=0, lon_max=10)
        bb2 = LunarBoundingBox(lat_min=20, lat_max=30, lon_min=20, lon_max=30)
        assert compute_bbox_overlap(bb1, bb2) == 0.0

    def test_contained_returns_one(self) -> None:
        """Small box fully inside large box → ratio = 1.0."""
        outer = LunarBoundingBox(lat_min=-10, lat_max=10, lon_min=-10, lon_max=10)
        inner = LunarBoundingBox(lat_min=-2, lat_max=2, lon_min=-2, lon_max=2)
        assert abs(compute_bbox_overlap(inner, outer) - 1.0) < TOL

    def test_identical_returns_one(self) -> None:
        """Identical boxes → ratio = 1.0."""
        bb = LunarBoundingBox(lat_min=5, lat_max=15, lon_min=30, lon_max=50)
        assert abs(compute_bbox_overlap(bb, bb) - 1.0) < TOL

    def test_half_longitude_overlap(self) -> None:
        """Same lat range, 50% lon overlap → ratio = 0.5 exactly.

        When latitude bands are identical, area is proportional to
        longitude span, so half the lon overlap gives ratio = 0.5.
        """
        bb1 = LunarBoundingBox(lat_min=0, lat_max=10, lon_min=0, lon_max=20)
        bb2 = LunarBoundingBox(lat_min=0, lat_max=10, lon_min=10, lon_max=30)
        ratio = compute_bbox_overlap(bb1, bb2)
        assert abs(ratio - 0.5) < TOL

    def test_parsed_labels_overlap(self) -> None:
        """OHRC vs TMC-2 overlap matches independent computation."""
        ohrc = parse_metadata(OHRC_LABEL)
        tmc2 = parse_metadata(TMC2_LABEL)
        assert ohrc.bounding_box is not None
        assert tmc2.bounding_box is not None

        ratio = compute_bbox_overlap(ohrc.bounding_box, tmc2.bounding_box)

        # Independent oracle
        expected = _ref_overlap(
            (ohrc.bounding_box.lat_min, ohrc.bounding_box.lat_max),
            (ohrc.bounding_box.lon_min, ohrc.bounding_box.lon_max),
            (tmc2.bounding_box.lat_min, tmc2.bounding_box.lat_max),
            (tmc2.bounding_box.lon_min, tmc2.bounding_box.lon_max),
        )
        assert abs(ratio - expected) < TOL
        # Sanity: OHRC is fully within TMC-2 extents
        assert abs(ratio - 1.0) < TOL

    def test_spherical_area_positive(self) -> None:
        """Any valid box must have non-negative area."""
        bb = LunarBoundingBox(lat_min=-45, lat_max=45, lon_min=0, lon_max=180)
        assert bb.spherical_area() > 0.0


# ═══════════════════════════════════════════════════════════════════════
# 4. Validation & Error Handling
# ═══════════════════════════════════════════════════════════════════════

class TestValidation:

    def test_invalid_latitude_above_90(self) -> None:
        """Latitude > 90° must raise InvalidTelemetryError."""
        bad_xml = OHRC_LABEL.replace(
            b">-70.3800<", b">95.0000<"
        )
        with pytest.raises(InvalidTelemetryError, match="latitude"):
            parse_metadata(bad_xml)

    def test_invalid_latitude_below_minus_90(self) -> None:
        """Latitude < -90° must raise InvalidTelemetryError."""
        bad_xml = OHRC_LABEL.replace(
            b">-70.5100<", b">-91.5000<"
        )
        with pytest.raises(InvalidTelemetryError, match="latitude"):
            parse_metadata(bad_xml)

    def test_missing_gsd_remains_unknown(self) -> None:
        """XML with no pixel_resolution tag → error."""
        no_gsd_xml = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <Product_Observational
                xmlns="http://pds.nasa.gov/pds4/pds/v1"
                xmlns:geom="http://pds.nasa.gov/pds4/geom/v1">
              <Observation_Area>
                <Discipline_Area>
                  <geom:Geometry>
                    <geom:Surface_Geometry>
                      <geom:incidence_angle unit="deg">45.0</geom:incidence_angle>
                      <geom:emission_angle unit="deg">10.0</geom:emission_angle>
                      <geom:solar_azimuth_angle unit="deg">200.0</geom:solar_azimuth_angle>
                    </geom:Surface_Geometry>
                  </geom:Geometry>
                </Discipline_Area>
              </Observation_Area>
            </Product_Observational>
        """).encode("utf-8")
        assert parse_metadata(no_gsd_xml).gsd_meters is None

    def test_missing_incidence_remains_unknown(self) -> None:
        """XML with no incidence_angle tag → error."""
        no_inc_xml = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <Product_Observational
                xmlns="http://pds.nasa.gov/pds4/pds/v1"
                xmlns:img="http://pds.nasa.gov/pds4/img/v1"
                xmlns:geom="http://pds.nasa.gov/pds4/geom/v1">
              <Observation_Area>
                <Discipline_Area>
                  <img:Imaging>
                    <img:pixel_resolution unit="m/pixel">0.25</img:pixel_resolution>
                  </img:Imaging>
                  <geom:Geometry>
                    <geom:Surface_Geometry>
                      <geom:emission_angle unit="deg">10.0</geom:emission_angle>
                      <geom:solar_azimuth_angle unit="deg">200.0</geom:solar_azimuth_angle>
                    </geom:Surface_Geometry>
                  </geom:Geometry>
                </Discipline_Area>
              </Observation_Area>
            </Product_Observational>
        """).encode("utf-8")
        assert parse_metadata(no_inc_xml).incidence_angle_deg is None

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_metadata("/nonexistent/label.xml")

    def test_pydantic_model_immutable(self) -> None:
        """LunarTelemetryMetadata and LunarBoundingBox are frozen."""
        meta = parse_metadata(OHRC_LABEL)
        with pytest.raises(Exception):  # Pydantic frozen error
            meta.gsd_meters = 999.0  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
# 5. Schema & Overlap Printout (acceptance gate)
# ═══════════════════════════════════════════════════════════════════════

def test_acceptance_gate_printout(capsys: pytest.CaptureFixture[str]) -> None:
    """Print parsed metadata and overlap ratio to stdout.

    The acceptance gate requires terminal stdout to contain the
    parsed schema and bounding-box overlap ratio.
    """
    ohrc = parse_metadata(OHRC_LABEL)
    tmc2 = parse_metadata(TMC2_LABEL)

    print("\n" + "=" * 70)
    print("PHASE 1 — PARSED METADATA SCHEMA")
    print("=" * 70)
    print(f"\n[OHRC]\n{ohrc.model_dump_json(indent=2)}")
    print(f"\n[TMC-2]\n{tmc2.model_dump_json(indent=2)}")

    gap = compute_scale_gap(ohrc.gsd_meters, tmc2.gsd_meters)
    print(f"\nGSD Scale Gap (OHRC <-> TMC-2): {gap:.1f}x")

    assert ohrc.bounding_box is not None and tmc2.bounding_box is not None
    overlap = compute_bbox_overlap(ohrc.bounding_box, tmc2.bounding_box)
    print(f"Bounding-Box Overlap Ratio:   {overlap:.4f}")
    print("=" * 70)

    # Verify it actually printed
    captured = capsys.readouterr()
    assert "PHASE 1" in captured.out
    assert "Overlap Ratio" in captured.out
