"""Pure-math tests for the physics-guided feature kernels.

These assert the closed-form correctness of f(RH), the PBL-normalized AOD, the
hygroscopic correction, the FNR ratio, Magnus RH-from-dewpoint, wind speed/dir,
and the aerosol-type flag — all NaN-safety guards included. No heavy or
credentialed dependencies are exercised; everything runs on NumPy alone.
"""

from __future__ import annotations

import numpy as np

from aqi_india.features import physics


def test_f_rh_unity_at_zero_humidity() -> None:
    # (1 - 0)^-g = 1 regardless of gamma.
    assert physics.f_rh(0.0) == 1.0
    assert physics.f_rh(0.0, gamma=1.0) == 1.0


def test_f_rh_monotonic_increasing_and_known_value() -> None:
    # f(RH) must grow with RH; check a hand-computed value at RH=50, g=0.6.
    expected = (1.0 - 0.5) ** (-0.6)  # = 2^0.6
    assert np.isclose(physics.f_rh(50.0, gamma=0.6), expected)
    grid = physics.f_rh(np.array([10.0, 40.0, 70.0, 90.0]), gamma=0.6)
    assert np.all(np.diff(grid) > 0.0)
    assert np.all(grid >= 1.0)


def test_f_rh_guards_saturation_and_out_of_range() -> None:
    # RH -> 100 is clamped to 99 (finite, large); RH>100 / negative -> NaN.
    near = physics.f_rh(100.0, gamma=0.6)
    assert np.isfinite(near) and near > 1.0
    assert np.isnan(physics.f_rh(120.0))
    assert np.isnan(physics.f_rh(-5.0))


def test_pbl_normalized_aod_basic_and_guard() -> None:
    assert np.isclose(physics.pbl_normalized_aod(0.8, 800.0), 0.001)
    out = physics.pbl_normalized_aod(
        np.array([0.5, 1.0, np.nan]), np.array([1000.0, 0.0, 500.0])
    )
    assert np.isclose(out[0], 0.0005)
    assert np.isnan(out[1])  # BLH=0 guarded
    assert np.isnan(out[2])  # NaN AOD propagates


def test_hygroscopic_correction_recovers_dry_aod() -> None:
    # AOD / f(RH); at RH=0 the correction is identity.
    assert np.isclose(physics.hygroscopic_correction(0.9, 0.0), 0.9)
    val = physics.hygroscopic_correction(1.0, 50.0, gamma=0.6)
    assert np.isclose(val, 1.0 / (2.0**0.6))
    assert val < 1.0  # high RH inflates ambient AOD, so dry < ambient


def test_fnr_ratio_and_masking() -> None:
    assert np.isclose(physics.fnr(2.0e-4, 1.0e-4), 2.0)
    out = physics.fnr(
        np.array([1.0e-4, 1.0e-4, -1.0]),
        np.array([1.0e-4, 0.0, 1.0e-4]),
    )
    assert np.isclose(out[0], 1.0)
    assert np.isnan(out[1])  # NO2 below min_no2 masked
    assert np.isnan(out[2])  # negative HCHO masked


def test_relative_humidity_from_dewpoint_bounds() -> None:
    # T == Td  =>  RH == 100 %.
    assert np.isclose(
        physics.relative_humidity_from_dewpoint(300.0, 300.0), 100.0
    )
    # Drier air (Td < T) => RH < 100.
    rh = physics.relative_humidity_from_dewpoint(300.0, 290.0)
    assert 0.0 < rh < 100.0
    # Celsius path agrees with Kelvin path.
    rh_c = physics.relative_humidity_from_dewpoint(27.0, 17.0, kelvin=False)
    assert np.isclose(rh, rh_c, atol=1e-6)


def test_wind_speed_and_direction_conventions() -> None:
    # Wind blowing toward +x (u=1, v=0) originates FROM the East (270 deg).
    assert np.isclose(physics.wind_speed(3.0, 4.0), 5.0)
    assert np.isclose(physics.wind_dir(1.0, 0.0), 270.0)
    # A southerly wind (v>0, blowing north) originates FROM the South (180 deg).
    assert np.isclose(physics.wind_dir(0.0, 1.0), 180.0)


def test_aerosol_type_flag_codes() -> None:
    c = physics.AEROSOL_TYPE_CODES
    # Absorbing + coarse -> dust.
    flag = physics.aerosol_type_flag(
        np.array([2.0]), np.array([0.3])
    )
    assert int(flag[0]) == c["dust"]
    # Absorbing + fine -> smoke.
    flag = physics.aerosol_type_flag(np.array([2.0]), np.array([1.5]))
    assert int(flag[0]) == c["smoke_absorbing"]
    # Clean + fine -> fine non-absorbing.
    flag = physics.aerosol_type_flag(np.array([0.1]), np.array([1.5]))
    assert int(flag[0]) == c["fine_nonabsorbing"]
    # Non-finite -> unknown.
    flag = physics.aerosol_type_flag(np.array([np.nan]), np.array([np.nan]))
    assert int(flag[0]) == c["unknown"]


def test_mass_extinction_efficiency_priors() -> None:
    assert physics.mass_extinction_efficiency("dust") < physics.mass_extinction_efficiency(
        "black_carbon"
    )
    codes = np.array(
        [physics.AEROSOL_TYPE_CODES["dust"], physics.AEROSOL_TYPE_CODES["unknown"]]
    )
    mee = physics.mee_for_codes(codes)
    assert mee[0] > 0.0
    assert np.isnan(mee[1])
