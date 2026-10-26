"""Tests for the fisheye / unscented-transform / eval3d path against pure-torch references."""

import math

import pytest
import torch

from gsplat._helper import load_test_data
from gsplat.cuda._wrapper import (
    RollingShutterType,
    fully_fused_projection,
    fully_fused_projection_with_ut,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels_eval3d,
)

device = torch.device("cuda:0")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")


# Fractions of elements allowed to miss the tolerance. They absorb Gaussians / pixels
# that sit exactly on a validity or clamp boundary of the camera model, where the CUDA
# kernel and the torch reference legitimately land on different sides.
OUTLIERS_GLOBAL_SHUTTER = 1e-4  # a handful of boundary Gaussians
OUTLIERS_ROLLING_SHUTTER = 2e-3  # the per-Gaussian row iteration converges to a different row in float32
OUTLIERS_EVAL3D = 5e-4  # pixels at the alpha / transmittance cut-offs


def assert_mostly_close(actual, expected, rtol, atol, max_outliers=OUTLIERS_GLOBAL_SHUTTER):
    """``assert_close`` that tolerates at most ``max_outliers`` (fraction) of the elements
    outside the tolerance."""
    err = (actual - expected).abs()
    outliers = float((err > atol + rtol * expected.abs()).float().mean())
    assert outliers <= max_outliers, (
        f"{outliers:.2e} of the elements exceed the tolerance (allowed {max_outliers:.0e}); "
        f"worst abs error {float(err.max()):.3e}"
    )


FISHEYE_K = torch.tensor([0.03, -0.012, 0.004, -0.0008])  # Kannala-Brandt k1..k4


@pytest.fixture
def scene():
    torch.manual_seed(0)
    (means, quats, scales, opacities, colors, viewmats, Ks, width, height) = load_test_data(
        device=device
    )
    C = viewmats.shape[0]
    # A small image and a subset of the Gaussians keep the dense torch eval3d
    # reference (every pixel against every binned Gaussian) affordable.
    keep = torch.randperm(means.shape[0], device=device)[:4000]
    means, quats, scales, opacities, colors = (
        means[keep], quats[keep], scales[keep], opacities[keep], colors[keep]
    )
    scale_img = 0.125
    width, height = int(width * scale_img), int(height * scale_img)
    Ks = Ks.clone()
    Ks[:, :2] *= scale_img
    return dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        C=C,
    )


def _fisheye_Ks(Ks, width, height):
    # Widen the field of view so the distortion matters: f ~ 0.4 * width.
    Ks = Ks.clone()
    Ks[:, 0, 0] = 0.4 * width
    Ks[:, 1, 1] = 0.4 * width
    Ks[:, 0, 2] = width / 2
    Ks[:, 1, 2] = height / 2
    return Ks


def _rolling_shutter_viewmats(viewmats):
    """End-of-exposure poses: a small translation and yaw away from the start."""
    C = viewmats.shape[0]
    delta = torch.eye(4, device=device).repeat(C, 1, 1)
    yaw = 0.01
    delta[:, 0, 0] = math.cos(yaw)
    delta[:, 0, 2] = math.sin(yaw)
    delta[:, 2, 0] = -math.sin(yaw)
    delta[:, 2, 2] = math.cos(yaw)
    delta[:, 0, 3] = 0.05
    delta[:, 2, 3] = 0.1
    return delta @ viewmats


# ------------------------------------------------------------------------------------
# camera model
# ------------------------------------------------------------------------------------


@cuda_only
def test_fisheye_camera_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from gsplat.cuda._torch_impl_ut import camera_project, camera_unproject

    torch.manual_seed(0)
    width, height = 640, 480
    K = torch.tensor([[260.0, 0.0, 322.0], [0.0, 258.0, 238.0], [0.0, 0.0, 1.0]], device=device)
    k = FISHEYE_K.to(device)
    pts = torch.randn(2000, 3, device=device)
    pts[:, 2] = pts[:, 2].abs() + 0.5
    uv, valid = camera_project(pts, K, width, height, "fisheye", k, margin_factor=0.0)
    ref, _ = cv2.fisheye.projectPoints(
        pts.cpu().numpy().reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        K.cpu().numpy().astype(np.float64),
        k.cpu().numpy().astype(np.float64),
    )
    ref = torch.as_tensor(ref.reshape(-1, 2), device=device, dtype=torch.float32)
    torch.testing.assert_close(uv[valid], ref[valid], rtol=1e-4, atol=5e-3)

    dirs, dvalid = camera_unproject(uv[valid], K, width, height, "fisheye", k)
    assert dvalid.all()
    torch.testing.assert_close(dirs, torch.nn.functional.normalize(pts[valid], dim=-1), rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------------------------------
# unscented-transform projection
# ------------------------------------------------------------------------------------


@cuda_only
@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
@pytest.mark.parametrize(
    "rolling_shutter",
    [RollingShutterType.GLOBAL, RollingShutterType.ROLLING_TOP_TO_BOTTOM],
)
@pytest.mark.parametrize("calc_compensations", [False, True])
def test_fully_fused_projection_ut(scene, camera_model, rolling_shutter, calc_compensations):
    from gsplat.cuda._torch_impl_ut import _fully_fused_projection_with_ut

    C, width, height = scene["C"], scene["width"], scene["height"]
    Ks = scene["Ks"] if camera_model == "pinhole" else _fisheye_Ks(scene["Ks"], width, height)
    radial = None if camera_model == "pinhole" else FISHEYE_K.to(device).repeat(C, 1)
    viewmats_rs = (
        None if rolling_shutter == RollingShutterType.GLOBAL else _rolling_shutter_viewmats(scene["viewmats"])
    )
    kwargs = dict(
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        calc_compensations=calc_compensations,
        camera_model=camera_model,
        radial_coeffs=radial,
        rolling_shutter=rolling_shutter,
        viewmats_rs=viewmats_rs,
    )
    out = fully_fused_projection_with_ut(
        scene["means"], scene["quats"], scene["scales"], scene["opacities"], scene["viewmats"], Ks, width, height, **kwargs
    )
    ref = _fully_fused_projection_with_ut(
        scene["means"], scene["quats"], scene["scales"], scene["opacities"], scene["viewmats"], Ks, width, height, **kwargs
    )
    radii, means2d, depths, conics, compensations = out
    _radii, _means2d, _depths, _conics, _compensations = ref
    valid = (radii > 0).all(-1)
    _valid = (_radii > 0).all(-1)
    # validity may legitimately differ on a few borderline Gaussians
    agree = (valid == _valid).float().mean()
    assert agree > 0.995, agree
    both = valid & _valid
    assert both.float().mean() > 0.05, "too few Gaussians visible to both for a meaningful comparison"
    max_outliers = (
        OUTLIERS_GLOBAL_SHUTTER
        if rolling_shutter == RollingShutterType.GLOBAL
        else OUTLIERS_ROLLING_SHUTTER
    )
    assert_mostly_close(radii[both].float(), _radii[both].float(), rtol=0, atol=2, max_outliers=max_outliers)
    assert_mostly_close(means2d[both], _means2d[both], rtol=2e-3, atol=5e-2, max_outliers=max_outliers)
    torch.testing.assert_close(depths[both], _depths[both], rtol=1e-5, atol=1e-5)
    assert_mostly_close(conics[both], _conics[both], rtol=2e-3, atol=2e-3, max_outliers=max_outliers)
    if calc_compensations:
        assert_mostly_close(
            compensations[both], _compensations[both], rtol=1e-3, atol=1e-3, max_outliers=max_outliers
        )
    else:
        assert compensations is None


@cuda_only
def test_ut_pinhole_matches_ewa_projection(scene):
    """With a perfect pinhole the UT footprint should be close to the EWA one."""
    C, width, height = scene["C"], scene["width"], scene["height"]
    zeros3 = torch.zeros(C, 3, device=device)
    radii_e, means2d_e, depths_e, conics_e, _, _ = fully_fused_projection(
        scene["means"],
        None,
        scene["quats"],
        scene["scales"],
        None,
        scene["viewmats"],
        scene["Ks"],
        width,
        height,
        zeros3,
        zeros3,
        torch.zeros(C, device=device),
        opacities=scene["opacities"],
    )
    radii_u, means2d_u, depths_u, conics_u, _ = fully_fused_projection_with_ut(
        scene["means"], scene["quats"], scene["scales"], scene["opacities"], scene["viewmats"], scene["Ks"], width, height
    )
    both = (radii_e > 0).all(-1) & (radii_u > 0).all(-1)
    assert both.float().mean() > 0.1
    torch.testing.assert_close(means2d_u[both], means2d_e[both], rtol=1e-2, atol=0.5)
    torch.testing.assert_close(depths_u[both], depths_e[both], rtol=1e-5, atol=1e-5)
    # conics agree to a few percent for small Gaussians (the UT is exact for linear maps)
    rel = ((conics_u[both] - conics_e[both]).abs() / conics_e[both].abs().clamp(min=1e-2)).median()
    assert rel < 0.05, rel


# ------------------------------------------------------------------------------------
# eval3d rasterizer
# ------------------------------------------------------------------------------------


def _project_and_bin(scene, Ks, camera_model, radial, rolling_shutter, viewmats_rs, tile_size):
    C, width, height = scene["C"], scene["width"], scene["height"]
    radii, means2d, depths, conics, _ = fully_fused_projection_with_ut(
        scene["means"],
        scene["quats"],
        scene["scales"],
        scene["opacities"],
        scene["viewmats"],
        Ks,
        width,
        height,
        camera_model=camera_model,
        radial_coeffs=radial,
        rolling_shutter=rolling_shutter,
        viewmats_rs=viewmats_rs,
    )
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    return radii, means2d, depths, isect_offsets, flatten_ids


@cuda_only
@pytest.mark.parametrize("camera_model", ["pinhole", "fisheye"])
@pytest.mark.parametrize(
    "rolling_shutter",
    [RollingShutterType.GLOBAL, RollingShutterType.ROLLING_TOP_TO_BOTTOM],
)
@pytest.mark.parametrize("channels", [3, 33])
def test_rasterize_to_pixels_eval3d(scene, camera_model, rolling_shutter, channels):
    from gsplat.cuda._torch_impl_ut import _rasterize_to_pixels_eval3d

    C, N, width, height = scene["C"], scene["means"].shape[0], scene["width"], scene["height"]
    Ks = scene["Ks"] if camera_model == "pinhole" else _fisheye_Ks(scene["Ks"], width, height)
    radial = None if camera_model == "pinhole" else FISHEYE_K.to(device).repeat(C, 1)
    viewmats_rs = (
        None if rolling_shutter == RollingShutterType.GLOBAL else _rolling_shutter_viewmats(scene["viewmats"])
    )
    tile_size = 16
    radii, means2d, depths, isect_offsets, flatten_ids = _project_and_bin(
        scene, Ks, camera_model, radial, rolling_shutter, viewmats_rs, tile_size
    )
    colors = torch.randn(C, N, channels, device=device)
    backgrounds = torch.rand(C, channels, device=device)
    opacities = scene["opacities"].repeat(C, 1)

    means = scene["means"].clone().requires_grad_(True)
    quats = scene["quats"].clone().requires_grad_(True)
    scales = scene["scales"].clone().requires_grad_(True)
    colors.requires_grad_(True)
    opacities.requires_grad_(True)
    backgrounds.requires_grad_(True)

    render, alphas = rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities,
        scene["viewmats"],
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        camera_model=camera_model,
        radial_coeffs=radial,
        rolling_shutter=rolling_shutter,
        viewmats_rs=viewmats_rs,
    )
    _render, _alphas = _rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities,
        scene["viewmats"],
        Ks,
        width,
        height,
        tile_size,
        radii,
        means2d,
        depths,
        backgrounds=backgrounds,
        camera_model=camera_model,
        radial_coeffs=radial,
        rolling_shutter=rolling_shutter,
        viewmats_rs=viewmats_rs,
    )
    assert torch.isfinite(render).all() and torch.isfinite(alphas).all()
    assert float(alphas.mean()) > 0.02
    # A few pixels sit on the transmittance / alpha thresholds where the two
    # implementations stop at different Gaussians.
    assert_mostly_close(alphas, _alphas, rtol=1e-3, atol=2e-3, max_outliers=OUTLIERS_EVAL3D)
    assert_mostly_close(render, _render, rtol=1e-3, atol=2e-3, max_outliers=OUTLIERS_EVAL3D)

    v_render = torch.randn_like(render)
    v_alphas = torch.randn_like(alphas)
    grads = torch.autograd.grad(
        (render * v_render).sum() + (alphas * v_alphas).sum(),
        (means, quats, scales, colors, opacities, backgrounds),
    )
    _grads = torch.autograd.grad(
        (_render * v_render).sum() + (_alphas * v_alphas).sum(),
        (means, quats, scales, colors, opacities, backgrounds),
    )
    names = ["means", "quats", "scales", "colors", "opacities", "backgrounds"]
    for name, g, _g in zip(names, grads, _grads):
        assert torch.isfinite(g).all(), name
        # Compare on the bulk of the distribution: a handful of Gaussians sit on
        # alpha / transmittance thresholds where the two implementations differ.
        err = (g - _g).abs().flatten()
        scale = _g.abs().flatten().max().clamp(min=1e-6)
        assert float(torch.quantile(err / scale, 0.999)) < 2e-2, (name, float((err / scale).max()))


@cuda_only
def test_eval3d_pinhole_matches_classic_rasterization():
    """UT + eval3d with a perfect pinhole should render (almost) the same image as EWA.

    The two agree for Gaussians that span several pixels; for sub-pixel Gaussians
    the EWA path's eps2d low-pass blur has no counterpart in the exact 3D
    evaluation, so the full scene is rendered at half resolution.
    """
    from gsplat.rendering import rasterization

    torch.manual_seed(0)
    means, quats, scales, opacities, _, viewmats, Ks, width, height = load_test_data(device=device)
    width, height = width // 2, height // 2
    Ks = Ks.clone()
    Ks[:, :2] *= 0.5
    N = means.shape[0]
    colors = torch.rand(N, 3, device=device)
    common = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        velocities=None,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=False,
        render_mode="RGB+ED",
    )
    classic, alpha_c, _ = rasterization(**common)
    ut, alpha_u, meta = rasterization(**common, with_ut=True, with_eval3d=True)
    assert meta["with_eval3d"] and meta["render_medians"] is None
    mse = float(((classic[..., :3] - ut[..., :3]) ** 2).mean())
    psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
    assert psnr > 28.0, psnr
    # alpha differs where the EWA blur (eps2d) widens small Gaussians that eval3d sees sharp
    assert float((alpha_u - alpha_c).abs().mean()) < 0.02
    valid = (alpha_c > 0.5) & (alpha_u > 0.5)
    depth_rel = ((classic[..., 3:] - ut[..., 3:]).abs() / classic[..., 3:].abs().clamp(min=1e-3))[valid]
    assert float(depth_rel.median()) < 0.05


@cuda_only
def test_rasterization_fisheye_smoke(scene):
    from gsplat.rendering import rasterization

    C, N, width, height = scene["C"], scene["means"].shape[0], scene["width"], scene["height"]
    Ks = _fisheye_Ks(scene["Ks"], width, height)
    radial = FISHEYE_K.to(device).repeat(C, 1)
    features = torch.rand(N, 128, device=device, requires_grad=True)
    means = scene["means"].clone().requires_grad_(True)
    render, alphas, meta = rasterization(
        means,
        scene["quats"],
        scene["scales"],
        scene["opacities"],
        features,
        None,
        scene["viewmats"],
        Ks,
        width,
        height,
        packed=False,
        render_mode="RGB+ED",
        channel_chunk=128,
        camera_model="fisheye",
        radial_coeffs=radial,
        with_ut=True,
        with_eval3d=True,
    )
    assert render.shape == (C, height, width, 129)
    assert alphas.shape == (C, height, width, 1)
    assert torch.isfinite(render).all()
    assert meta["camera_model"] == "fisheye"
    # 128 feature channels do not fit the eval3d backward's shared memory at tile 16
    assert meta["tile_size"] == 8
    render.sum().backward()
    assert means.grad is not None and torch.isfinite(means.grad).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()

    with pytest.raises(AssertionError):
        rasterization(
            means, scene["quats"], scene["scales"], scene["opacities"], features, None,
            scene["viewmats"], Ks, width, height, packed=False, camera_model="fisheye",
            radial_coeffs=radial, with_ut=False,
        )
    with pytest.raises(AssertionError):
        rasterization(
            means, scene["quats"], scene["scales"], scene["opacities"], features, None,
            scene["viewmats"], Ks, width, height, packed=False, camera_model="fisheye",
            radial_coeffs=radial, with_ut=True, with_eval3d=True, rasterize_mode="antialiased",
        )
