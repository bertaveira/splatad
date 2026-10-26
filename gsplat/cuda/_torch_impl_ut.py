"""Pure-PyTorch references for the unscented-transform projection and the eval3d
rasterizer, used by the tests to check the CUDA kernels.

The camera model, sigma-point construction and compositing follow upstream gsplat's
``Cameras.cuh`` (3DGUT, NVIDIA / Nerfstudio, Apache-2.0); the code here is a compact
re-implementation for single-batch inputs that favours readability over speed.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ._wrapper import RollingShutterType, UnscentedTransformParameters

# --------------------------------------------------------------------------------------
# Camera models
# --------------------------------------------------------------------------------------


FLT_EPS = 1.19e-7  # float32 epsilon, the kernels' degenerate-Gaussian guard


def _fisheye_max_angle(k: Tensor, width: int, height: int, fx: Tensor, fy: Tensor, cx: Tensor, cy: Tensor) -> Tensor:
    """Largest angle at which the Kannala-Brandt polynomial is still monotonic,
    clamped to the image diagonal (mirrors ``OpenCVFisheyeCameraModel``)."""
    # theta_d'(theta) = 1 + 3k1 t^2 + 5k2 t^4 + 7k3 t^6 + 9k4 t^8 ; find its first root.
    # Deliberately brute force (dense scan) rather than the kernel's closed-form cubic.
    thetas = torch.linspace(0.0, math.pi, 4096, device=k.device, dtype=k.dtype)
    t2 = thetas**2
    dpoly = 1 + 3 * k[0] * t2 + 5 * k[1] * t2**2 + 7 * k[2] * t2**3 + 9 * k[3] * t2**4
    bad = (dpoly <= 0).nonzero()
    max_angle = thetas[bad[0, 0]] if len(bad) else thetas.new_tensor(math.pi)
    # clamp to the field of view of the image corners
    corners = torch.tensor(
        [[0.0, 0.0], [width, 0.0], [0.0, height], [width, height]], device=k.device, dtype=k.dtype
    )
    m = (corners - torch.stack([cx, cy])) / torch.stack([fx, fy])
    max_dist = torch.linalg.norm(m, dim=-1).max()
    theta_corner = _fisheye_undistort_angle(max_dist[None], k, max_angle)[0]
    return torch.minimum(max_angle, theta_corner)


def _fisheye_distort_angle(theta: Tensor, k: Tensor) -> Tensor:
    t2 = theta**2
    return theta * (1 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4)


def _fisheye_undistort_angle(theta_d: Tensor, k: Tensor, max_angle: Tensor, iterations: int = 20) -> Tensor:
    theta = theta_d.clone()
    for _ in range(iterations):
        t2 = theta**2
        f = _fisheye_distort_angle(theta, k) - theta_d
        df = 1 + 3 * k[0] * t2 + 5 * k[1] * t2**2 + 7 * k[2] * t2**3 + 9 * k[3] * t2**4
        theta = theta - f / df
        theta = theta.clamp(min=0.0)
    return theta


def camera_project(
    points_cam: Tensor,  # [..., 3]
    K: Tensor,  # [3, 3]
    width: int,
    height: int,
    camera_model: str,
    radial_coeffs: Optional[Tensor],  # [4] fisheye
    margin_factor: float,
) -> Tuple[Tensor, Tensor]:
    """Camera-frame points -> pixels, with the validity flag of ``Cameras.cuh``."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y, z = points_cam[..., 0], points_cam[..., 1], points_cam[..., 2]
    if camera_model == "pinhole":
        valid = z > 0
        zs = torch.where(valid, z, torch.ones_like(z))
        uv = torch.stack([fx * x / zs + cx, fy * y / zs + cy], dim=-1)
    elif camera_model == "fisheye":
        k = radial_coeffs if radial_coeffs is not None else K.new_zeros(4)
        max_angle = _fisheye_max_angle(k, width, height, fx, fy, cx, cy)
        r = torch.sqrt(x**2 + y**2)
        theta_full = torch.atan2(r, z)
        theta = torch.minimum(theta_full, max_angle)
        safe_r = torch.where(r > 0, r, torch.ones_like(r))
        delta = torch.where(r > 0, _fisheye_distort_angle(theta, k) / safe_r, torch.ones_like(r))
        uv = torch.stack([fx * delta * x + cx, fy * delta * y + cy], dim=-1)
        valid = (z > 0) & (theta_full < max_angle)
    else:
        raise ValueError(camera_model)
    mx, my = margin_factor * width, margin_factor * height
    in_bounds = (uv[..., 0] >= -mx) & (uv[..., 0] < width + mx) & (uv[..., 1] >= -my) & (uv[..., 1] < height + my)
    return uv, valid & in_bounds


def camera_unproject(
    pixels: Tensor,  # [..., 2]
    K: Tensor,
    width: int,
    height: int,
    camera_model: str,
    radial_coeffs: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Pixels -> unit camera-frame ray directions and a validity flag."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    m = (pixels - torch.stack([cx, cy])) / torch.stack([fx, fy])
    if camera_model == "pinhole":
        d = torch.cat([m, torch.ones_like(m[..., :1])], dim=-1)
        return F.normalize(d, dim=-1), torch.ones_like(m[..., 0], dtype=torch.bool)
    k = radial_coeffs if radial_coeffs is not None else K.new_zeros(4)
    max_angle = _fisheye_max_angle(k, width, height, fx, fy, cx, cy)
    theta_d = torch.linalg.norm(m, dim=-1)
    theta = _fisheye_undistort_angle(theta_d, k, max_angle)
    safe = torch.where(theta_d > 0, theta_d, torch.ones_like(theta_d))
    scale = torch.where(theta_d > 0, torch.sin(theta) / safe, torch.ones_like(theta_d))
    d = torch.stack([m[..., 0] * scale, m[..., 1] * scale, torch.cos(theta)], dim=-1)
    return d, theta < max_angle


# --------------------------------------------------------------------------------------
# Poses and rolling shutter
# --------------------------------------------------------------------------------------


def _quat_from_rotmat(R: Tensor) -> Tensor:  # wxyz
    return _quat_normalize(_rotmat_to_quat(R))


def _rotmat_to_quat(R: Tensor) -> Tensor:
    # Shepperd's method, batched over leading dims
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    t = m00 + m11 + m22
    w = torch.sqrt(torch.clamp(1 + t, min=0)) / 2
    x = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0)) / 2
    y = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0)) / 2
    z = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0)) / 2
    x = torch.copysign(x, m21 - m12)
    y = torch.copysign(y, m02 - m20)
    z = torch.copysign(z, m10 - m01)
    return torch.stack([w, x, y, z], dim=-1)


def _quat_normalize(q: Tensor) -> Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


def _quat_to_rotmat(q: Tensor) -> Tensor:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=-2,
    )


def _quat_slerp(q0: Tensor, q1: Tensor, t: Tensor) -> Tensor:
    """glm::slerp semantics (shortest path), t broadcastable to [...]."""
    cos = (q0 * q1).sum(-1)
    q1 = torch.where((cos < 0)[..., None], -q1, q1)
    cos = cos.abs()
    # glm uses a linear interpolation when the quaternions are nearly parallel
    near = cos > 1 - 1e-6
    theta = torch.acos(cos.clamp(max=1.0))
    s0 = torch.where(near, 1 - t, torch.sin((1 - t) * theta) / torch.sin(theta).clamp(min=1e-12))
    s1 = torch.where(near, t, torch.sin(t * theta) / torch.sin(theta).clamp(min=1e-12))
    return s0[..., None] * q0 + s1[..., None] * q1


def interpolate_shutter_pose(
    t: Tensor, viewmat0: Tensor, viewmat1: Optional[Tensor]
) -> Tuple[Tensor, Tensor]:
    """World-to-camera (R, t) at relative exposure time ``t`` in [0, 1]."""
    R0, t0 = viewmat0[:3, :3], viewmat0[:3, 3]
    if viewmat1 is None:
        return R0.expand(*t.shape, 3, 3), t0.expand(*t.shape, 3)
    R1, t1 = viewmat1[:3, :3], viewmat1[:3, 3]
    q0, q1 = _quat_from_rotmat(R0), _quat_from_rotmat(R1)
    q = _quat_normalize(_quat_slerp(q0.expand(*t.shape, 4), q1.expand(*t.shape, 4), t))
    trans = (1 - t)[..., None] * t0 + t[..., None] * t1
    return _quat_to_rotmat(q), trans


def shutter_relative_time(uv: Tensor, width: int, height: int, rs: RollingShutterType) -> Tensor:
    u, v = uv[..., 0], uv[..., 1]
    if rs == RollingShutterType.GLOBAL:
        return torch.zeros_like(u)
    if rs == RollingShutterType.ROLLING_TOP_TO_BOTTOM:
        return torch.floor(v) / (height - 1)
    if rs == RollingShutterType.ROLLING_BOTTOM_TO_TOP:
        return (height - torch.ceil(v)) / (height - 1)
    if rs == RollingShutterType.ROLLING_LEFT_TO_RIGHT:
        return torch.floor(u) / (width - 1)
    if rs == RollingShutterType.ROLLING_RIGHT_TO_LEFT:
        return (width - torch.ceil(u)) / (width - 1)
    raise ValueError(rs)


def world_point_to_image_point_shutter(
    points: Tensor,  # [..., 3] world
    viewmat0: Tensor,
    viewmat1: Optional[Tensor],
    K: Tensor,
    width: int,
    height: int,
    camera_model: str,
    radial_coeffs: Optional[Tensor],
    rs: RollingShutterType,
    margin_factor: float,
    iterations: int = 10,
) -> Tuple[Tensor, Tensor]:
    """Mirror of ``world_point_to_image_point_shutter_pose``: fixed-point iteration
    of (pixel -> shutter time -> pose -> pixel)."""

    def project_at(R, t):
        pc = torch.einsum("...ij,...j->...i", R, points) + t
        return camera_project(pc, K, width, height, camera_model, radial_coeffs, margin_factor)

    R0, t0 = viewmat0[:3, :3], viewmat0[:3, 3]
    uv_start, valid_start = project_at(R0, t0)
    if rs == RollingShutterType.GLOBAL or viewmat1 is None:
        return uv_start, valid_start
    R1, t1 = viewmat1[:3, :3], viewmat1[:3, 3]
    uv_end, valid_end = project_at(R1, t1)
    uv = torch.where(valid_start[..., None], uv_start, uv_end)
    valid = valid_start | valid_end
    prev_time = torch.full_like(uv[..., 0], -1.0)
    for _ in range(iterations):
        time = shutter_relative_time(uv, width, height, rs)
        R, t = interpolate_shutter_pose(time, viewmat0, viewmat1)
        uv_new, valid_new = project_at(R, t)
        converged = time == prev_time
        uv = torch.where(converged[..., None], uv, uv_new)
        valid = torch.where(converged, valid, valid_new)
        prev_time = time
    return uv, valid


def pixel_to_world_ray_shutter(
    pixels: Tensor,  # [..., 2]
    viewmat0: Tensor,
    viewmat1: Optional[Tensor],
    K: Tensor,
    width: int,
    height: int,
    camera_model: str,
    radial_coeffs: Optional[Tensor],
    rs: RollingShutterType,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pixels -> world ray origins, directions and validity, with the pose at the
    pixel's shutter time."""
    d_cam, valid = camera_unproject(pixels, K, width, height, camera_model, radial_coeffs)
    time = shutter_relative_time(pixels, width, height, rs)
    R, t = interpolate_shutter_pose(time, viewmat0, viewmat1)
    Rt = R.transpose(-1, -2)
    origins = -torch.einsum("...ij,...j->...i", Rt, t)
    dirs = torch.einsum("...ij,...j->...i", Rt, d_cam)
    return origins, dirs, valid


# --------------------------------------------------------------------------------------
# Unscented-transform projection
# --------------------------------------------------------------------------------------


def _fully_fused_projection_with_ut(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Optional[Tensor],  # [N]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    calc_compensations: bool,
    camera_model: str = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    radial_coeffs: Optional[Tensor] = None,  # [C, 4]
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Reference for :func:`gsplat.cuda._wrapper.fully_fused_projection_with_ut`."""
    ut = ut_params or UnscentedTransformParameters()
    C, N = viewmats.shape[0], means.shape[0]
    D = 3
    lam = ut.alpha**2 * (D + ut.kappa) - D
    w_mean0 = lam / (D + lam)
    w_cov0 = w_mean0 + 1 - ut.alpha**2 + ut.beta
    w_i = 1 / (2 * (D + lam))

    quats_n = _quat_normalize(quats)
    R = _quat_to_rotmat(quats_n)  # [N, 3, 3]
    axes = R.transpose(-1, -2) * scales[..., None]  # axes[n, i] = s_i * R[:, i]; scaled by sqrt(D+lam) below
    sqrt_dl = math.sqrt(D + lam)
    # sigma points: mean, mean +/- sqrt(D+lam) * s_i * R[:, i]
    offsets = torch.cat([torch.zeros_like(means[:, None]), sqrt_dl * axes, -sqrt_dl * axes], dim=1)  # [N, 7, 3]
    sigma_points = means[:, None, :] + offsets  # [N, 7, 3]
    w_mean = means.new_tensor([w_mean0] + [w_i] * (2 * D))
    w_cov = means.new_tensor([w_cov0] + [w_i] * (2 * D))

    radii = torch.zeros(C, N, 2, dtype=torch.int32, device=means.device)
    means2d = torch.zeros(C, N, 2, device=means.device)
    depths = torch.zeros(C, N, device=means.device)
    conics = torch.zeros(C, N, 3, device=means.device)
    compensations = torch.zeros(C, N, device=means.device) if calc_compensations else None

    for c in range(C):
        vm0 = viewmats[c]
        vm1 = viewmats_rs[c] if viewmats_rs is not None else None
        rc = radial_coeffs[c] if radial_coeffs is not None else None
        # centre depth at the mid-exposure pose
        Rm, tm = interpolate_shutter_pose(means.new_tensor(0.5), vm0, vm1)
        mean_c = means @ Rm.T + tm
        depth = mean_c[:, 2]
        valid = (depth >= near_plane) & (depth <= far_plane)
        valid &= ~(torch.linalg.norm(quats, dim=-1) < FLT_EPS) & (scales.abs() >= FLT_EPS).all(-1)

        uv, uv_valid = world_point_to_image_point_shutter(
            sigma_points, vm0, vm1, Ks[c], width, height, camera_model, rc, rolling_shutter, ut.in_image_margin_factor
        )  # [N, 7, 2]
        if ut.require_all_sigma_points_valid:
            valid &= uv_valid.all(-1)
        else:
            valid &= uv_valid.any(-1)
        mean2d = (w_mean[None, :, None] * uv).sum(1)  # [N, 2]
        d = uv - mean2d[:, None]
        cov = (w_cov[None, :, None, None] * d[..., :, None] * d[..., None, :]).sum(1)  # [N, 2, 2]
        det_orig = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
        cov = cov + eps2d * torch.eye(2, device=means.device)
        det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0]
        comp = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
        valid &= det > 0
        valid &= (cov[:, 0, 0] >= 0) & (cov[:, 1, 1] >= 0)

        extend = torch.full_like(depth, 3.0)
        if opacities is not None:
            op = opacities * comp if calc_compensations else opacities
            valid &= op >= 1.0 / 255.0
            extend = torch.minimum(extend, torch.sqrt(torch.clamp(2 * torch.log(op * 255.0), min=0.0)))
        b = 0.5 * (cov[:, 0, 0] + cov[:, 1, 1])
        v1 = b + torch.sqrt(torch.clamp(b * b - det, min=0.01))
        r1 = extend * torch.sqrt(v1)
        rx = torch.ceil(torch.minimum(extend * torch.sqrt(cov[:, 0, 0].clamp(min=0)), r1))
        ry = torch.ceil(torch.minimum(extend * torch.sqrt(cov[:, 1, 1].clamp(min=0)), r1))
        valid &= ~((rx <= radius_clip) & (ry <= radius_clip))
        valid &= ~(
            (mean2d[:, 0] + rx <= 0) | (mean2d[:, 0] - rx >= width) | (mean2d[:, 1] + ry <= 0) | (mean2d[:, 1] - ry >= height)
        )
        inv = torch.linalg.inv(cov)
        radii[c, valid, 0] = rx[valid].int()
        radii[c, valid, 1] = ry[valid].int()
        means2d[c, valid] = mean2d[valid]
        depths[c, valid] = depth[valid]
        conics[c, valid] = torch.stack([inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]], -1)[valid]
        if compensations is not None:
            compensations[c, valid] = comp[valid]
    return radii, means2d, depths, conics, compensations


# --------------------------------------------------------------------------------------
# eval3d rasterization (dense, small images only)
# --------------------------------------------------------------------------------------


def _rasterize_to_pixels_eval3d(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    colors: Tensor,  # [C, N, D]
    opacities: Tensor,  # [C, N]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    tile_size: int,
    radii: Tensor,  # [C, N, 2] from the UT projection (tile binning)
    means2d: Tensor,  # [C, N, 2]
    depths: Tensor,  # [C, N]
    backgrounds: Optional[Tensor] = None,  # [C, D]
    camera_model: str = "pinhole",
    radial_coeffs: Optional[Tensor] = None,
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Dense reference for :func:`gsplat.cuda._wrapper.rasterize_to_pixels_eval3d`.

    Every pixel considers every Gaussian whose tile box (from the UT ``radii``)
    covers the pixel's tile, in increasing projected-depth order, and composites
    ``alpha = min(0.999, opac * exp(-0.5 |grd x gro|^2))`` front to back, skipping
    Gaussians below the 1/255 alpha threshold or with their closest approach behind
    the ray origin, and stopping once the transmittance drops to 1e-4. Autograd
    provides the gradients.
    """
    C, N, D = colors.shape
    device = means.device
    quats_n = _quat_normalize(quats)
    R = _quat_to_rotmat(quats_n)  # [N, 3, 3]
    Mt = torch.diag_embed(1.0 / scales) @ R.transpose(-1, -2)  # [N, 3, 3]

    vs, us = torch.meshgrid(
        torch.arange(height, device=device, dtype=means.dtype) + 0.5,
        torch.arange(width, device=device, dtype=means.dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([us, vs], -1).reshape(-1, 2)  # [P, 2]
    tile_x = (pixels[:, 0] // tile_size).long()
    tile_y = (pixels[:, 1] // tile_size).long()
    P = pixels.shape[0]

    renders = torch.zeros(C, P, D, device=device, dtype=colors.dtype)
    alphas_out = torch.zeros(C, P, 1, device=device, dtype=colors.dtype)
    for c in range(C):
        vm0 = viewmats[c]
        vm1 = viewmats_rs[c] if viewmats_rs is not None else None
        rc = radial_coeffs[c] if radial_coeffs is not None else None
        o, d, ray_valid = pixel_to_world_ray_shutter(
            pixels, vm0, vm1, Ks[c], width, height, camera_model, rc, rolling_shutter
        )  # [P, 3]
        valid_g = (radii[c] > 0).all(-1)  # [N]
        order = torch.argsort(depths[c] + torch.where(valid_g, 0.0, 1e30), stable=True)
        order = order[valid_g[order]]
        if len(order) == 0:
            continue
        # tile box coverage, as isect_tiles computes it
        m2d = means2d[c, order]
        rad = radii[c, order].to(means.dtype)
        tx_min = torch.floor((m2d[:, 0] - rad[:, 0]) / tile_size)
        tx_max = torch.ceil((m2d[:, 0] + rad[:, 0]) / tile_size)
        ty_min = torch.floor((m2d[:, 1] - rad[:, 1]) / tile_size)
        ty_max = torch.ceil((m2d[:, 1] + rad[:, 1]) / tile_size)
        covers = (
            (tile_x[:, None] >= tx_min[None]) & (tile_x[:, None] < tx_max[None])
            & (tile_y[:, None] >= ty_min[None]) & (tile_y[:, None] < ty_max[None])
        )  # [P, G]

        gro = torch.einsum("gij,pgj->pgi", Mt[order], o[:, None, :] - means[order][None])  # [P, G, 3]
        grd = F.normalize(torch.einsum("gij,pj->pgi", Mt[order], d), dim=-1)
        hit_t = -(grd * gro).sum(-1)
        gcrod = torch.cross(grd, gro, dim=-1)
        power = -0.5 * (gcrod * gcrod).sum(-1)
        alpha = torch.clamp(opacities[c, order][None] * torch.exp(power), max=0.999)
        active = covers & ray_valid[:, None] & (hit_t >= 0) & (alpha >= 1.0 / 255.0)
        alpha = torch.where(active, alpha, torch.zeros_like(alpha))
        # transmittance before each Gaussian; stop (exclusive) once it would drop to 1e-4
        one_minus = 1 - alpha
        T_before = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), one_minus[:, :-1]], 1), 1)
        T_after = T_before * one_minus
        stop = (T_after <= 1e-4) & active
        # everything from the first stop on contributes nothing
        stopped = torch.cumsum(stop.to(torch.int32), 1) > 0
        weight = torch.where(stopped, torch.zeros_like(alpha), alpha * T_before)
        T_final = torch.where(stopped.any(1), T_before[torch.arange(P), stop.to(torch.int32).argmax(1)], T_after[:, -1])
        renders[c] = torch.einsum("pg,gd->pd", weight, colors[c, order])
        if backgrounds is not None:
            renders[c] = renders[c] + T_final[:, None] * backgrounds[c][None]
        alphas_out[c, :, 0] = 1 - T_final
    return renders.reshape(C, height, width, D), alphas_out.reshape(C, height, width, 1)
