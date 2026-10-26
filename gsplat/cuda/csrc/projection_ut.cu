// Fused projection of 3D Gaussians through an arbitrary camera model with the
// unscented transform (3DGUT). Forward only: it produces the 2D footprints used
// for tile binning and depth sorting; gradients flow through the eval3d
// rasterizer (rasterization_eval3d.cu), which evaluates the Gaussians in 3D.
//
// Ported from nerfstudio-project/gsplat v1.5.3 ProjectionUT3DGSFused.cu
// (Apache-2.0, Copyright the Nerfstudio Team and contributors, 3DGUT by NVIDIA)
// without batch dimensions and the f-theta model.
#include "bindings.h"
#include "cameras.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void fully_fused_projection_ut_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const float *__restrict__ means,     // [N, 3]
    const float *__restrict__ quats,     // [N, 4]
    const float *__restrict__ scales,    // [N, 3]
    const float *__restrict__ opacities, // [N] optional
    const float *__restrict__ viewmats0, // [C, 4, 4]
    const float *__restrict__ viewmats1, // [C, 4, 4] optional, pose at the end of the exposure
    const float *__restrict__ Ks,        // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model_type,
    const UnscentedTransformParameters ut_params,
    const ShutterType rs_type,
    const float *__restrict__ radial_coeffs,     // [C, 6] (pinhole) or [C, 4] (fisheye) optional
    const float *__restrict__ tangential_coeffs, // [C, 2] optional
    const float *__restrict__ thin_prism_coeffs, // [C, 4] optional
    // outputs
    int32_t *__restrict__ radii,         // [C, N, 2]
    float *__restrict__ means2d,         // [C, N, 2]
    float *__restrict__ depths,          // [C, N]
    float *__restrict__ conics,          // [C, N, 3]
    float *__restrict__ compensations    // [C, N] optional
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    const glm::vec3 mean = glm::make_vec3(means + gid * 3);
    const glm::vec3 scale = glm::make_vec3(scales + gid * 3);
    glm::quat quat = glm::quat{
        quats[gid * 4 + 0], quats[gid * 4 + 1], quats[gid * 4 + 2], quats[gid * 4 + 3]
    }; // w, x, y, z

    // A zero-length quaternion has no orientation and a near-zero scale has a
    // divergent precision matrix (the eval3d rasterizer evaluates 1/scale), so
    // such Gaussians must not reach the intersection list.
    const float quat_norm2 = glm::dot(quat, quat);
    if (is_near_zero(quat_norm2) || is_near_zero(scale[0]) || is_near_zero(scale[1]) ||
        is_near_zero(scale[2])) {
        return;
    }

    auto rs_params = RollingShutterParameters(
        viewmats0 + cid * 16, viewmats1 == nullptr ? nullptr : viewmats1 + cid * 16
    );

    // Depth and near/far culling use the mid-exposure pose as the single
    // per-Gaussian camera pose.
    const auto shutter_pose = interpolate_shutter_pose(0.5f, rs_params);
    const glm::vec3 mean_c = glm::rotate(shutter_pose.q, mean) + shutter_pose.t;
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        return;
    }

    quat *= rsqrtf(quat_norm2);

    ImageGaussianReturn image_gaussian_return;
    const bool known_model = with_camera_model(
        camera_model_type,
        cid,
        Ks,
        image_width,
        image_height,
        rs_type,
        radial_coeffs,
        tangential_coeffs,
        thin_prism_coeffs,
        [&](auto const &camera_model) {
            image_gaussian_return =
                world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
                    camera_model, rs_params, ut_params, mean, scale, quat
                );
        }
    );
    if (!known_model) {
        return;
    }

    auto [mean2d, covar2d, valid_ut] = image_gaussian_return;
    if (!valid_ut) {
        return;
    }

    float compensation;
    const float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        return;
    }

    // The UT covariance estimate is not guaranteed positive semi-definite (the
    // centre weight is very negative for alpha = 0.1); a negative diagonal would
    // give NaN extents below, so treat it as a failed projection.
    if (covar2d[0][0] < 0.f || covar2d[1][1] < 0.f) {
        return;
    }

    glm::mat2 covar2d_inv = glm::inverse(covar2d);

    // Opacity-aware extent, matching the EWA projection: 3 sigma at most, and
    // less when the Gaussian's peak alpha is already close to the threshold.
    float extend = 3.f;
    if (opacities != nullptr) {
        float opacity = opacities[gid];
        if (calc_compensations) {
            opacity *= compensation;
        }
        if (opacity < ALPHA_THRESHOLD) {
            return;
        }
        extend = min(extend, sqrtf(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // Tight rectangular bounding box (https://arxiv.org/pdf/2402.00525).
    const float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    const float tmp = sqrtf(max(0.01f, b * b - det));
    const float v1 = b + tmp; // larger eigenvalue
    const float r1 = extend * sqrtf(v1);
    const float radius_x = ceilf(min(extend * sqrtf(covar2d[0][0]), r1));
    const float radius_y = ceilf(min(extend * sqrtf(covar2d[1][1]), r1));

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        return;
    }

    // mask out gaussians outside the image region
    if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
        mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
        return;
    }

    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fully_fused_projection_ut_fwd_tensor(
    const torch::Tensor &means,                   // [N, 3]
    const torch::Tensor &quats,                   // [N, 4]
    const torch::Tensor &scales,                  // [N, 3]
    const at::optional<torch::Tensor> &opacities, // [N] optional
    const torch::Tensor &viewmats0,               // [C, 4, 4]
    const at::optional<torch::Tensor> &viewmats1, // [C, 4, 4] optional
    const torch::Tensor &Ks,                      // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model,
    const UnscentedTransformParameters ut_params,
    const ShutterType rs_type,
    const at::optional<torch::Tensor> &radial_coeffs,     // [C, 6] or [C, 4] optional
    const at::optional<torch::Tensor> &tangential_coeffs, // [C, 2] optional
    const at::optional<torch::Tensor> &thin_prism_coeffs  // [C, 4] optional
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats0);
    CHECK_INPUT(Ks);
    if (opacities.has_value()) {
        CHECK_INPUT(opacities.value());
    }
    if (viewmats1.has_value()) {
        CHECK_INPUT(viewmats1.value());
    }
    if (radial_coeffs.has_value()) {
        CHECK_INPUT(radial_coeffs.value());
    }
    if (tangential_coeffs.has_value()) {
        CHECK_INPUT(tangential_coeffs.value());
    }
    if (thin_prism_coeffs.has_value()) {
        CHECK_INPUT(thin_prism_coeffs.value());
    }
    TORCH_CHECK(
        camera_model == CameraModelType::PINHOLE || camera_model == CameraModelType::FISHEYE,
        "only the pinhole and fisheye camera models are implemented"
    );
    if (camera_model == CameraModelType::FISHEYE && radial_coeffs.has_value()) {
        TORCH_CHECK(radial_coeffs.value().size(-1) == 4, "fisheye radial_coeffs must be [C, 4]");
    }
    if (camera_model == CameraModelType::PINHOLE && radial_coeffs.has_value()) {
        TORCH_CHECK(radial_coeffs.value().size(-1) == 6, "pinhole radial_coeffs must be [C, 6]");
    }
    if (rs_type != ShutterType::GLOBAL) {
        TORCH_CHECK(viewmats1.has_value(), "rolling shutter requires viewmats1");
    }

    uint32_t N = means.size(0);     // number of gaussians
    uint32_t C = viewmats0.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Zero-initialised: the kernel returns early for culled Gaussians and leaves radii = 0.
    torch::Tensor radii = torch::zeros({C, N, 2}, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor depths = torch::empty({C, N}, means.options());
    torch::Tensor conics = torch::empty({C, N, 3}, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({C, N}, means.options());
    }
    if (C && N) {
        fully_fused_projection_ut_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS,
                                               N_THREADS, 0, stream>>>(
            C,
            N,
            means.data_ptr<float>(),
            quats.data_ptr<float>(),
            scales.data_ptr<float>(),
            opacities.has_value() ? opacities.value().data_ptr<float>() : nullptr,
            viewmats0.data_ptr<float>(),
            viewmats1.has_value() ? viewmats1.value().data_ptr<float>() : nullptr,
            Ks.data_ptr<float>(),
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model,
            ut_params,
            rs_type,
            radial_coeffs.has_value() ? radial_coeffs.value().data_ptr<float>() : nullptr,
            tangential_coeffs.has_value() ? tangential_coeffs.value().data_ptr<float>()
                                          : nullptr,
            thin_prism_coeffs.has_value() ? thin_prism_coeffs.value().data_ptr<float>()
                                          : nullptr,
            radii.data_ptr<int32_t>(),
            means2d.data_ptr<float>(),
            depths.data_ptr<float>(),
            conics.data_ptr<float>(),
            calc_compensations ? compensations.data_ptr<float>() : nullptr
        );
    }
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}
