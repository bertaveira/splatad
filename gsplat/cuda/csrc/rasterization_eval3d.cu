// "eval3d" rasterization (3DGUT): every pixel casts a ray through the camera
// model (pinhole or fisheye, optionally with a rolling shutter) and the
// Gaussians found by the unscented-transform tile binning are evaluated in 3D
// along that ray. Alpha is opac * exp(-0.5 |grd x gro|^2) with gro / grd the
// ray origin and direction in the Gaussian's unit frame. The backward is
// analytic and produces gradients for means, quats, scales, colors and
// opacities; there is no means2d gradient.
//
// Ported from nerfstudio-project/gsplat v1.5.3
// RasterizeToPixelsFromWorld3DGS{Fwd,Bwd}.cu (Apache-2.0, Copyright the
// Nerfstudio Team and contributors, 3DGUT by NVIDIA) without batch dimensions
// and the f-theta model, including the upstream fixes that followed v1.5.3
// (skip Gaussians whose closest approach is behind the camera; identical
// forward/backward expression forms).
#include "bindings.h"
#include "cameras.cuh"
#include "helpers.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace {

// World-space ray through the centre of pixel (j, i) of camera `cid`.
inline __device__ WorldRay eval3d_pixel_ray(
    const float px,
    const float py,
    const uint32_t cid,
    const uint32_t image_width,
    const uint32_t image_height,
    const float *__restrict__ viewmats0,
    const float *__restrict__ viewmats1,
    const float *__restrict__ Ks,
    const CameraModelType camera_model_type,
    const ShutterType rs_type,
    const float *__restrict__ radial_coeffs,
    const float *__restrict__ tangential_coeffs,
    const float *__restrict__ thin_prism_coeffs
) {
    auto rs_params = RollingShutterParameters(
        viewmats0 + cid * 16, viewmats1 == nullptr ? nullptr : viewmats1 + cid * 16
    );
    WorldRay ray{glm::vec3(0.f), glm::vec3(0.f, 0.f, 1.f), false};
    with_camera_model(
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
            ray = camera_model.image_point_to_world_ray_shutter_pose(
                glm::vec2(px, py), rs_params
            );
        }
    );
    return ray;
}

// S^-1 * R^T: maps world offsets into the Gaussian's unit frame. The same
// expression form is used in forward and backward on purpose.
inline __device__ glm::mat3 eval3d_inverse_scale_rotation(
    const glm::mat3 &R, const glm::vec3 &scale
) {
    const glm::mat3 S = glm::mat3(
        1.0f / scale[0], 0.f, 0.f, 0.f, 1.0f / scale[1], 0.f, 0.f, 0.f, 1.0f / scale[2]
    );
    return S * glm::transpose(R);
}

// Footprint filter, the 3D counterpart of eps2d: widens each Gaussian by
// footprint_variance * (pixel angle * t)^2 per axis and scales its response by
// the determinant ratio of its covariance projected perpendicular to the ray.

constexpr float FOOTPRINT_OFFSET_PX = 5.f;

// Angle a pixel subtends, measured between rays on both image diagonals; falls
// back to 1 / focal length where a ray is invalid.
inline __device__ float eval3d_pixel_footprint(
    const float px,
    const float py,
    const uint32_t cid,
    const uint32_t image_width,
    const uint32_t image_height,
    const float *__restrict__ Ks,
    const CameraModelType camera_model_type,
    const ShutterType rs_type,
    const float *__restrict__ radial_coeffs,
    const float *__restrict__ tangential_coeffs,
    const float *__restrict__ thin_prism_coeffs
) {
    const float h = FOOTPRINT_OFFSET_PX;
    float footprint = rsqrtf(Ks[cid * 9 + 0] * Ks[cid * 9 + 4]);
    with_camera_model(
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
            const float x_max = (float)image_width - 0.5f, y_max = (float)image_height - 0.5f;
            auto inside = [&](float x, float y) {
                return glm::vec2(fminf(fmaxf(x, 0.5f), x_max), fminf(fmaxf(y, 0.5f), y_max));
            };
            const glm::vec2 pa0 = inside(px + h, py + h);
            const glm::vec2 pa1 = inside(px - h, py - h);
            const glm::vec2 pb0 = inside(px + h, py - h);
            const glm::vec2 pb1 = inside(px - h, py + h);
            const CameraRay a0 = camera_model.image_point_to_camera_ray(pa0);
            const CameraRay a1 = camera_model.image_point_to_camera_ray(pa1);
            const CameraRay b0 = camera_model.image_point_to_camera_ray(pb0);
            const CameraRay b1 = camera_model.image_point_to_camera_ray(pb1);
            const float da = glm::length(pa0 - pa1);
            const float db = glm::length(pb0 - pb1);
            if (a0.valid_flag && a1.valid_flag && b0.valid_flag && b1.valid_flag && da > 0.f &&
                db > 0.f) {
                const float angle_a = atan2f(
                    glm::length(glm::cross(a0.ray_dir, a1.ray_dir)), glm::dot(a0.ray_dir, a1.ray_dir)
                );
                const float angle_b = atan2f(
                    glm::length(glm::cross(b0.ray_dir, b1.ray_dir)), glm::dot(b0.ray_dir, b1.ray_dir)
                );
                footprint = 0.5f * (angle_a / da + angle_b / db);
            }
        }
    );
    return footprint;
}

// q^T adj(diag(a)) q and its gradient w.r.t. a.
inline __device__ float eval3d_adjugate_form(const glm::vec3 &q, const glm::vec3 &a) {
    return q.x * q.x * a.y * a.z + q.y * q.y * a.x * a.z + q.z * q.z * a.x * a.y;
}

inline __device__ glm::vec3 eval3d_adjugate_form_grad(const glm::vec3 &q, const glm::vec3 &a) {
    return {
        q.y * q.y * a.z + q.z * q.z * a.y,
        q.x * q.x * a.z + q.z * q.z * a.x,
        q.x * q.x * a.y + q.y * q.y * a.x,
    };
}

struct FootprintFilter {
    glm::vec3 widened_scale; // sqrt(scale^2 + added variance)
    float compensation;      // response multiplier, in [0, 1]
    float t;                 // distance along the ray to the centre (>= 0)
    glm::vec3 q;             // R^T ray_d
    float f_a, f_b;          // adjugate forms of scale^2 and widened_scale^2
};

inline __device__ FootprintFilter eval3d_footprint_filter(
    const glm::mat3 &R,
    const glm::vec3 &scale,
    const glm::vec3 &mean,
    const glm::vec3 &ray_o,
    const glm::vec3 &ray_d,
    const float spread2
) {
    FootprintFilter f;
    f.t = fmaxf(glm::dot(mean - ray_o, ray_d), 0.f);
    const float added = spread2 * f.t * f.t;
    const glm::vec3 a = scale * scale;
    const glm::vec3 b = a + added;
    f.widened_scale = {sqrtf(b.x), sqrtf(b.y), sqrtf(b.z)};
    f.q = glm::transpose(R) * ray_d;
    f.f_a = eval3d_adjugate_form(f.q, a);
    f.f_b = eval3d_adjugate_form(f.q, b);
    f.compensation = sqrtf(fminf(f.f_a / fmaxf(f.f_b, 1e-30f), 1.f));
    return f;
}

} // namespace

/****************************************************************************
 * Forward
 ****************************************************************************/

template <uint32_t CDIM>
__global__ void rasterize_to_pixels_eval3d_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const glm::vec3 *__restrict__ means,   // [N, 3]
    const glm::vec4 *__restrict__ quats,   // [N, 4]
    const glm::vec3 *__restrict__ scales,  // [N, 3]
    const float *__restrict__ colors,      // [C, N, CDIM]
    const float *__restrict__ opacities,   // [C, N]
    const float *__restrict__ backgrounds, // [C, CDIM] optional
    const bool *__restrict__ masks,        // [C, tile_height, tile_width] optional
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // camera model
    const float *__restrict__ viewmats0, // [C, 4, 4]
    const float *__restrict__ viewmats1, // [C, 4, 4] optional for rolling shutter
    const float *__restrict__ Ks,        // [C, 3, 3]
    const CameraModelType camera_model_type,
    const ShutterType rs_type,
    const float *__restrict__ radial_coeffs,     // [C, 6] or [C, 4] optional
    const float *__restrict__ tangential_coeffs, // [C, 2] optional
    const float *__restrict__ thin_prism_coeffs, // [C, 4] optional
    const float footprint_variance,              // ray-footprint filter, px^2; 0 disables
    // intersections
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // outputs
    float *__restrict__ render_colors, // [C, image_height, image_width, CDIM]
    float *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    int32_t *__restrict__ last_ids     // [C, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile
    auto block = cg::this_thread_block();
    const int32_t cid = block.group_index().x;
    const int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += cid * tile_height * tile_width;
    render_colors += cid * image_height * image_width * CDIM;
    render_alphas += cid * image_height * image_width;
    last_ids += cid * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += cid * CDIM;
    }
    if (masks != nullptr) {
        masks += cid * tile_height * tile_width;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = i * image_width + j;

    const WorldRay ray = eval3d_pixel_ray(
        px, py, cid, image_width, image_height, viewmats0, viewmats1, Ks,
        camera_model_type, rs_type, radial_coeffs, tangential_coeffs, thin_prism_coeffs
    );
    const glm::vec3 ray_d = ray.ray_dir;
    const glm::vec3 ray_o = ray.ray_org;
    const bool filtered = footprint_variance > 0.f;
    float spread2 = 0.f;
    if (filtered) {
        const float footprint = eval3d_pixel_footprint(
            px, py, cid, image_width, image_height, Ks, camera_model_type, rs_type,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs
        );
        spread2 = footprint_variance * footprint * footprint;
    }

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    const bool inside = (i < image_height && j < image_width);
    bool done = (!inside) || (!ray.valid_flag);

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] = backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        render_alphas[pix_id] = 0.0f;
        last_ids[pix_id] = 0;
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = reinterpret_cast<int32_t *>(s);                            // [block_size]
    glm::vec4 *xyz_opacity_batch = reinterpret_cast<glm::vec4 *>(&id_batch[block_size]); // [block_size]
    // S^-1 R^T, or R when filtered
    glm::mat3 *iscl_rot_batch =
        reinterpret_cast<glm::mat3 *>(&xyz_opacity_batch[block_size]); // [block_size]
    glm::vec3 *scale_batch = reinterpret_cast<glm::vec3 *>(&iscl_rot_batch[block_size]); // [block_size]

    // current visibility left to render
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    const uint32_t tr = block.thread_rank();

    float pix_out[CDIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            const int32_t isect_id = flatten_ids[idx]; // flatten index in [C * N]
            const int32_t isect_gid = isect_id % N;    // gaussian index
            id_batch[tr] = isect_id;
            const glm::vec3 xyz = means[isect_gid];
            const float opac = opacities[isect_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
            const glm::vec4 quat = quats[isect_gid];
            const glm::vec3 scale = scales[isect_gid];
            if (filtered) {
                iscl_rot_batch[tr] = quat_to_rotmat(quat);
                scale_batch[tr] = scale;
            } else {
                iscl_rot_batch[tr] = eval3d_inverse_scale_rotation(quat_to_rotmat(quat), scale);
            }
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        const uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const glm::vec4 xyz_opac = xyz_opacity_batch[t];
            const float opac = xyz_opac[3];
            const glm::vec3 xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
            glm::mat3 iscl_rot = iscl_rot_batch[t];
            float compensation = 1.f;
            if (filtered) {
                const FootprintFilter f = eval3d_footprint_filter(
                    iscl_rot, scale_batch[t], xyz, ray_o, ray_d, spread2
                );
                iscl_rot = eval3d_inverse_scale_rotation(iscl_rot, f.widened_scale);
                compensation = f.compensation;
            }

            const glm::vec3 gro = iscl_rot * (ray_o - xyz);
            const glm::vec3 grd = safe_normalize(iscl_rot * ray_d);
            // closest approach behind the camera origin: the forward half-line
            // never meets this Gaussian
            const float hit_t = -glm::dot(grd, gro);
            if (hit_t < 0.f) {
                continue;
            }
            const glm::vec3 gcrod = glm::cross(grd, gro);
            const float gray_dist = glm::dot(gcrod, gcrod);
            const float power = -0.5f * gray_dist;

            const float alpha = min(0.999f, opac * compensation * __expf(power));
            if (alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done: exclusive
                done = true;
                break;
            }

            const int32_t isect_id = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + isect_id * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? pix_out[k] : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rasterize_to_pixels_eval3d_fwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means,                     // [N, 3]
    const torch::Tensor &quats,                     // [N, 4]
    const torch::Tensor &scales,                    // [N, 3]
    const torch::Tensor &colors,                    // [C, N, channels]
    const torch::Tensor &opacities,                 // [C, N]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const torch::Tensor &viewmats0,               // [C, 4, 4]
    const at::optional<torch::Tensor> &viewmats1, // [C, 4, 4] optional for rolling shutter
    const torch::Tensor &Ks,                      // [C, 3, 3]
    const CameraModelType camera_model,
    const ShutterType rs_type,
    const at::optional<torch::Tensor> &radial_coeffs,     // [C, 6] or [C, 4] optional
    const at::optional<torch::Tensor> &tangential_coeffs, // [C, 2] optional
    const at::optional<torch::Tensor> &thin_prism_coeffs, // [C, 4] optional
    const float footprint_variance,                       // ray-footprint filter, px^2; 0 disables
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(viewmats0);
    CHECK_INPUT(Ks);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
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
    TORCH_CHECK(opacities.dim() == 2, "packed mode is not supported by the eval3d rasterizer");
    TORCH_CHECK(
        camera_model == CameraModelType::PINHOLE || camera_model == CameraModelType::FISHEYE,
        "only the pinhole and fisheye camera models are implemented"
    );

    const uint32_t N = means.size(0);
    const uint32_t C = viewmats0.size(0);
    const uint32_t channels = colors.size(-1);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width = tile_offsets.size(2);
    const uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor renders =
        torch::empty({C, image_height, image_width, channels}, means.options());
    torch::Tensor alphas = torch::empty({C, image_height, image_width, 1}, means.options());
    torch::Tensor last_ids =
        torch::empty({C, image_height, image_width}, means.options().dtype(torch::kInt32));

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(glm::vec4) + sizeof(glm::mat3) + sizeof(glm::vec3));

#define LAUNCH_EVAL3D_FWD(CDIM)                                                            \
    case CDIM:                                                                                 \
        if (cudaFuncSetAttribute(                                                              \
                rasterize_to_pixels_eval3d_fwd_kernel<CDIM>,                                   \
                cudaFuncAttributeMaxDynamicSharedMemorySize,                                   \
                shared_mem                                                                     \
            ) != cudaSuccess) {                                                                \
            AT_ERROR(                                                                          \
                "Failed to set maximum shared memory size (requested ",                        \
                shared_mem,                                                                    \
                " bytes), try lowering tile_size."                                             \
            );                                                                                 \
        }                                                                                      \
        rasterize_to_pixels_eval3d_fwd_kernel<CDIM><<<blocks, threads, shared_mem, stream>>>( \
            C,                                                                                 \
            N,                                                                                 \
            n_isects,                                                                          \
            reinterpret_cast<glm::vec3 *>(means.data_ptr<float>()),                            \
            reinterpret_cast<glm::vec4 *>(quats.data_ptr<float>()),                            \
            reinterpret_cast<glm::vec3 *>(scales.data_ptr<float>()),                           \
            colors.data_ptr<float>(),                                                          \
            opacities.data_ptr<float>(),                                                       \
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,         \
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,                      \
            image_width,                                                                       \
            image_height,                                                                      \
            tile_size,                                                                         \
            tile_width,                                                                        \
            tile_height,                                                                       \
            viewmats0.data_ptr<float>(),                                                       \
            viewmats1.has_value() ? viewmats1.value().data_ptr<float>() : nullptr,             \
            Ks.data_ptr<float>(),                                                              \
            camera_model,                                                                      \
            rs_type,                                                                           \
            radial_coeffs.has_value() ? radial_coeffs.value().data_ptr<float>() : nullptr,     \
            tangential_coeffs.has_value() ? tangential_coeffs.value().data_ptr<float>()        \
                                          : nullptr,                                           \
            thin_prism_coeffs.has_value() ? thin_prism_coeffs.value().data_ptr<float>()        \
                                          : nullptr,                                           \
            footprint_variance,                                                                \
            tile_offsets.data_ptr<int32_t>(),                                                  \
            flatten_ids.data_ptr<int32_t>(),                                                   \
            renders.data_ptr<float>(),                                                         \
            alphas.data_ptr<float>(),                                                          \
            last_ids.data_ptr<int32_t>()                                                       \
        );                                                                                     \
        break;

    switch (channels) {
        LAUNCH_EVAL3D_FWD(1)
        LAUNCH_EVAL3D_FWD(2)
        LAUNCH_EVAL3D_FWD(3)
        LAUNCH_EVAL3D_FWD(4)
        LAUNCH_EVAL3D_FWD(5)
        LAUNCH_EVAL3D_FWD(8)
        LAUNCH_EVAL3D_FWD(9)
        LAUNCH_EVAL3D_FWD(16)
        LAUNCH_EVAL3D_FWD(17)
        LAUNCH_EVAL3D_FWD(32)
        LAUNCH_EVAL3D_FWD(33)
        LAUNCH_EVAL3D_FWD(64)
        LAUNCH_EVAL3D_FWD(65)
        LAUNCH_EVAL3D_FWD(128)
        LAUNCH_EVAL3D_FWD(129)
        LAUNCH_EVAL3D_FWD(256)
        LAUNCH_EVAL3D_FWD(257)
        LAUNCH_EVAL3D_FWD(512)
        LAUNCH_EVAL3D_FWD(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
#undef LAUNCH_EVAL3D_FWD
    return std::make_tuple(renders, alphas, last_ids);
}

/****************************************************************************
 * Backward
 ****************************************************************************/

template <uint32_t CDIM>
__global__ void rasterize_to_pixels_eval3d_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    // fwd inputs
    const glm::vec3 *__restrict__ means,   // [N, 3]
    const glm::vec4 *__restrict__ quats,   // [N, 4]
    const glm::vec3 *__restrict__ scales,  // [N, 3]
    const float *__restrict__ colors,      // [C, N, CDIM]
    const float *__restrict__ opacities,   // [C, N]
    const float *__restrict__ backgrounds, // [C, CDIM] optional
    const bool *__restrict__ masks,        // [C, tile_height, tile_width] optional
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // camera model
    const float *__restrict__ viewmats0, // [C, 4, 4]
    const float *__restrict__ viewmats1, // [C, 4, 4] optional for rolling shutter
    const float *__restrict__ Ks,        // [C, 3, 3]
    const CameraModelType camera_model_type,
    const ShutterType rs_type,
    const float *__restrict__ radial_coeffs,     // [C, 6] or [C, 4] optional
    const float *__restrict__ tangential_coeffs, // [C, 2] optional
    const float *__restrict__ thin_prism_coeffs, // [C, 4] optional
    const float footprint_variance,              // ray-footprint filter, px^2; 0 disables
    // intersections
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const float *__restrict__ render_alphas, // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,    // [C, image_height, image_width]
    // grad outputs
    const float *__restrict__ v_render_colors, // [C, image_height, image_width, CDIM]
    const float *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    // grad inputs
    glm::vec3 *__restrict__ v_means,  // [N, 3]
    glm::vec4 *__restrict__ v_quats,  // [N, 4]
    glm::vec3 *__restrict__ v_scales, // [N, 3]
    float *__restrict__ v_colors,     // [C, N, CDIM]
    float *__restrict__ v_opacities   // [C, N]
) {
    auto block = cg::this_thread_block();
    const uint32_t cid = block.group_index().x;
    const uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += cid * tile_height * tile_width;
    render_alphas += cid * image_height * image_width;
    last_ids += cid * image_height * image_width;
    v_render_colors += cid * image_height * image_width * CDIM;
    v_render_alphas += cid * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += cid * CDIM;
    }
    if (masks != nullptr) {
        masks += cid * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    const WorldRay ray = eval3d_pixel_ray(
        px, py, cid, image_width, image_height, viewmats0, viewmats1, Ks,
        camera_model_type, rs_type, radial_coeffs, tangential_coeffs, thin_prism_coeffs
    );
    const glm::vec3 ray_d = ray.ray_dir;
    const glm::vec3 ray_o = ray.ray_org;
    const bool filtered = footprint_variance > 0.f;
    float spread2 = 0.f;
    if (filtered) {
        const float footprint = eval3d_pixel_footprint(
            px, py, cid, image_width, image_height, Ks, camera_model_type, rs_type,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs
        );
        spread2 = footprint_variance * footprint * footprint;
    }

    // keep not rasterizing threads around for reading data
    const bool active = (i < image_height && j < image_width) && ray.valid_flag;

    // have all threads in tile process the same gaussians in batches
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = reinterpret_cast<int32_t *>(s);                                    // [block_size]
    glm::vec4 *xyz_opacity_batch = reinterpret_cast<glm::vec4 *>(&id_batch[block_size]);   // [block_size]
    glm::vec3 *scale_batch = reinterpret_cast<glm::vec3 *>(&xyz_opacity_batch[block_size]); // [block_size]
    glm::vec4 *quat_batch = reinterpret_cast<glm::vec4 *>(&scale_batch[block_size]);        // [block_size]
    float *rgbs_batch = reinterpret_cast<float *>(&quat_batch[block_size]);                // [block_size * CDIM]

    // this is the T AFTER the last gaussian in this pixel
    const float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = active ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            const int32_t isect_id = flatten_ids[idx]; // flatten index in [C * N]
            const int32_t isect_gid = isect_id % N;    // gaussian index
            id_batch[tr] = isect_id;
            const glm::vec3 xyz = means[isect_gid];
            const float opac = opacities[isect_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
            scale_batch[tr] = scales[isect_gid];
            quat_batch[tr] = quats[isect_gid];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[isect_id * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = active;
            if (batch_end - t > bin_final) {
                valid = false;
            }
            float alpha;
            float opac;
            float vis;

            glm::mat3 R;
            glm::vec3 xyz;
            glm::vec3 scale;
            glm::vec4 quat;
            glm::mat3 Mt;
            glm::vec3 o_minus_mu, gro, grd, grd_n, gcrod;
            FootprintFilter filter;
            float compensation = 1.f;
            if (valid) {
                const glm::vec4 xyz_opac = xyz_opacity_batch[t];
                opac = xyz_opac[3];
                xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
                scale = scale_batch[t];
                quat = quat_batch[t];

                R = quat_to_rotmat(quat);
                if (filtered) {
                    filter = eval3d_footprint_filter(R, scale, xyz, ray_o, ray_d, spread2);
                    compensation = filter.compensation;
                    Mt = eval3d_inverse_scale_rotation(R, filter.widened_scale);
                } else {
                    Mt = eval3d_inverse_scale_rotation(R, scale);
                }
                o_minus_mu = ray_o - xyz;
                gro = Mt * o_minus_mu;
                grd = Mt * ray_d;
                grd_n = safe_normalize(grd);
                // closest approach behind the camera origin: skipped in forward
                const float hit_t = -glm::dot(grd_n, gro);
                if (hit_t < 0.f) {
                    valid = false;
                } else {
                    gcrod = glm::cross(grd_n, gro);
                    const float gray_dist = glm::dot(gcrod, gcrod);
                    const float power = -0.5f * gray_dist;
                    vis = __expf(power);
                    alpha = min(0.999f, opac * compensation * vis);
                    if (alpha < ALPHA_THRESHOLD) {
                        valid = false;
                    }
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            glm::vec3 v_mean_local = {0.f, 0.f, 0.f};
            glm::vec3 v_scale_local = {0.f, 0.f, 0.f};
            glm::vec4 v_quat_local = {0.f, 0.f, 0.f, 0.f};
            float v_opacity_local = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                const float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) * v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * compensation * vis <= 0.999f) {
                    const float v_vis = opac * compensation * v_alpha;
                    const float v_grad_dist = -0.5f * vis * v_vis;
                    const glm::vec3 v_gcrod = 2.0f * v_grad_dist * gcrod;
                    const glm::vec3 v_grd_n = -glm::cross(v_gcrod, gro);
                    const glm::vec3 v_gro = glm::cross(v_gcrod, grd_n);
                    const glm::vec3 v_grd = safe_normalize_bw(grd, v_grd_n);
                    const glm::mat3 v_Mt =
                        glm::outerProduct(v_grd, ray_d) + glm::outerProduct(v_gro, o_minus_mu);
                    const glm::vec3 v_o_minus_mu = glm::transpose(Mt) * v_gro;

                    v_mean_local += -v_o_minus_mu;
                    if (filtered) {
                        // w = sqrt(b), b = scale^2 + spread2 t^2, c = sqrt(f(a) / f(b))
                        const glm::vec3 w = filter.widened_scale;
                        glm::vec3 v_w = {0.f, 0.f, 0.f};
                        quat_scale_to_preci_half_vjp(
                            quat, w, R, glm::transpose(v_Mt), v_quat_local, v_w
                        );
                        glm::vec3 v_b = v_w / (2.f * w);
                        glm::vec3 v_a = {0.f, 0.f, 0.f};
                        if (compensation < 1.f && filter.f_a > 1e-30f) {
                            const glm::vec3 a = scale * scale;
                            const glm::vec3 b = w * w;
                            const float v_log_c = opac * vis * v_alpha * compensation;
                            v_a += 0.5f * v_log_c * eval3d_adjugate_form_grad(filter.q, a) / filter.f_a;
                            v_b -= 0.5f * v_log_c * eval3d_adjugate_form_grad(filter.q, b) / filter.f_b;
                            const glm::vec3 q = filter.q;
                            const glm::vec3 v_q =
                                v_log_c * q *
                                (glm::vec3(a.y * a.z, a.x * a.z, a.x * a.y) / filter.f_a -
                                 glm::vec3(b.y * b.z, b.x * b.z, b.x * b.y) / filter.f_b);
                            quat_to_rotmat_vjp(quat, glm::outerProduct(ray_d, v_q), v_quat_local);
                        }
                        v_a += v_b;
                        v_scale_local += 2.f * scale * v_a;
                        if (filter.t > 0.f) {
                            v_mean_local +=
                                (v_b.x + v_b.y + v_b.z) * 2.f * spread2 * filter.t * ray_d;
                        }
                    } else {
                        quat_scale_to_preci_half_vjp(
                            quat, scale, R, glm::transpose(v_Mt), v_quat_local, v_scale_local
                        );
                    }
                    v_opacity_local = compensation * vis * v_alpha;
                }

#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
            warpSum<CDIM, float>(v_rgb_local, warp);
            warpSum(v_mean_local, warp);
            warpSum(v_scale_local, warp);
            warpSum(v_quat_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                const int32_t isect_id = id_batch[t]; // flatten index in [C * N]
                const int32_t isect_gid = isect_id % N;
                float *v_rgb_ptr = v_colors + CDIM * isect_id;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_mean_ptr = reinterpret_cast<float *>(v_means) + 3 * isect_gid;
                gpuAtomicAdd(v_mean_ptr, v_mean_local.x);
                gpuAtomicAdd(v_mean_ptr + 1, v_mean_local.y);
                gpuAtomicAdd(v_mean_ptr + 2, v_mean_local.z);

                float *v_scale_ptr = reinterpret_cast<float *>(v_scales) + 3 * isect_gid;
                gpuAtomicAdd(v_scale_ptr, v_scale_local.x);
                gpuAtomicAdd(v_scale_ptr + 1, v_scale_local.y);
                gpuAtomicAdd(v_scale_ptr + 2, v_scale_local.z);

                float *v_quat_ptr = reinterpret_cast<float *>(v_quats) + 4 * isect_gid;
                gpuAtomicAdd(v_quat_ptr, v_quat_local.x);
                gpuAtomicAdd(v_quat_ptr + 1, v_quat_local.y);
                gpuAtomicAdd(v_quat_ptr + 2, v_quat_local.z);
                gpuAtomicAdd(v_quat_ptr + 3, v_quat_local.w);

                gpuAtomicAdd(v_opacities + isect_id, v_opacity_local);
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_eval3d_bwd_tensor(
    // Gaussian parameters
    const torch::Tensor &means,                     // [N, 3]
    const torch::Tensor &quats,                     // [N, 4]
    const torch::Tensor &scales,                    // [N, 3]
    const torch::Tensor &colors,                    // [C, N, channels]
    const torch::Tensor &opacities,                 // [C, N]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const torch::Tensor &viewmats0,               // [C, 4, 4]
    const at::optional<torch::Tensor> &viewmats1, // [C, 4, 4] optional for rolling shutter
    const torch::Tensor &Ks,                      // [C, 3, 3]
    const CameraModelType camera_model,
    const ShutterType rs_type,
    const at::optional<torch::Tensor> &radial_coeffs,     // [C, 6] or [C, 4] optional
    const at::optional<torch::Tensor> &tangential_coeffs, // [C, 2] optional
    const at::optional<torch::Tensor> &thin_prism_coeffs, // [C, 4] optional
    const float footprint_variance,                       // ray-footprint filter, px^2; 0 disables
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
    const torch::Tensor &last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const torch::Tensor &v_render_colors, // [C, image_height, image_width, channels]
    const torch::Tensor &v_render_alphas  // [C, image_height, image_width, 1]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(viewmats0);
    CHECK_INPUT(Ks);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(v_render_colors);
    CHECK_INPUT(v_render_alphas);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
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
    TORCH_CHECK(opacities.dim() == 2, "packed mode is not supported by the eval3d rasterizer");

    const uint32_t N = means.size(0);
    const uint32_t C = viewmats0.size(0);
    const uint32_t channels = colors.size(-1);
    const uint32_t tile_height = tile_offsets.size(1);
    const uint32_t tile_width = tile_offsets.size(2);
    const uint32_t n_isects = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_quats = torch::zeros_like(quats);
    torch::Tensor v_scales = torch::zeros_like(scales);
    torch::Tensor v_colors = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);

    if (n_isects == 0) {
        return std::make_tuple(v_means, v_quats, v_scales, v_colors, v_opacities);
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(glm::vec4) + sizeof(glm::vec3) + sizeof(glm::vec4) +
         sizeof(float) * channels);

#define LAUNCH_EVAL3D_BWD(CDIM)                                                            \
    case CDIM:                                                                                 \
        if (cudaFuncSetAttribute(                                                              \
                rasterize_to_pixels_eval3d_bwd_kernel<CDIM>,                                   \
                cudaFuncAttributeMaxDynamicSharedMemorySize,                                   \
                shared_mem                                                                     \
            ) != cudaSuccess) {                                                                \
            AT_ERROR(                                                                          \
                "Failed to set maximum shared memory size (requested ",                        \
                shared_mem,                                                                    \
                " bytes), try lowering tile_size or channel_chunk."                            \
            );                                                                                 \
        }                                                                                      \
        rasterize_to_pixels_eval3d_bwd_kernel<CDIM><<<blocks, threads, shared_mem, stream>>>( \
            C,                                                                                 \
            N,                                                                                 \
            n_isects,                                                                          \
            reinterpret_cast<glm::vec3 *>(means.data_ptr<float>()),                            \
            reinterpret_cast<glm::vec4 *>(quats.data_ptr<float>()),                            \
            reinterpret_cast<glm::vec3 *>(scales.data_ptr<float>()),                           \
            colors.data_ptr<float>(),                                                          \
            opacities.data_ptr<float>(),                                                       \
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,         \
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,                      \
            image_width,                                                                       \
            image_height,                                                                      \
            tile_size,                                                                         \
            tile_width,                                                                        \
            tile_height,                                                                       \
            viewmats0.data_ptr<float>(),                                                       \
            viewmats1.has_value() ? viewmats1.value().data_ptr<float>() : nullptr,             \
            Ks.data_ptr<float>(),                                                              \
            camera_model,                                                                      \
            rs_type,                                                                           \
            radial_coeffs.has_value() ? radial_coeffs.value().data_ptr<float>() : nullptr,     \
            tangential_coeffs.has_value() ? tangential_coeffs.value().data_ptr<float>()        \
                                          : nullptr,                                           \
            thin_prism_coeffs.has_value() ? thin_prism_coeffs.value().data_ptr<float>()        \
                                          : nullptr,                                           \
            footprint_variance,                                                                \
            tile_offsets.data_ptr<int32_t>(),                                                  \
            flatten_ids.data_ptr<int32_t>(),                                                   \
            render_alphas.data_ptr<float>(),                                                   \
            last_ids.data_ptr<int32_t>(),                                                      \
            v_render_colors.data_ptr<float>(),                                                 \
            v_render_alphas.data_ptr<float>(),                                                 \
            reinterpret_cast<glm::vec3 *>(v_means.data_ptr<float>()),                          \
            reinterpret_cast<glm::vec4 *>(v_quats.data_ptr<float>()),                          \
            reinterpret_cast<glm::vec3 *>(v_scales.data_ptr<float>()),                         \
            v_colors.data_ptr<float>(),                                                        \
            v_opacities.data_ptr<float>()                                                      \
        );                                                                                     \
        break;

    switch (channels) {
        LAUNCH_EVAL3D_BWD(1)
        LAUNCH_EVAL3D_BWD(2)
        LAUNCH_EVAL3D_BWD(3)
        LAUNCH_EVAL3D_BWD(4)
        LAUNCH_EVAL3D_BWD(5)
        LAUNCH_EVAL3D_BWD(8)
        LAUNCH_EVAL3D_BWD(9)
        LAUNCH_EVAL3D_BWD(16)
        LAUNCH_EVAL3D_BWD(17)
        LAUNCH_EVAL3D_BWD(32)
        LAUNCH_EVAL3D_BWD(33)
        LAUNCH_EVAL3D_BWD(64)
        LAUNCH_EVAL3D_BWD(65)
        LAUNCH_EVAL3D_BWD(128)
        LAUNCH_EVAL3D_BWD(129)
        LAUNCH_EVAL3D_BWD(256)
        LAUNCH_EVAL3D_BWD(257)
        LAUNCH_EVAL3D_BWD(512)
        LAUNCH_EVAL3D_BWD(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
#undef LAUNCH_EVAL3D_BWD
    return std::make_tuple(v_means, v_quats, v_scales, v_colors, v_opacities);
}
