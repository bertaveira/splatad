#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_sh_fwd", &compute_sh_fwd_tensor);
    m.def("compute_sh_bwd", &compute_sh_bwd_tensor);

    m.def("quat_scale_to_covar_preci_fwd", &quat_scale_to_covar_preci_fwd_tensor);
    m.def("quat_scale_to_covar_preci_bwd", &quat_scale_to_covar_preci_bwd_tensor);

    m.def("persp_proj_fwd", &persp_proj_fwd_tensor);
    m.def("persp_proj_bwd", &persp_proj_bwd_tensor);

    m.def("lidar_proj_fwd", &lidar_proj_fwd_tensor);
    m.def("lidar_proj_bwd", &lidar_proj_bwd_tensor);

    m.def("world_to_cam_fwd", &world_to_cam_fwd_tensor);
    m.def("world_to_cam_bwd", &world_to_cam_bwd_tensor);

    m.def("compute_pix_velocity_fwd", &compute_pix_velocity_fwd_tensor);
    m.def("compute_pix_velocity_bwd", &compute_pix_velocity_bwd_tensor);
    
    m.def("compute_lidar_velocity_fwd", &compute_lidar_velocity_fwd_tensor);
    m.def("compute_lidar_velocity_bwd", &compute_lidar_velocity_bwd_tensor);

    m.def("fully_fused_projection_fwd", &fully_fused_projection_fwd_tensor);
    m.def("fully_fused_projection_bwd", &fully_fused_projection_bwd_tensor);

    m.def("fully_fused_lidar_projection_fwd", &fully_fused_lidar_projection_fwd_tensor);
    m.def("fully_fused_lidar_projection_bwd", &fully_fused_lidar_projection_bwd_tensor);

    m.def("isect_tiles", &isect_tiles_tensor);
    m.def("isect_lidar_tiles", &isect_lidar_tiles_tensor);
    m.def("isect_offset_encode", &isect_offset_encode_tensor);

    m.def("map_points_to_lidar_tiles", &map_points_to_lidar_tiles_tensor);
    m.def("points_mapping_offset_encode", &points_mapping_offset_encode_tensor);
    m.def("populate_image_from_points", &populate_image_from_points_tensor);

    m.def("rasterize_to_pixels_fwd", &rasterize_to_pixels_fwd_tensor);
    m.def("rasterize_to_pixels_bwd", &rasterize_to_pixels_bwd_tensor);

    m.def("rasterize_to_points_fwd", &rasterize_to_points_fwd_tensor);
    m.def("rasterize_to_points_bwd", &rasterize_to_points_bwd_tensor);

    m.def("rasterize_to_indices_in_range", &rasterize_to_indices_in_range_tensor);
    m.def("rasterize_to_indices_in_range_lidar", &rasterize_to_indices_in_range_lidar_tensor);

    // packed version
    m.def("fully_fused_projection_packed_fwd", &fully_fused_projection_packed_fwd_tensor);
    m.def("fully_fused_projection_packed_bwd", &fully_fused_projection_packed_bwd_tensor);
    
    m.def("compute_relocation", &compute_relocation_tensor);

    // unscented-transform projection + eval3d rasterization (fisheye / 3DGUT)
    py::enum_<CameraModelType>(m, "CameraModelType")
        .value("PINHOLE", CameraModelType::PINHOLE)
        .value("FISHEYE", CameraModelType::FISHEYE)
        .export_values();
    py::enum_<ShutterType>(m, "ShutterType")
        .value("ROLLING_TOP_TO_BOTTOM", ShutterType::ROLLING_TOP_TO_BOTTOM)
        .value("ROLLING_LEFT_TO_RIGHT", ShutterType::ROLLING_LEFT_TO_RIGHT)
        .value("ROLLING_BOTTOM_TO_TOP", ShutterType::ROLLING_BOTTOM_TO_TOP)
        .value("ROLLING_RIGHT_TO_LEFT", ShutterType::ROLLING_RIGHT_TO_LEFT)
        .value("GLOBAL", ShutterType::GLOBAL)
        .export_values();
    py::class_<UnscentedTransformParameters>(m, "UnscentedTransformParameters")
        .def(py::init<>())
        .def_readwrite("alpha", &UnscentedTransformParameters::alpha)
        .def_readwrite("beta", &UnscentedTransformParameters::beta)
        .def_readwrite("kappa", &UnscentedTransformParameters::kappa)
        .def_readwrite("in_image_margin_factor", &UnscentedTransformParameters::in_image_margin_factor)
        .def_readwrite("require_all_sigma_points_valid", &UnscentedTransformParameters::require_all_sigma_points_valid);

    m.def("fully_fused_projection_ut_fwd", &fully_fused_projection_ut_fwd_tensor);
    m.def("rasterize_to_pixels_eval3d_fwd", &rasterize_to_pixels_eval3d_fwd_tensor);
    m.def("rasterize_to_pixels_eval3d_bwd", &rasterize_to_pixels_eval3d_bwd_tensor);
}