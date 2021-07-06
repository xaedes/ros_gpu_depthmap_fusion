#pragma once

#include "gpu_depthmap_fusion/compute_program.h"

#include "gpu_depthmap_fusion/compute/compute_transform_points.h"
#include "gpu_depthmap_fusion/compute/compute_transform_points_indirect.h"

#include "gpu_depthmap_fusion/compute/compute_transfer_data.h"
#include "gpu_depthmap_fusion/compute/compute_transfer_data_from.h"
#include "gpu_depthmap_fusion/compute/compute_transfer_masked_data.h"

#include "gpu_depthmap_fusion/compute/compute_add_uints_times_scalar.h"
#include "gpu_depthmap_fusion/compute/compute_decrement_uints.h"
#include "gpu_depthmap_fusion/compute/compute_uints_to_chars.h"
#include "gpu_depthmap_fusion/compute/compute_chars_to_uints.h"
#include "gpu_depthmap_fusion/compute/compute_uints_to_words.h"
#include "gpu_depthmap_fusion/compute/compute_words_to_uints.h"
#include "gpu_depthmap_fusion/compute/compute_vec4_to_vec3.h"
#include "gpu_depthmap_fusion/compute/compute_vec3_to_vec4.h"
#include "gpu_depthmap_fusion/compute/compute_zero_uints.h"
#include "gpu_depthmap_fusion/compute/compute_set_uints.h"
#include "gpu_depthmap_fusion/compute/compute_max_with_uints_times_scalar.h"

#include "gpu_depthmap_fusion/compute/compute_convert_depthmap_to_points.h"
#include "gpu_depthmap_fusion/compute/compute_filter_flying_pixels.h"

#include "gpu_depthmap_fusion/compute/compute_filter_point_sequence.h"

#include "gpu_depthmap_fusion/compute/compute_rollbuffer_count_discarded_points.h"
#include "gpu_depthmap_fusion/compute/compute_rollbuffer_count_discarded_seqs.h"
#include "gpu_depthmap_fusion/compute/compute_rollbuffer_remaining_points_indices.h"
#include "gpu_depthmap_fusion/compute/compute_rollbuffer_remaining_points_copy_and_update.h"
#include "gpu_depthmap_fusion/compute/compute_rollbuffer_remaining_seqs_indices.h"
#include "gpu_depthmap_fusion/compute/compute_rollbuffer_select_timespan_points.h"
#include "gpu_depthmap_fusion/compute/compute_rollbuffer_select_timespan_sequences.h"
#include "gpu_depthmap_fusion/compute/compute_rollbuffer_transfer_selected_transform_indices.h"
#include "gpu_depthmap_fusion/compute/compute_rollbuffer_transfer_selected_transforms.h"

#include "gpu_depthmap_fusion/compute/compute_crop_points.h"
#include "gpu_depthmap_fusion/compute/compute_apply_point_mask.h"

#include "gpu_depthmap_fusion/compute/compute_voxel_coords.h"
#include "gpu_depthmap_fusion/compute/compute_voxel_grid_occupancy_of_points.h"
#include "gpu_depthmap_fusion/compute/compute_layers_connections.h"

