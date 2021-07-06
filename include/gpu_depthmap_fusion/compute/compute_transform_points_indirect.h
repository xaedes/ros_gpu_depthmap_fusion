#pragma once
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeTransformPointsIndirect : public ComputeProgram
{
public:
    virtual ~ComputeTransformPointsIndirect() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_in_transforms, 
        uint bufbind_in_transform_indices, 
        uint bufbind_in_points, 
        uint bufbind_in_mask, 
        uint bufbind_out_points)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "transform_points_indirect.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_TRANSFORMS",       std::to_string(bufbind_in_transforms)},
            {"BUFBIND_IN_TRANSFORM_INDICES",std::to_string(bufbind_in_transform_indices)},
            {"BUFBIND_IN_POINTS",           std::to_string(bufbind_in_points)},
            {"BUFBIND_IN_MASK",             std::to_string(bufbind_in_mask)},
            {"BUFBIND_OUT_POINTS",          std::to_string(bufbind_out_points)}
        });
        num_items.init(m_glProgram, "num_items");
        in_transforms_offset.init(m_glProgram, "in_transforms_offset");
        in_transform_indices_offset.init(m_glProgram, "in_transform_indices_offset");
        in_points_offset.init(m_glProgram, "in_points_offset");
        in_mask_offset.init(m_glProgram, "in_mask_offset");
        out_points_offset.init(m_glProgram, "out_points_offset");
    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> in_transforms_offset;
    ProgramUniform<uint> in_transform_indices_offset;
    ProgramUniform<uint> in_points_offset;
    ProgramUniform<uint> in_mask_offset;
    ProgramUniform<uint> out_points_offset;

    virtual ComputeProgram& dispatch()
    {
        dispatch(num_items.get());
        return *this;
    }
    virtual ComputeProgram& dispatch(int count)
    {
        dispatch(count, 1, 1);
        return *this;
    }
    virtual ComputeProgram& dispatch(int x, int y, int z)
    {
        ComputeProgram::dispatch(x,y,z,m_groupsize_x,m_groupsize_y,m_groupsize_z);
        return *this;
    }
protected:
    uint m_groupsize_x;
    uint m_groupsize_y;
    uint m_groupsize_z;
};