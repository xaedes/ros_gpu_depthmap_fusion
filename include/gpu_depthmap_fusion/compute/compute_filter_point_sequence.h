#pragma once
#include <vector>
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeFilterPointSequence : public ComputeProgram
{
public:
    virtual ~ComputeFilterPointSequence() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_in_mask, 
        uint bufbind_out_mask, 
        uint bufbind_points)
    {
        m_groupsize_x = 32;
        m_groupsize_y = 32;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "filter_point_sequence.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_MASK",    std::to_string(bufbind_in_mask)},
            {"BUFBIND_OUT_MASK",   std::to_string(bufbind_out_mask)},
            {"BUFBIND_IN_POINTS",  std::to_string(bufbind_points)}
        });
        num_items.init(m_glProgram, "num_items");
        in_point_offset.init(m_glProgram, "in_point_offset");
        in_mask_offset.init(m_glProgram, "in_mask_offset");
        out_mask_offset.init(m_glProgram, "out_mask_offset");
        filter_size.init(m_glProgram, "filter_size");
        threshold_view_angle.init(m_glProgram, "threshold_view_angle");
    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> in_point_offset;
    ProgramUniform<uint> in_mask_offset;
    ProgramUniform<uint> out_mask_offset;
    ProgramUniform<uint> filter_size;
    ProgramUniform<float> threshold_view_angle;
    virtual ComputeProgram& dispatch()
    {
        dispatch(num_items.get(), 1, 1);
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

