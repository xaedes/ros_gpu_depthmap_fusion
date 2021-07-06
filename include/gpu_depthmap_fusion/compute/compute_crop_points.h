#pragma once
#include <vector>
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeCropPoints : public ComputeProgram
{
public:
    virtual ~ComputeCropPoints() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_in_mask, 
        uint bufbind_out_mask, 
        uint bufbind_points)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "crop_points.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_MASK",  std::to_string(bufbind_in_mask)},
            {"BUFBIND_OUT_MASK", std::to_string(bufbind_out_mask)},
            {"BUFBIND_POINTS",   std::to_string(bufbind_points)}
        });
        num_items.init(m_glProgram, "num_items");
        point_offset.init(m_glProgram, "point_offset");
        mask_offset.init(m_glProgram, "mask_offset");
        lower_bound.init(m_glProgram, "lower_bound");
        upper_bound.init(m_glProgram, "upper_bound");
    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> point_offset;
    ProgramUniform<uint> mask_offset;
    ProgramUniform<glm::vec3> lower_bound;
    ProgramUniform<glm::vec3> upper_bound;

    virtual ComputeProgram& dispatch()
    {
        dispatch(num_items.get(), 1, 1);
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

