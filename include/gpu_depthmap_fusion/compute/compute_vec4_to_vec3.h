#pragma once
#include <vector>
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeVec4ToVec3 : public ComputeProgram
{
public:
    virtual ~ComputeVec4ToVec3() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_in_data, 
        uint bufbind_out_data)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "vec4_to_vec3.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_DATA",  std::to_string(bufbind_in_data)},
            {"BUFBIND_OUT_DATA", std::to_string(bufbind_out_data)}
        });
        num_items.init(m_glProgram, "num_items");
        in_data_offset.init(m_glProgram, "in_data_offset");
        out_data_offset.init(m_glProgram, "out_data_offset");
    }
    ProgramUniform<uint> num_items;
    ProgramUniform<uint> in_data_offset;
    ProgramUniform<uint> out_data_offset;

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

