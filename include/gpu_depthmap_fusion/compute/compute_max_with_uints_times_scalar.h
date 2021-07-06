#pragma once
#include <vector>
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


// out = in * multiplier
class ComputeMaxWithUintsTimesScalar : public ComputeProgram
{
public:
    virtual ~ComputeMaxWithUintsTimesScalar() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_in_a,
        uint bufbind_in_b,
        uint bufbind_out)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "max_with_uints_times_scalar.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_A", std::to_string(bufbind_in_a)},
            {"BUFBIND_IN_B", std::to_string(bufbind_in_b)},
            {"BUFBIND_OUT", std::to_string(bufbind_out)}
        });
        num_items.init(m_glProgram, "num_items");
        offset.init(m_glProgram, "offset");
        multiplier.init(m_glProgram, "multiplier");
    }
    ProgramUniform<uint> num_items;
    ProgramUniform<uint> offset;
    ProgramUniform<uint> multiplier;

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

