#pragma once
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeTEMPLATE : public ComputeProgram
{
public:
    virtual ~ComputeTEMPLATE() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_a, 
        uint bufbind_b)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "template.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_A", std::to_string(bufbind_a)},
            {"BUFBIND_B", std::to_string(bufbind_b)}
        });
        num_items.init(m_glProgram, "num_items");
    }

    ProgramUniform<uint> num_items;

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
