#pragma once
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeTransferData : public ComputeProgram
{
public:
    virtual ~ComputeTransferData() {}
    virtual void init(const std::string& shader_path, 
        const std::string& data_type,
        const std::string& optional_data_type_decl, /* can be empty */
        uint bufbind_in_data, 
        uint bufbind_out_data)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "transfer_data.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"DATA_TYPE",        data_type},
            {"DATA_TYPE_DECLARATION", optional_data_type_decl},
            {"BUFBIND_IN_DATA",  std::to_string(bufbind_in_data)},
            {"BUFBIND_OUT_DATA", std::to_string(bufbind_out_data)}
        });
        num_items.init(m_glProgram, "num_items");
        in_offset.init(m_glProgram, "in_offset");
        out_offset.init(m_glProgram, "out_offset");
    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> in_offset;
    ProgramUniform<uint> out_offset;

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