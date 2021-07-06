#pragma once
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeTransferMaskedData : public ComputeProgram
{
public:
    virtual ~ComputeTransferMaskedData() {}
    virtual void init(const std::string& shader_path, 
        const std::string& data_type,
        uint bufbind_in_data, 
        uint bufbind_out_data, 
        uint bufbind_mask, 
        uint bufbind_scratchpad)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "transfer_masked_data.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"DATA_TYPE",          data_type},
            {"BUFBIND_IN_DATA",    std::to_string(bufbind_in_data)},
            {"BUFBIND_OUT_DATA",   std::to_string(bufbind_out_data)},
            {"BUFBIND_IN_MASK",       std::to_string(bufbind_mask)},
            {"BUFBIND_SCRATCHPAD", std::to_string(bufbind_scratchpad)}
        });
        num_items.init(m_glProgram, "num_items");
        in_data_offset.init(m_glProgram, "in_data_offset");
        out_data_offset.init(m_glProgram, "out_data_offset");
        in_mask_offset.init(m_glProgram, "in_mask_offset");
        scratchpad_offset.init(m_glProgram, "scratchpad_offset");
        num_transfer_per_mask_item.init(m_glProgram, "num_transfer_per_mask_item");
    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> in_data_offset;
    ProgramUniform<uint> out_data_offset;
    ProgramUniform<uint> in_mask_offset;
    ProgramUniform<uint> scratchpad_offset;
    ProgramUniform<uint> num_transfer_per_mask_item;

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