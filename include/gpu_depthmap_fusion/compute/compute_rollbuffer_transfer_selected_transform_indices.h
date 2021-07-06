#pragma once
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeRollbufferTransferSelectedTransformIndices : public ComputeProgram
{
public:
    virtual ~ComputeRollbufferTransferSelectedTransformIndices() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_in_seq_idcs, 
        uint bufbind_out_tf_idcs)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "rollbuffer_transfer_selected_transform_indices.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_SEQ_IDCS", std::to_string(bufbind_in_seq_idcs)},
            {"BUFBIND_OUT_TF_IDCS", std::to_string(bufbind_out_tf_idcs)}
        });
        num_items.init(m_glProgram, "num_items");
        in_seq_idcs_offset.init(m_glProgram, "in_seq_idcs_offset");
        out_tf_idcs_offset.init(m_glProgram, "out_tf_idcs_offset");
    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> in_seq_idcs_offset;
    ProgramUniform<uint> out_tf_idcs_offset;

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
