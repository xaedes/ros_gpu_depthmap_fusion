#pragma once
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeRollbufferRemainingPointsCopyAndUpdate : public ComputeProgram
{
public:
    virtual ~ComputeRollbufferRemainingPointsCopyAndUpdate() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_in_indices,
        uint bufbind_in_points,
        uint bufbind_in_seq_idcs,
        uint bufbind_in_mask,
        uint bufbind_out_points,
        uint bufbind_out_seq_idcs,
        uint bufbind_out_mask)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "rollbuffer_remaining_points_copy_and_update.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_INDICES",   std::to_string(bufbind_in_indices)},
            {"BUFBIND_IN_POINTS",    std::to_string(bufbind_in_points)},
            {"BUFBIND_IN_SEQ_IDCS",  std::to_string(bufbind_in_seq_idcs)},
            {"BUFBIND_IN_MASK",      std::to_string(bufbind_in_mask)},
            {"BUFBIND_OUT_POINTS",   std::to_string(bufbind_out_points)},
            {"BUFBIND_OUT_SEQ_IDCS", std::to_string(bufbind_out_seq_idcs)},
            {"BUFBIND_OUT_MASK",     std::to_string(bufbind_out_mask)}
        });
        num_items.init(m_glProgram, "num_items");
        num_discarded_seqs.init(m_glProgram, "num_discarded_seqs");
        in_remaining_idcs_offset.init(m_glProgram, "in_remaining_idcs_offset");
        in_points_offset.init(m_glProgram, "in_points_offset");
        in_seq_idcs_offset.init(m_glProgram, "in_seq_idcs_offset");
        in_mask_offset.init(m_glProgram, "in_mask_offset");
        out_points_offset.init(m_glProgram, "out_points_offset");
        out_seq_idcs_offset.init(m_glProgram, "out_seq_idcs_offset");
        out_mask_offset.init(m_glProgram, "out_mask_offset");

    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> num_discarded_seqs;
    ProgramUniform<uint> in_remaining_idcs_offset;
    ProgramUniform<uint> in_points_offset;
    ProgramUniform<uint> in_seq_idcs_offset;
    ProgramUniform<uint> in_mask_offset;
    ProgramUniform<uint> out_points_offset;
    ProgramUniform<uint> out_seq_idcs_offset;
    ProgramUniform<uint> out_mask_offset;

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
