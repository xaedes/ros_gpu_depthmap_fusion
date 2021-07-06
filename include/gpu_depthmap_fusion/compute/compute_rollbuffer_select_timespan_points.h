#pragma once
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeRollbufferSelectTimespanPoints : public ComputeProgram
{
public:
    virtual ~ComputeRollbufferSelectTimespanPoints() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_selected, 
        uint bufbind_in_seq_idcs, 
        uint bufbind_in_sequences)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "rollbuffer_select_timespan_points.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_SELECTED",     std::to_string(bufbind_selected)},
            {"BUFBIND_IN_SEQ_IDCS",  std::to_string(bufbind_in_seq_idcs)},
            {"BUFBIND_IN_SEQUENCES", std::to_string(bufbind_in_sequences)}
        });
        num_items.init(m_glProgram, "num_items");
        min_timestamp_sec.init(m_glProgram, "min_timestamp_sec");
        min_timestamp_nsec.init(m_glProgram, "min_timestamp_nsec");
        max_timestamp_sec.init(m_glProgram, "max_timestamp_sec");
        max_timestamp_nsec.init(m_glProgram, "max_timestamp_nsec");
        in_seq_idcs_offset.init(m_glProgram, "in_seq_idcs_offset");
        in_sequences_offset.init(m_glProgram, "in_sequences_offset");
        selection_offset.init(m_glProgram, "selection_offset");
    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> min_timestamp_sec;
    ProgramUniform<uint> min_timestamp_nsec;
    ProgramUniform<uint> max_timestamp_sec;
    ProgramUniform<uint> max_timestamp_nsec;
    ProgramUniform<uint> in_seq_idcs_offset;
    ProgramUniform<uint> in_sequences_offset;
    ProgramUniform<uint> selection_offset;

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
