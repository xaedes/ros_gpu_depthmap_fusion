#pragma once
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeRollbufferCountDiscardedSeqs : public ComputeProgram
{
public:
    virtual ~ComputeRollbufferCountDiscardedSeqs() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_in_sequences, 
        uint bufbind_scratchpad)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "rollbuffer_count_discarded_seqs.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_SEQUENCES",  std::to_string(bufbind_in_sequences)},
            {"BUFBIND_SCRATCHPAD", std::to_string(bufbind_scratchpad)}
        });
        num_items.init(m_glProgram, "num_items");
        min_timestamp_sec.init(m_glProgram, "min_timestamp_sec");
        min_timestamp_nsec.init(m_glProgram, "min_timestamp_nsec");
        in_seqs_offset.init(m_glProgram, "in_seqs_offset");
        scratchpad_offset.init(m_glProgram, "scratchpad_offset");
    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> min_timestamp_sec;
    ProgramUniform<uint> min_timestamp_nsec;
    ProgramUniform<uint> in_seqs_offset;
    ProgramUniform<uint> scratchpad_offset;

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

