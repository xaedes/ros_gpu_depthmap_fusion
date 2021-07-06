#pragma once
#include "gpu_depthmap_fusion/compute_program.h"
#include "gpu_depthmap_fusion/program_uniform.h"


class ComputeRollbufferTransferSelectedTransforms : public ComputeProgram
{
public:
    virtual ~ComputeRollbufferTransferSelectedTransforms() {}
    virtual void init(const std::string& shader_path, 
        uint bufbind_in_sequences, 
        uint bufbind_out_tfs_world, 
        uint bufbind_out_tfs_crop)
    {
        m_groupsize_x = 1024;
        m_groupsize_y = 1;
        m_groupsize_z = 1;        
        ComputeProgram::init(shader_path + "rollbuffer_transfer_selected_transforms.glsl", {
            {"GROUPSIZE_X", std::to_string(m_groupsize_x)},
            {"GROUPSIZE_Y", std::to_string(m_groupsize_y)},
            {"GROUPSIZE_Z", std::to_string(m_groupsize_z)},
            {"BUFBIND_IN_SEQUENCES",  std::to_string(bufbind_in_sequences)},
            {"BUFBIND_OUT_TFS_WORLD", std::to_string(bufbind_out_tfs_world)},
            {"BUFBIND_OUT_TFS_CROP",  std::to_string(bufbind_out_tfs_crop)}
        });
        num_items.init(m_glProgram, "num_items");
        in_seqs_offset.init(m_glProgram, "in_seqs_offset");
        out_tfs_world_offset.init(m_glProgram, "out_tfs_world_offset");
        out_tfs_crop_offset.init(m_glProgram, "out_tfs_crop_offset");
        tf_world_move.init(m_glProgram, "tf_world_move");
        tf_crop_move.init(m_glProgram, "tf_crop_move");
    }

    ProgramUniform<uint> num_items;
    ProgramUniform<uint> in_seqs_offset;
    ProgramUniform<uint> out_tfs_world_offset;
    ProgramUniform<uint> out_tfs_crop_offset;
    ProgramUniform<cv::Matx44f> tf_world_move;
    ProgramUniform<cv::Matx44f> tf_crop_move;

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
