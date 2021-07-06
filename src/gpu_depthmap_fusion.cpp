#include "gpu_depthmap_fusion/gpu_depthmap_fusion.h"


#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cv_bridge/cv_bridge.h>

#include <ros/ros.h>

#include <chrono>

#include "gpu_depthmap_fusion/voxelize.h"
#include <sensor_msgs/point_cloud2_iterator.h>

GPUDepthmapFusion::GPUDepthmapFusion()
    : m_glWindow(0)
{
    m_pointSequencesCollect = &m_pointSequencesA;
    m_pointSequencesUpload = &m_pointSequencesB;
}

GPUDepthmapFusion::~GPUDepthmapFusion()
{
    cleanupGL();
    endGLFWEventLoop();
}

void GPUDepthmapFusion::init(
    const std::string& shaderPath)
{
    m_shaderPath = shaderPath;
    initGL();

    m_bufDepthPairs.init(GL_DYNAMIC_DRAW,4);
    m_bufMaskA.init(GL_DYNAMIC_DRAW,4);
    m_bufMaskB.init(GL_DYNAMIC_DRAW,4);

    m_bufPointsA.init(GL_DYNAMIC_DRAW,4);
    m_bufPointsB.init(GL_DYNAMIC_DRAW,4);
    m_bufPointsC.init(GL_DYNAMIC_DRAW,4);

    m_bufNewPointSequencesPoints.init(GL_DYNAMIC_DRAW,4);
    m_bufNewPointSequencesPointsMaskA.init(GL_DYNAMIC_DRAW,4);
    m_bufNewPointSequencesPointsMaskB.init(GL_DYNAMIC_DRAW,4);
    m_bufNewPointSequences.init(GL_DYNAMIC_DRAW,4);
    m_bufRollbufferScratchpad.init(GL_DYNAMIC_DRAW,4);
    m_bufHistoricPointSequencePointsA.init(GL_DYNAMIC_DRAW,4);
    m_bufHistoricPointSequencePointsB.init(GL_DYNAMIC_DRAW,4);
    m_bufHistoricPointSequencePointsMaskA.init(GL_DYNAMIC_DRAW,4);
    m_bufHistoricPointSequencePointsMaskB.init(GL_DYNAMIC_DRAW,4);
    m_bufHistoricPointSequenceSeqIdcsA.init(GL_DYNAMIC_DRAW,4);
    m_bufHistoricPointSequenceSeqIdcsB.init(GL_DYNAMIC_DRAW,4);
    m_bufHistoricPointSequencesA.init(GL_DYNAMIC_DRAW,4);
    m_bufHistoricPointSequencesB.init(GL_DYNAMIC_DRAW,4);
    m_bufHistoricPointSequenceRemainingPointIndices.init(GL_DYNAMIC_DRAW,4);
    m_bufHistoricPointSequenceRemainingSeqIndices.init(GL_DYNAMIC_DRAW,4);
    m_bufRollbufferSelected.init(GL_DYNAMIC_DRAW,4);
    m_bufRollbufferSelectedTransformIdcs.init(GL_DYNAMIC_DRAW,4);
    m_bufRollbufferSelectedTransformsWorld.init(GL_DYNAMIC_DRAW,4);
    m_bufRollbufferSelectedTransformsCrop.init(GL_DYNAMIC_DRAW,4);

    m_bufRectifyMap.init(GL_DYNAMIC_DRAW,4);
    m_bufScratchpad.init(GL_DYNAMIC_DRAW,1);
    m_bufVoxelCoords.init(GL_DYNAMIC_DRAW,4);
    m_bufVoxelOccupancyA.init(GL_DYNAMIC_DRAW,4);
    m_bufVoxelOccupancyB.init(GL_DYNAMIC_DRAW,4);

    m_bufHistoricVoxelOccupancyA.init(GL_DYNAMIC_DRAW,4);
    m_bufHistoricVoxelOccupancyB.init(GL_DYNAMIC_DRAW,4);

    m_bufCCLabeledLayersA.init(GL_DYNAMIC_DRAW,4);
    m_bufCCLabeledLayersB.init(GL_DYNAMIC_DRAW,4);
    m_bufCCNumLabelsPerLayer.init(GL_DYNAMIC_DRAW,4);
    m_bufCCStats.init(GL_DYNAMIC_DRAW,4);
    m_bufCCCentroids.init(GL_DYNAMIC_DRAW,4);
    m_bufCCStatsDataStarts.init(GL_DYNAMIC_DRAW,4);
    m_bufCCCentroidsDataStarts.init(GL_DYNAMIC_DRAW,4);
    m_bufCCLayersConnectionsDataStarts.init(GL_DYNAMIC_DRAW,4);
    m_bufCCLayersConnectionsDataA.init(GL_DYNAMIC_DRAW,4);
    m_bufCCLayersConnectionsDataB.init(GL_DYNAMIC_DRAW,4);

    std::cout << "m_bufDepthPairs                       " << m_bufDepthPairs.m_glBuf << std::endl;
    std::cout << "m_bufMaskA                            " << m_bufMaskA.m_glBuf << std::endl;
    std::cout << "m_bufMaskB                            " << m_bufMaskB.m_glBuf << std::endl;
    std::cout << "m_bufPointsA                          " << m_bufPointsA.m_glBuf << std::endl;
    std::cout << "m_bufPointsB                          " << m_bufPointsB.m_glBuf << std::endl;
    std::cout << "m_bufPointsC                          " << m_bufPointsC.m_glBuf << std::endl;

    std::cout << "m_bufNewPointSequencesPoints          " << m_bufNewPointSequencesPoints.m_glBuf << std::endl;
    std::cout << "m_bufNewPointSequencesPointsMaskA     " << m_bufNewPointSequencesPointsMaskA.m_glBuf << std::endl;
    std::cout << "m_bufNewPointSequencesPointsMaskB     " << m_bufNewPointSequencesPointsMaskB.m_glBuf << std::endl;
    std::cout << "m_bufNewPointSequences                " << m_bufNewPointSequences.m_glBuf << std::endl;
    std::cout << "m_bufRollbufferScratchpad             " << m_bufRollbufferScratchpad.m_glBuf << std::endl;
    std::cout << "m_bufHistoricPointSequencePointsA     " << m_bufHistoricPointSequencePointsA.m_glBuf << std::endl;
    std::cout << "m_bufHistoricPointSequencePointsB     " << m_bufHistoricPointSequencePointsB.m_glBuf << std::endl;
    std::cout << "m_bufHistoricPointSequencePointsMaskA " << m_bufHistoricPointSequencePointsMaskA.m_glBuf << std::endl;
    std::cout << "m_bufHistoricPointSequencePointsMaskB " << m_bufHistoricPointSequencePointsMaskB.m_glBuf << std::endl;
    std::cout << "m_bufHistoricPointSequenceSeqIdcsA    " << m_bufHistoricPointSequenceSeqIdcsA.m_glBuf << std::endl;
    std::cout << "m_bufHistoricPointSequenceSeqIdcsB    " << m_bufHistoricPointSequenceSeqIdcsB.m_glBuf << std::endl;
    std::cout << "m_bufHistoricPointSequencesA          " << m_bufHistoricPointSequencesA.m_glBuf << std::endl;
    std::cout << "m_bufHistoricPointSequencesB          " << m_bufHistoricPointSequencesB.m_glBuf << std::endl;
    std::cout << "m_bufHistoricPointSequenceRemainingPointIndices " << m_bufHistoricPointSequenceRemainingPointIndices.m_glBuf << std::endl;
    std::cout << "m_bufHistoricPointSequenceRemainingSeqIndices " << m_bufHistoricPointSequenceRemainingSeqIndices.m_glBuf << std::endl;
    std::cout << "m_bufRollbufferSelected                " << m_bufRollbufferSelected.m_glBuf << std::endl;
    std::cout << "m_bufRollbufferSelectedTransformIdcs   " << m_bufRollbufferSelectedTransformIdcs.m_glBuf << std::endl;
    std::cout << "m_bufRollbufferSelectedTransformsWorld " << m_bufRollbufferSelectedTransformsWorld.m_glBuf << std::endl;
    std::cout << "m_bufRollbufferSelectedTransformsCrop  " << m_bufRollbufferSelectedTransformsCrop.m_glBuf << std::endl;

    std::cout << "m_bufRectifyMap                       " << m_bufRectifyMap.m_glBuf << std::endl;
    std::cout << "m_bufScratchpad                       " << m_bufScratchpad.m_glBuf << std::endl;
    std::cout << "m_bufVoxelCoords                      " << m_bufVoxelCoords.m_glBuf << std::endl;
    std::cout << "m_bufVoxelOccupancyA                  " << m_bufVoxelOccupancyA.m_glBuf << std::endl;
    std::cout << "m_bufVoxelOccupancyB                  " << m_bufVoxelOccupancyB.m_glBuf << std::endl;
    std::cout << "m_bufHistoricVoxelOccupancyA          " << m_bufHistoricVoxelOccupancyA.m_glBuf << std::endl;
    std::cout << "m_bufHistoricVoxelOccupancyB          " << m_bufHistoricVoxelOccupancyB.m_glBuf << std::endl;
    std::cout << "m_bufCCLabeledLayersA                 " << m_bufCCLabeledLayersA.m_glBuf << std::endl;
    std::cout << "m_bufCCLabeledLayersB                 " << m_bufCCLabeledLayersB.m_glBuf << std::endl;
    std::cout << "m_bufCCNumLabelsPerLayer              " << m_bufCCNumLabelsPerLayer.m_glBuf << std::endl;
    std::cout << "m_bufCCStats                          " << m_bufCCStats.m_glBuf << std::endl;
    std::cout << "m_bufCCCentroids                      " << m_bufCCCentroids.m_glBuf << std::endl;
    std::cout << "m_bufCCStatsDataStarts                " << m_bufCCStatsDataStarts.m_glBuf << std::endl;
    std::cout << "m_bufCCCentroidsDataStarts            " << m_bufCCCentroidsDataStarts.m_glBuf << std::endl;
    std::cout << "m_bufCCLayersConnectionsDataStarts    " << m_bufCCLayersConnectionsDataStarts.m_glBuf << std::endl;
    std::cout << "m_bufCCLayersConnectionsDataA         " << m_bufCCLayersConnectionsDataA.m_glBuf << std::endl;
    std::cout << "m_bufCCLayersConnectionsDataB         " << m_bufCCLayersConnectionsDataB.m_glBuf << std::endl;


    m_bufBindings.init();
    std::vector<BufferIndexBindings::BufferIndexBinding> bindings;
    /////////////////////////////////
    // depth map processing
    /////////////////////////////////
    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufDepthPairs.glBuf(),
        m_bufMaskA.glBuf(),
        m_bufPointsA.glBuf(), // points in camera coord
        m_bufPointsB.glBuf(), // points in world coord
        m_bufPointsC.glBuf(), // points in crop coord
        m_bufRectifyMap.glBuf()
    });
    m_computeConvertDepthmapToPoints.init(m_shaderPath, 
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex,
        bindings[3].bindingIndex,
        bindings[4].bindingIndex,
        bindings[5].bindingIndex
    );
    m_bufBindings.add(m_computeConvertDepthmapToPoints.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufMaskA.glBuf(),
        m_bufMaskB.glBuf(),
        m_bufPointsA.glBuf()
    });
    m_computeFilterFlyingPixels.init(m_shaderPath, 
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex
    );    
    m_bufBindings.add(m_computeFilterFlyingPixels.getGLProgram(), bindings);

    /////////////////////////////////
    // new point sequence processing
    /////////////////////////////////
    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufNewPointSequencesPointsMaskA.glBuf()
    });
    m_computeInitNewPointSequencesPointsMask.init(m_shaderPath,
        bindings[0].bindingIndex
    );
    m_bufBindings.add(m_computeInitNewPointSequencesPointsMask.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufNewPointSequencesPointsMaskA.glBuf(),
        m_bufNewPointSequencesPointsMaskB.glBuf(),
        m_bufNewPointSequencesPoints.glBuf()
    });
    m_computeFilterNewPointSequencePoints.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex
    );
    m_bufBindings.add(m_computeFilterNewPointSequencePoints.getGLProgram(), bindings);
    
    
    /////////////////////////////////
    // new point sequence inserting
    /////////////////////////////////
    const std::string strGLSLPointSequenceDecl = 
        "struct PointSequence       \n"
        "{                          \n"
        "    uint timestampSec;     \n"
        "    uint timestampNSec;    \n"
        "    uint start;            \n"
        "    uint numPoints;        \n"
        "    uint padding[12];      \n"
        "    mat4 transform_move;   \n"
        "};                         \n";

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequencePointsB.glBuf(),
        m_bufHistoricPointSequencePointsA.glBuf()
    });
    m_computeRollbufferTransferOldPoints.init(m_shaderPath,
        "vec4", "",
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferTransferOldPoints.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequencePointsMaskB.glBuf(),
        m_bufHistoricPointSequencePointsMaskA.glBuf()
    });
    m_computeRollbufferTransferOldPointsMask.init(m_shaderPath,
        "uint", "",
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferTransferOldPointsMask.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequencesB.glBuf(),
        m_bufHistoricPointSequencesA.glBuf()
    });
    m_computeRollbufferTransferOldSeqs.init(m_shaderPath,
        "PointSequence",
        strGLSLPointSequenceDecl,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferTransferOldSeqs.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequenceSeqIdcsB.glBuf(),
        m_bufHistoricPointSequenceSeqIdcsA.glBuf()
    });
    m_computeRollbufferTransferOldSeqIdcs.init(m_shaderPath,
        "uint", "",
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferTransferOldSeqIdcs.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufNewPointSequencesPoints.glBuf(),
        m_bufHistoricPointSequencePointsA.glBuf()
    });
    m_computeRollbufferTransferNewPoints.init(m_shaderPath,
        "vec4", "",
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferTransferNewPoints.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufNewPointSequencesPointsMaskB.glBuf(),
        m_bufHistoricPointSequencePointsMaskA.glBuf()
    });
    m_computeRollbufferTransferNewPointsMask.init(m_shaderPath,
        "uint", "",
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferTransferNewPointsMask.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufNewPointSequences.glBuf(),
        m_bufHistoricPointSequencesA.glBuf()
    });
    m_computeRollbufferTransferNewSeqs.init(m_shaderPath,
        "PointSequence",
        strGLSLPointSequenceDecl,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferTransferNewSeqs.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequenceSeqIdcsA.glBuf()
    });
    m_computeRollbufferSetNewPointsSeqIdcs.init(m_shaderPath,
        bindings[0].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferSetNewPointsSeqIdcs.getGLProgram(), bindings);

    ////////////////////////////////////////////
    // historic point sequence buffer rolling
    ////////////////////////////////////////////
    // two variants for the compute programs operating on swap buffers
    // they wired so we can append to buf[toggle], that is ((toggle==0) ? bufA : bufB)
    // m_computeRollbufferCountDiscarded
    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequenceSeqIdcsA.glBuf(),
        m_bufHistoricPointSequencesA.glBuf(),
        m_bufRollbufferScratchpad.glBuf()
    });
    m_computeRollbufferCountDiscardedPoints.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferCountDiscardedPoints.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequencesB.glBuf(),
        m_bufRollbufferScratchpad.glBuf()
    });
    m_computeRollbufferCountDiscardedSeqs.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferCountDiscardedSeqs.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequenceRemainingPointIndices.glBuf()
    });
    m_computeRollbufferRemainingPointsIndices.init(m_shaderPath,
        bindings[0].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferRemainingPointsIndices.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequenceRemainingPointIndices.glBuf(),
        m_bufHistoricPointSequencePointsA.glBuf(),
        m_bufHistoricPointSequenceSeqIdcsA.glBuf(),
        m_bufHistoricPointSequencePointsMaskA.glBuf(),
        m_bufHistoricPointSequencePointsB.glBuf(),
        m_bufHistoricPointSequenceSeqIdcsB.glBuf(),
        m_bufHistoricPointSequencePointsMaskB.glBuf()
    });
    m_computeRollbufferRemainingPointsCopyAndUpdate.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex,
        bindings[3].bindingIndex,
        bindings[4].bindingIndex,
        bindings[5].bindingIndex,
        bindings[6].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferRemainingPointsCopyAndUpdate.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequenceRemainingSeqIndices.glBuf()
    });
    m_computeRollbufferRemainingSeqsIndices.init(m_shaderPath,
        bindings[0].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferRemainingSeqsIndices.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequenceRemainingSeqIndices.glBuf(),
        m_bufHistoricPointSequencesA.glBuf(),
        m_bufHistoricPointSequencesB.glBuf()
    });
    m_computeRollbufferRemainingSeqsCopy.init(m_shaderPath,
        "PointSequence",
        strGLSLPointSequenceDecl,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferRemainingSeqsCopy.getGLProgram(), bindings);


    ////////////////////////////////////////////////////////
    // select point sequence timespan from historic data
    ////////////////////////////////////////////////////////
    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufRollbufferSelected.glBuf(),
        m_bufHistoricPointSequenceSeqIdcsB.glBuf(),
        m_bufHistoricPointSequencesB.glBuf()
    });
    m_computeRollbufferSelectTimespanPoints.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferSelectTimespanPoints.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufRollbufferSelected.glBuf(),
        m_bufHistoricPointSequencesB.glBuf()
    });
    m_computeRollbufferSelectTimespanSequences.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferSelectTimespanSequences.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequencePointsMaskB.glBuf(),
        m_bufMaskB.glBuf()
    });
    m_computeRollbufferTransferSelectedMask.init(m_shaderPath,
        "uint", "",
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferTransferSelectedMask.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequenceSeqIdcsB.glBuf(),
        m_bufRollbufferSelectedTransformIdcs.glBuf()
    });
    m_computeRollbufferTransferSelectedTransformIndices.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferTransferSelectedTransformIndices.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricPointSequencesB.glBuf(),
        m_bufRollbufferSelectedTransformsWorld.glBuf(),
        m_bufRollbufferSelectedTransformsCrop.glBuf()
    });
    m_computeRollbufferTransferSelectedTransforms.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex
    );
    m_bufBindings.add(m_computeRollbufferTransferSelectedTransforms.getGLProgram(), bindings);

    ///////////////////////////////////////
    // transform selected point sequence 
    ///////////////////////////////////////
    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufRollbufferSelectedTransformsWorld.glBuf(),
        m_bufRollbufferSelectedTransformIdcs.glBuf(),
        m_bufHistoricPointSequencePointsB.glBuf(), // points in camera coord
        m_bufHistoricPointSequencePointsMaskB.glBuf(),
        m_bufPointsB.glBuf()  // points in world coord
    });
    m_computeTransformPointSequenceWorld.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex,
        bindings[3].bindingIndex,
        bindings[4].bindingIndex
    );
    m_bufBindings.add(m_computeTransformPointSequenceWorld.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufRollbufferSelectedTransformsCrop.glBuf(),
        m_bufRollbufferSelectedTransformIdcs.glBuf(),
        m_bufHistoricPointSequencePointsB.glBuf(), // points in camera coord
        m_bufHistoricPointSequencePointsMaskB.glBuf(),
        m_bufPointsC.glBuf()  // points in crop coord
    });
    m_computeTransformPointSequenceCrop.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex,
        bindings[3].bindingIndex,
        bindings[4].bindingIndex
    );
    m_bufBindings.add(m_computeTransformPointSequenceCrop.getGLProgram(), bindings);

    //////////////////////////////////////
    // crop all points 
    //////////////////////////////////////
    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufMaskB.glBuf(),
        m_bufMaskA.glBuf(),
        m_bufPointsC.glBuf()
    });
    m_computeCropPoints.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex
    );
    m_bufBindings.add(m_computeCropPoints.getGLProgram(), bindings);

    //////////////////////////////////////
    // apply mask to all points 
    //////////////////////////////////////
    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufPointsB.glBuf(), // input points in world coord
        m_bufPointsA.glBuf(), // output points in world coord
        m_bufMaskA.glBuf(), 
        m_bufScratchpad.glBuf()
    });
    m_computeApplyPointMask.init(m_shaderPath, 
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex,
        bindings[3].bindingIndex
    );
    m_bufBindings.add(m_computeApplyPointMask.getGLProgram(), bindings);


    //////////////////////////////////////
    // voxel coordinates
    //////////////////////////////////////
    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufPointsA.glBuf(), 
        m_bufVoxelCoords.glBuf()
    });
    m_computeVoxelCoords.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeVoxelCoords.getGLProgram(), bindings);

    //////////////////////////////////////
    // voxel grid occupancy
    //////////////////////////////////////
    
    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufVoxelOccupancyA.glBuf()
    });
    m_computeVoxelGridOccupancyClear.init(m_shaderPath,
        bindings[0].bindingIndex
    );
    m_bufBindings.add(m_computeVoxelGridOccupancyClear.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufVoxelCoords.glBuf(),
        m_bufVoxelOccupancyA.glBuf()
    });
    m_computeVoxelGridOccupancyOfPoints.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeVoxelGridOccupancyOfPoints.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricVoxelOccupancyA.glBuf()
    });
    m_computeHistoricVoxelGridOccupancyClear.init(m_shaderPath,
        bindings[0].bindingIndex
    );
    m_bufBindings.add(m_computeHistoricVoxelGridOccupancyClear.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricVoxelOccupancyA.glBuf(),
        m_bufHistoricVoxelOccupancyB.glBuf()
    });
    m_computeHistoricVoxelGridOccupancyDec.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeHistoricVoxelGridOccupancyDec.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricVoxelOccupancyB.glBuf(),
        m_bufVoxelOccupancyA.glBuf(),
        m_bufHistoricVoxelOccupancyA.glBuf()
    });
    m_computeHistoricVoxelGridOccupancySet.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex
    );
    m_bufBindings.add(m_computeHistoricVoxelGridOccupancySet.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufHistoricVoxelOccupancyA.glBuf(),
        m_bufVoxelOccupancyB.glBuf()
    });
    m_computeVoxelGridOccupancyToChars.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeVoxelGridOccupancyToChars.getGLProgram(), bindings);


    //////////////////////////////////////
    // connect layers
    //////////////////////////////////////
    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufCCLayersConnectionsDataA.glBuf()
    });
    m_computeClearLayersConnections.init(m_shaderPath,
        bindings[0].bindingIndex
    );
    m_bufBindings.add(m_computeClearLayersConnections.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufCCLabeledLayersA.glBuf(),
        m_bufCCLabeledLayersB.glBuf()
    });
    m_computeLabelWordsToUints.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeLabelWordsToUints.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufCCLabeledLayersB.glBuf(),
        m_bufCCNumLabelsPerLayer.glBuf(),
        m_bufCCLayersConnectionsDataStarts.glBuf(),
        m_bufCCLayersConnectionsDataA.glBuf()
    });
    m_computeLayersConnections.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex,
        bindings[2].bindingIndex,
        bindings[3].bindingIndex
    );
    m_bufBindings.add(m_computeLayersConnections.getGLProgram(), bindings);

    //------------------------------------
    bindings = m_bufBindings.make({
        m_bufCCLayersConnectionsDataA.glBuf(),
        m_bufCCLayersConnectionsDataB.glBuf()
    });
    m_computeLayersConnectionsToChars.init(m_shaderPath,
        bindings[0].bindingIndex,
        bindings[1].bindingIndex
    );
    m_bufBindings.add(m_computeLayersConnectionsToChars.getGLProgram(), bindings);
    //////////////////////////////////////


    m_measureTime.gain = 0.1;
    m_measureTime.sections = std::vector<std::string>{
        "begin",
        "upload",
        "compute",
        "download"
    };

    m_mappedPoints = nullptr;

    m_voxelRadixGrouper = VoxelGrouper(m_voxelGroupSize);

    m_rollBufferNumPoints              = 0;
    m_rollBufferNumSeqs                = 0;
    m_rollBufferSelectionPointStart    = 0;
    m_rollBufferSelectionPointCount    = 0;
    m_rollBufferSelectionSequenceStart = 0;
    m_rollBufferSelectionSequenceCount = 0;
    m_rollBufferEarliestTimeSec        = 0;
    m_rollBufferEarliestTimeNSec       = 0;
    m_rollBufferLastTimeSec            = 0;
    m_rollBufferLastTimeNSec           = 0;
    // startGLFWEventLoop();
}

void MessageCallback( GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam )
{
    fprintf( 
        stderr, 
        "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
        ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
        type, severity, message 
    );

}

void GPUDepthmapFusion::initGL()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    m_glWindow = glfwCreateWindow(32, 32, "gpu_depthmap_fusion", nullptr, nullptr);
    glfwMakeContextCurrent(m_glWindow);
    glfwSwapInterval(1);
    gladLoadGL();
    glEnable(GL_DEBUG_OUTPUT);


    // During init, enable debug output
    glEnable              ( GL_DEBUG_OUTPUT );
    glDebugMessageCallback( MessageCallback, 0 );

    // m_bufDepthmap.init(GL_DYNAMIC_DRAW);
    // m_bufPoints.init(GL_DYNAMIC_READ);
    // m_bufDepthmap.bufferBase(0);
    // m_bufPoints.bufferBase(1);
    // m_compute.init(m_shaderPath);
}
void GPUDepthmapFusion::cleanupGL()
{
    glfwDestroyWindow(m_glWindow);
    glfwTerminate();
}

void GPUDepthmapFusion::clear()
{
    m_rollBufferSelectionPointCount = 0;
    m_rollBufferSelectionSequenceCount = 0;
    m_depthmapsTotalElements = 0;
    m_depthmaps.clear();
    // swapPointSequencesBuffers();
}

void GPUDepthmapFusion::swapPointSequencesBuffers()
{
    if (m_pointSequencesCollect == &m_pointSequencesA)
    {
        m_pointSequencesUpload = &m_pointSequencesA;
        m_pointSequencesCollect = &m_pointSequencesB;
    }
    else
    {
        m_pointSequencesCollect = &m_pointSequencesA;
        m_pointSequencesUpload = &m_pointSequencesB;
    }
}
void GPUDepthmapFusion::addPointSequence(
    const sensor_msgs::PointCloud2& pointcloud, 
    uint32_t timestampSec,
    uint32_t timestampNSec,
    cv::Matx44f transform_move)
{
    assert(m_pointSequencesCollect != nullptr);
    
    int numPoints = pointcloud.width * pointcloud.height;
    PointSequence pts;
    pts.timestampSec    = timestampSec;
    pts.timestampNSec   = timestampNSec;
    pts.start           = m_pointSequencesCollect->num_points_total;
    pts.numPoints       = numPoints;
    pts.transform_move = transform_move;
    m_pointSequencesCollect->num_points_total += numPoints;
    // sensor_msgs::PointCloud2& nonconst_pointcloud = const_cast<sensor_msgs::PointCloud2&>(pointcloud);
    // sensor_msgs::PointCloud2Iterator<float> iter_x(nonconst_pointcloud, "x");
    // sensor_msgs::PointCloud2Iterator<float> iter_y(nonconst_pointcloud, "y");
    // sensor_msgs::PointCloud2Iterator<float> iter_z(nonconst_pointcloud, "z");
    // uint8_t* x = reinterpret_cast<uint8_t*>(&(*iter_x));
    // uint8_t* y = reinterpret_cast<uint8_t*>(&(*iter_y));
    // uint8_t* z = reinterpret_cast<uint8_t*>(&(*iter_z));
    int point_step = pointcloud.point_step;
    auto& points_buffer = m_pointSequencesCollect->pointData;
    // points_buffer.resize(m_pointSequencesCollect->num_points_total*3);
    points_buffer.resize(m_pointSequencesCollect->num_points_total*4);
    // #pragma omp parallel for 
    for (int k=0; k<std::max(0,numPoints); ++k)
    {
        points_buffer[4*(pts.start+k)+0] = *reinterpret_cast<const float*>(&pointcloud.data[k*point_step+0*sizeof(float)]);
        points_buffer[4*(pts.start+k)+1] = *reinterpret_cast<const float*>(&pointcloud.data[k*point_step+1*sizeof(float)]);
        points_buffer[4*(pts.start+k)+2] = *reinterpret_cast<const float*>(&pointcloud.data[k*point_step+2*sizeof(float)]);
        points_buffer[4*(pts.start+k)+3] = 1;

        // points_buffer[3*(pts.start+k)+0] = *reinterpret_cast<const float*>(&pointcloud.data[k*point_step+0*sizeof(float)]);
        // points_buffer[3*(pts.start+k)+1] = *reinterpret_cast<const float*>(&pointcloud.data[k*point_step+1*sizeof(float)]);
        // points_buffer[3*(pts.start+k)+2] = *reinterpret_cast<const float*>(&pointcloud.data[k*point_step+2*sizeof(float)]);

        // points_buffer[3*(pts.start+k)+0] = *(iter_x+k);
        // points_buffer[3*(pts.start+k)+1] = *(iter_y+k);
        // points_buffer[3*(pts.start+k)+2] = *(iter_z+k);

        // points_buffer[3*(pts.start+k)+0] = *reinterpret_cast<float*>(x+point_step);
        // points_buffer[3*(pts.start+k)+1] = *reinterpret_cast<float*>(y+point_step);
        // points_buffer[3*(pts.start+k)+2] = *reinterpret_cast<float*>(z+point_step);
    }
    m_pointSequencesCollect->sequences.push_back(pts);

}

void GPUDepthmapFusion::addDepthmap(
    const cv::Mat_<uint16_t>& depthmap,
    float depthScale,
    float fx, float fy,
    float cx, float cy,
    cv::Matx44f transform_world,
    cv::Matx44f transform_crop
)
{
    m_depthmaps.push_back({
        depthmap.data, (uint)depthmap.total(), 
        depthmap.cols, depthmap.rows, 
        depthScale, 
        fx, fy, cx, cy, 
        transform_world,
        transform_crop
    });
    m_depthmapsTotalElements += depthmap.total();
}


void GPUDepthmapFusion::uploadPointSequences()
{
    swapPointSequencesBuffers();
    m_pointSequencesCollect->clear(); 

    m_bufNewPointSequencesPoints.bind();
    m_bufNewPointSequencesPoints.resize(m_pointSequencesUpload->num_points_total);
    const auto& sequences = m_pointSequencesUpload->sequences;
    for (int i=0; i<sequences.size(); ++i)
    {
        const auto& item = m_pointSequencesUpload->sequences[i];
        m_bufNewPointSequencesPoints.upload(
            &m_pointSequencesUpload->pointData[item.start*4],
            item.start, item.numPoints);

    }
 
    m_bufNewPointSequences.bind();
    m_bufNewPointSequences.resize(sequences.size());
    // either this (preferable): 
    //* (toggle by adding or removing a leading / in this line)
    m_bufNewPointSequences.upload(
        sequences.data(),
        0, sequences.size());
    /*/
    // or that:
    std::cout << "sizeof(PointSequence) " << sizeof(PointSequence) << std::endl;
    for (int i=0; i<sequences.size(); ++i)
    {
        m_bufNewPointSequences.upload(
            &sequences[i],
            i, 1);
    }
    //*/
    
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // checkAllPointSequenceBuffers();
}

void GPUDepthmapFusion::checkAllPointSequenceBuffers()
{

    static std::vector<glm::vec4>     vecNewPointSequencesPoints;
    static std::vector<PointSequence> vecNewPointSequences;
    static std::vector<uint32_t>      vecNewPointSequencesPointsMaskA;
    static std::vector<uint32_t>      vecNewPointSequencesPointsMaskB;
    static std::vector<glm::vec4>     vecHistoricPointSequencePointsA;
    static std::vector<glm::vec4>     vecHistoricPointSequencePointsB;
    static std::vector<uint32_t>      vecHistoricPointSequencePointsMaskA;
    static std::vector<uint32_t>      vecHistoricPointSequencePointsMaskB;
    static std::vector<uint32_t>      vecHistoricPointSequenceSeqIdcsA;
    static std::vector<uint32_t>      vecHistoricPointSequenceSeqIdcsB;
    static std::vector<uint32_t>      vecHistoricPointSequenceRemainingPointIndices;
    static std::vector<uint32_t>      vecHistoricPointSequenceRemainingSeqIndices;
    static std::vector<PointSequence> vecHistoricPointSequencesA;
    static std::vector<PointSequence> vecHistoricPointSequencesB;
    static std::vector<uint32_t>      vecRollbufferScratchpad;
    static std::vector<uint32_t>      vecRollbufferSelected;
    static std::vector<uint32_t>      vecRollbufferSelectedTransformIdcs;
    static std::vector<glm::mat4>     vecRollbufferSelectedTransformsWorld;
    static std::vector<glm::mat4>     vecRollbufferSelectedTransformsCrop;

    vecNewPointSequencesPoints.resize(m_bufNewPointSequencesPoints.size());
    vecNewPointSequences.resize(m_bufNewPointSequences.size());
    vecNewPointSequencesPointsMaskA.resize(m_bufNewPointSequencesPointsMaskA.size());
    vecNewPointSequencesPointsMaskB.resize(m_bufNewPointSequencesPointsMaskB.size());
    vecHistoricPointSequencePointsA.resize(m_bufHistoricPointSequencePointsA.size());
    vecHistoricPointSequencePointsB.resize(m_bufHistoricPointSequencePointsB.size());
    vecHistoricPointSequencePointsMaskA.resize(m_bufHistoricPointSequencePointsMaskA.size());
    vecHistoricPointSequencePointsMaskB.resize(m_bufHistoricPointSequencePointsMaskB.size());
    vecHistoricPointSequenceSeqIdcsA.resize(m_bufHistoricPointSequenceSeqIdcsA.size());
    vecHistoricPointSequenceSeqIdcsB.resize(m_bufHistoricPointSequenceSeqIdcsB.size());
    vecHistoricPointSequenceRemainingPointIndices.resize(m_bufHistoricPointSequenceRemainingPointIndices.size());
    vecHistoricPointSequenceRemainingSeqIndices.resize(m_bufHistoricPointSequenceRemainingSeqIndices.size());
    vecHistoricPointSequencesA.resize(m_bufHistoricPointSequencesA.size());
    vecHistoricPointSequencesB.resize(m_bufHistoricPointSequencesB.size());
    vecRollbufferScratchpad.resize(m_bufRollbufferScratchpad.size());
    vecRollbufferSelected.resize(m_bufRollbufferSelected.size());
    vecRollbufferSelectedTransformIdcs.resize(m_bufRollbufferSelectedTransformIdcs.size());
    vecRollbufferSelectedTransformsWorld.resize(m_bufRollbufferSelectedTransformsWorld.size());
    vecRollbufferSelectedTransformsCrop.resize(m_bufRollbufferSelectedTransformsCrop.size());


    m_bufNewPointSequencesPoints.bind().download(vecNewPointSequencesPoints.data(), 0 , vecNewPointSequencesPoints.size());
    m_bufNewPointSequences.bind().download(vecNewPointSequences.data(), 0 , vecNewPointSequences.size());
    m_bufNewPointSequencesPointsMaskA.bind().download(vecNewPointSequencesPointsMaskA.data(), 0 , vecNewPointSequencesPointsMaskA.size());
    m_bufNewPointSequencesPointsMaskB.bind().download(vecNewPointSequencesPointsMaskB.data(), 0 , vecNewPointSequencesPointsMaskB.size());
    m_bufHistoricPointSequencePointsA.bind().download(vecHistoricPointSequencePointsA.data(), 0 , vecHistoricPointSequencePointsA.size());
    m_bufHistoricPointSequencePointsB.bind().download(vecHistoricPointSequencePointsB.data(), 0 , vecHistoricPointSequencePointsB.size());
    m_bufHistoricPointSequencePointsMaskA.bind().download(vecHistoricPointSequencePointsMaskA.data(), 0 , vecHistoricPointSequencePointsMaskA.size());
    m_bufHistoricPointSequencePointsMaskB.bind().download(vecHistoricPointSequencePointsMaskB.data(), 0 , vecHistoricPointSequencePointsMaskB.size());
    m_bufHistoricPointSequenceSeqIdcsA.bind().download(vecHistoricPointSequenceSeqIdcsA.data(), 0 , vecHistoricPointSequenceSeqIdcsA.size());
    m_bufHistoricPointSequenceSeqIdcsB.bind().download(vecHistoricPointSequenceSeqIdcsB.data(), 0 , vecHistoricPointSequenceSeqIdcsB.size());
    m_bufHistoricPointSequenceRemainingPointIndices.bind().download(vecHistoricPointSequenceRemainingPointIndices.data(), 0 , vecHistoricPointSequenceRemainingPointIndices.size());
    m_bufHistoricPointSequenceRemainingSeqIndices.bind().download(vecHistoricPointSequenceRemainingSeqIndices.data(), 0 , vecHistoricPointSequenceRemainingSeqIndices.size());
    m_bufHistoricPointSequencesA.bind().download(vecHistoricPointSequencesA.data(), 0 , vecHistoricPointSequencesA.size());
    m_bufHistoricPointSequencesB.bind().download(vecHistoricPointSequencesB.data(), 0 , vecHistoricPointSequencesB.size());
    m_bufRollbufferScratchpad.bind().download(vecRollbufferScratchpad.data(), 0 , vecRollbufferScratchpad.size());
    m_bufRollbufferSelected.bind().download(vecRollbufferSelected.data(), 0 , vecRollbufferSelected.size());
    m_bufRollbufferSelectedTransformIdcs.bind().download(vecRollbufferSelectedTransformIdcs.data(), 0 , vecRollbufferSelectedTransformIdcs.size());
    m_bufRollbufferSelectedTransformsWorld.bind().download(vecRollbufferSelectedTransformsWorld.data(), 0 , vecRollbufferSelectedTransformsWorld.size());
    m_bufRollbufferSelectedTransformsCrop.bind().download(vecRollbufferSelectedTransformsCrop.data(), 0 , vecRollbufferSelectedTransformsCrop.size());

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    int i=0;
}

void GPUDepthmapFusion::filterNewPointSequences(float threshold, uint filter_size)
{
    m_bufNewPointSequencesPointsMaskA.resize(m_pointSequencesUpload->num_points_total);
    m_bufNewPointSequencesPointsMaskB.resize(m_pointSequencesUpload->num_points_total);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    // init other points mask with 1  (i.e. all valid)
    m_bufBindings.bind(m_computeInitNewPointSequencesPointsMask.getGLProgram());
    m_computeInitNewPointSequencesPointsMask.use();
    m_computeInitNewPointSequencesPointsMask.num_items.set(m_pointSequencesUpload->num_points_total);
    m_computeInitNewPointSequencesPointsMask.offset.set(0);
    m_computeInitNewPointSequencesPointsMask.value.set(1);
    m_computeInitNewPointSequencesPointsMask.dispatch();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    //---- if we want to filter all sequence points together
    //*
    m_bufBindings.bind(m_computeFilterNewPointSequencePoints.getGLProgram());
    m_computeFilterNewPointSequencePoints.use();
    m_computeFilterNewPointSequencePoints.num_items.set(m_pointSequencesUpload->num_points_total);
    m_computeFilterNewPointSequencePoints.in_point_offset.set(0);
    m_computeFilterNewPointSequencePoints.in_mask_offset.set(0);
    m_computeFilterNewPointSequencePoints.out_mask_offset.set(0);
    m_computeFilterNewPointSequencePoints.threshold_view_angle.set(threshold);
    m_computeFilterNewPointSequencePoints.filter_size.set(filter_size);
    m_computeFilterNewPointSequencePoints.dispatch();

    //---- if we want to filter each sequence seperate
    /*/
    m_bufBindings.bind(m_computeFilterNewPointSequencePoints.getGLProgram());
    m_computeFilterNewPointSequencePoints.use();
    const auto& sequences = m_pointSequencesUpload->sequences;
    for (int i=0; i<sequences.size(); ++i)
    {

        const auto& sequence = m_pointSequencesUpload->sequences[i];

        m_computeFilterNewPointSequencePoints.num_items.set(sequence.numPoints);
        m_computeFilterNewPointSequencePoints.point_offset.set(sequence.start);
        m_computeFilterNewPointSequencePoints.in_mask_offset.set(0);
        m_computeFilterNewPointSequencePoints.out_mask_offset.set(0);
        m_computeFilterNewPointSequencePoints.dispatch();
    }
    //*/
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // checkAllPointSequenceBuffers();
}


void GPUDepthmapFusion::insertNewPointSequencesInRollbuffer()
{
     
    // transfer valid part of point sequence to historic point sequence buffer
    uint32_t num_points = m_rollBufferNumPoints;
    uint32_t num_seqs = m_rollBufferNumSeqs;
    // two variants for the compute programs operating on swap buffers
    // they wired so we can append to buf[toggle], that is ((toggle==0) ? bufA : bufB)

    const auto& sequences = m_pointSequencesUpload->sequences;

    uint32_t num_new_points = m_pointSequencesUpload->num_points_total;
    uint32_t num_new_seqs = sequences.size();


    // std::cout << "num_new_points " << num_new_points << std::endl;
    // std::cout << "num_new_seqs   " << num_new_seqs << std::endl;    

    // resize target buffers
    m_bufHistoricPointSequencePointsA.resize(num_points + num_new_points);
    m_bufHistoricPointSequenceSeqIdcsA.resize(num_points + num_new_points);
    m_bufHistoricPointSequencePointsMaskA.resize(num_points + num_new_points);
    m_bufHistoricPointSequencesA.resize(num_seqs + num_new_seqs);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // copy old values to rollbuffer
    m_bufBindings.bind(m_computeRollbufferTransferOldPoints.getGLProgram());
    m_computeRollbufferTransferOldPoints.use();
    m_computeRollbufferTransferOldPoints.num_items.set(num_points);
    m_computeRollbufferTransferOldPoints.in_offset.set(0);
    m_computeRollbufferTransferOldPoints.out_offset.set(0);
    m_computeRollbufferTransferOldPoints.dispatch();

    m_bufBindings.bind(m_computeRollbufferTransferOldPointsMask.getGLProgram());
    m_computeRollbufferTransferOldPointsMask.use();
    m_computeRollbufferTransferOldPointsMask.num_items.set(num_points);
    m_computeRollbufferTransferOldPointsMask.in_offset.set(0);
    m_computeRollbufferTransferOldPointsMask.out_offset.set(0);
    m_computeRollbufferTransferOldPointsMask.dispatch();

    m_bufBindings.bind(m_computeRollbufferTransferOldSeqs.getGLProgram());
    m_computeRollbufferTransferOldSeqs.use();
    m_computeRollbufferTransferOldSeqs.num_items.set(num_seqs);
    m_computeRollbufferTransferOldSeqs.in_offset.set(0);
    m_computeRollbufferTransferOldSeqs.out_offset.set(0);
    m_computeRollbufferTransferOldSeqs.dispatch();

    m_bufBindings.bind(m_computeRollbufferTransferOldSeqIdcs.getGLProgram());
    m_computeRollbufferTransferOldSeqIdcs.use();
    m_computeRollbufferTransferOldSeqIdcs.num_items.set(num_points);
    m_computeRollbufferTransferOldSeqIdcs.in_offset.set(0);
    m_computeRollbufferTransferOldSeqIdcs.out_offset.set(0);
    m_computeRollbufferTransferOldSeqIdcs.dispatch();


    // append new points to rollbuffer
    m_bufBindings.bind(m_computeRollbufferTransferNewPoints.getGLProgram());
    m_computeRollbufferTransferNewPoints.use();
    m_computeRollbufferTransferNewPoints.num_items.set(num_new_points);
    m_computeRollbufferTransferNewPoints.in_offset.set(0);
    m_computeRollbufferTransferNewPoints.out_offset.set(num_points);
    m_computeRollbufferTransferNewPoints.dispatch();

    // append new points mask to rollbuffer
    m_bufBindings.bind(m_computeRollbufferTransferNewPointsMask.getGLProgram());
    m_computeRollbufferTransferNewPointsMask.use();
    m_computeRollbufferTransferNewPointsMask.num_items.set(num_new_points);
    m_computeRollbufferTransferNewPointsMask.in_offset.set(0);
    m_computeRollbufferTransferNewPointsMask.out_offset.set(num_points);
    m_computeRollbufferTransferNewPointsMask.dispatch();

    // set sequence indices (starting from num_seqs) for new points
    m_bufBindings.bind(m_computeRollbufferSetNewPointsSeqIdcs.getGLProgram());
    m_computeRollbufferSetNewPointsSeqIdcs.use();
    for (uint32_t seqIdx=0; seqIdx<num_new_seqs; ++seqIdx)
    {
        const auto& seq = sequences[seqIdx];
        m_computeRollbufferSetNewPointsSeqIdcs.num_items.set(seq.numPoints);
        m_computeRollbufferSetNewPointsSeqIdcs.offset.set(num_points + seq.start);
        m_computeRollbufferSetNewPointsSeqIdcs.value.set(num_seqs + seqIdx);
        m_computeRollbufferSetNewPointsSeqIdcs.dispatch();
    }
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    
    // append new sequences to rollbuffer
    m_bufBindings.bind(m_computeRollbufferTransferNewSeqs.getGLProgram());
    m_computeRollbufferTransferNewSeqs.use();
    m_computeRollbufferTransferNewSeqs.num_items.set(num_new_seqs);
    m_computeRollbufferTransferNewSeqs.in_offset.set(0);
    m_computeRollbufferTransferNewSeqs.out_offset.set(num_seqs);
    m_computeRollbufferTransferNewSeqs.dispatch();
    
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    m_rollBufferNumPoints = num_points + num_new_points;
    m_rollBufferNumSeqs = num_seqs + num_new_seqs;
    
    if (num_new_seqs > 0)
    {
        m_rollBufferLastTimeSec = sequences[sequences.size()-1].timestampSec;
        m_rollBufferLastTimeNSec = sequences[sequences.size()-1].timestampNSec;
    }

    // std::cout << "m_rollBufferNumPoints " << m_rollBufferNumPoints << std::endl;
    // std::cout << "m_rollBufferNumSeqs   " << m_rollBufferNumSeqs << std::endl;    

    // checkAllPointSequenceBuffers();
}

int compareTime(uint timeSecA, uint timeNSecA, uint timeSecB, uint timeNSecB)
{
    if (timeSecA < timeSecB)        return -1;
    else if (timeSecA > timeSecB)   return +1;
    else if (timeNSecA < timeNSecB) return -1;
    else if (timeNSecA > timeNSecB) return +1;
    else return 0;
}

void GPUDepthmapFusion::rollPointSequenceRollbufferCPU(uint32_t minTimestampSec, uint32_t minTimestampNSec)
{
    static std::vector<PointSequence> vecHistoricPointSequencesA;
    vecHistoricPointSequencesA.resize(m_bufHistoricPointSequencesA.size());
    m_bufHistoricPointSequencesA.bind().download(vecHistoricPointSequencesA.data(), 0 , vecHistoricPointSequencesA.size());

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // find earliest sequence with timestamp >= minTimestamp
    int num_discarded_seqs = 0;
    int num_discarded_pts = 0;
    for (int i = 0; i < vecHistoricPointSequencesA.size(); ++i)
    {
        const auto& seq = vecHistoricPointSequencesA[i];
        int cmp = compareTime(seq.timestampSec, seq.timestampNSec, minTimestampSec, minTimestampNSec);
        if (cmp < 0)
        {
            num_discarded_pts += seq.numPoints;
            // num_discarded_seqs++; // instead set final value in else
        }
        else
        {
            num_discarded_seqs = i;
            break;
        }
    }
    // 

    // num_discarded_pts = m_rollBufferNumPoints / 2;
    // num_discarded_seqs = m_rollBufferNumSeqs / 2;

    if (vecHistoricPointSequencesA.size() > num_discarded_seqs)
    {
        m_rollBufferEarliestTimeSec = vecHistoricPointSequencesA[num_discarded_seqs].timestampSec;
        m_rollBufferEarliestTimeNSec = vecHistoricPointSequencesA[num_discarded_seqs].timestampNSec;
    }
    else
    {
        m_rollBufferEarliestTimeSec = 0;
        m_rollBufferEarliestTimeNSec = 0;
        m_rollBufferLastTimeSec = 0;
        m_rollBufferLastTimeNSec = 0;
    }

    // std::cout << "minTimestampSec  " << minTimestampSec << std::endl;
    // std::cout << "minTimestampNSec " << minTimestampNSec << std::endl;

    // std::cout << "num_discarded_pts  " << num_discarded_pts << std::endl;
    // std::cout << "num_discarded_seqs " << num_discarded_seqs << std::endl;
    
    uint32_t num_points = m_rollBufferNumPoints;
    uint32_t num_seqs = m_rollBufferNumSeqs;

    uint32_t num_remaining_pts = num_points - num_discarded_pts;
    uint32_t num_remaining_seqs = num_seqs - num_discarded_seqs;
    
    // resize target buffers
    m_bufHistoricPointSequenceRemainingPointIndices.resize(num_remaining_pts);
    m_bufHistoricPointSequenceRemainingSeqIndices.resize(num_remaining_seqs);
    m_bufHistoricPointSequencePointsB.resize(num_remaining_pts);
    m_bufHistoricPointSequenceSeqIdcsB.resize(num_remaining_pts);
    m_bufHistoricPointSequencePointsMaskB.resize(num_remaining_pts);
    m_bufHistoricPointSequencesB.resize(num_remaining_seqs);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // compute remaining point input indices
    m_bufBindings.bind(m_computeRollbufferRemainingPointsIndices.getGLProgram());
    m_computeRollbufferRemainingPointsIndices.use();
    m_computeRollbufferRemainingPointsIndices.num_items.set(num_remaining_pts);
    m_computeRollbufferRemainingPointsIndices.num_discarded_pts.set(num_discarded_pts);
    m_computeRollbufferRemainingPointsIndices.out_offset.set(0);
    m_computeRollbufferRemainingPointsIndices.dispatch();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // copy remaining points from input indices and update point sequence indices
    m_bufBindings.bind(m_computeRollbufferRemainingPointsCopyAndUpdate.getGLProgram());
    m_computeRollbufferRemainingPointsCopyAndUpdate.use();
    m_computeRollbufferRemainingPointsCopyAndUpdate.num_items.set(num_remaining_pts);
    m_computeRollbufferRemainingPointsCopyAndUpdate.num_discarded_seqs.set(num_discarded_seqs);
    m_computeRollbufferRemainingPointsCopyAndUpdate.in_remaining_idcs_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.in_points_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.in_seq_idcs_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.in_mask_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.out_points_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.out_seq_idcs_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.out_mask_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.dispatch();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
    // compute remaining sequence input indices
    m_bufBindings.bind(m_computeRollbufferRemainingSeqsIndices.getGLProgram());
    m_computeRollbufferRemainingSeqsIndices.use();
    m_computeRollbufferRemainingSeqsIndices.num_items.set(num_remaining_seqs);
    m_computeRollbufferRemainingSeqsIndices.num_discarded_seqs.set(num_discarded_seqs);
    m_computeRollbufferRemainingSeqsIndices.out_offset.set(0);
    m_computeRollbufferRemainingSeqsIndices.dispatch();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // copy remaining sequences from input indices
    m_bufBindings.bind(m_computeRollbufferRemainingSeqsCopy.getGLProgram());
    m_computeRollbufferRemainingSeqsCopy.use();
    m_computeRollbufferRemainingSeqsCopy.num_items.set(num_remaining_seqs);
    m_computeRollbufferRemainingSeqsCopy.in_indices_offset.set(0);
    m_computeRollbufferRemainingSeqsCopy.in_data_offset.set(0);
    m_computeRollbufferRemainingSeqsCopy.out_data_offset.set(0);
    m_computeRollbufferRemainingSeqsCopy.dispatch();

    m_rollBufferNumPoints = num_remaining_pts;
    m_rollBufferNumSeqs = num_remaining_seqs;

    // std::cout << "m_rollBufferNumPoints " << m_rollBufferNumPoints << std::endl;
    // std::cout << "m_rollBufferNumSeqs   " << m_rollBufferNumSeqs << std::endl;

    // if (num_discarded_seqs > 0)
        // checkAllPointSequenceBuffers();
}
void GPUDepthmapFusion::rollPointSequenceRollbuffer(uint32_t minTimestampSec, uint32_t minTimestampNSec)
{
    uint zero = 0;
    // roll buffers discarding elements older than given timestamp
    
    uint32_t num_points = m_rollBufferNumPoints;
    uint32_t num_seqs = m_rollBufferNumSeqs;
    // two variants for the compute programs operating on swap buffers
    // they wired so the active rollbuffer to write new data into is buf[toggle] 

    // clear scratchpad
    uint32_t num_discarded_pts = 0;
    uint32_t num_discarded_seqs = 0;
    m_bufRollbufferScratchpad.bind();
    m_bufRollbufferScratchpad.resize(2);
    m_bufRollbufferScratchpad.upload(&num_discarded_pts, 0, 1);
    m_bufRollbufferScratchpad.upload(&num_discarded_seqs, 1, 1);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // count discarded points
    m_bufBindings.bind(m_computeRollbufferCountDiscardedPoints.getGLProgram());
    m_computeRollbufferCountDiscardedPoints.use();
    m_computeRollbufferCountDiscardedPoints.num_items.set(num_points);
    m_computeRollbufferCountDiscardedPoints.min_timestamp_sec.set(minTimestampSec);
    m_computeRollbufferCountDiscardedPoints.min_timestamp_nsec.set(minTimestampNSec);
    m_computeRollbufferCountDiscardedPoints.in_seq_idcs_offset.set(0);
    m_computeRollbufferCountDiscardedPoints.in_seqs_offset.set(0);
    m_computeRollbufferCountDiscardedPoints.scratchpad_offset.set(0);
    m_computeRollbufferCountDiscardedPoints.dispatch();

    // count discarded sequences
    m_bufBindings.bind(m_computeRollbufferCountDiscardedSeqs.getGLProgram());
    m_computeRollbufferCountDiscardedSeqs.use();
    m_computeRollbufferCountDiscardedSeqs.num_items.set(num_seqs);
    m_computeRollbufferCountDiscardedSeqs.min_timestamp_sec.set(minTimestampSec);
    m_computeRollbufferCountDiscardedSeqs.min_timestamp_nsec.set(minTimestampNSec);
    m_computeRollbufferCountDiscardedSeqs.in_seqs_offset.set(0);
    m_computeRollbufferCountDiscardedSeqs.scratchpad_offset.set(1);
    m_computeRollbufferCountDiscardedSeqs.dispatch();
    
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    m_bufRollbufferScratchpad.bind();
    m_bufRollbufferScratchpad.download(&num_discarded_pts, 0, 1);
    m_bufRollbufferScratchpad.download(&num_discarded_seqs, 1, 1);


    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // debug: discard all
    // num_discarded_pts = num_points;
    // num_discarded_seqs = num_seqs;
    if (num_points > 100000)
    {
        num_discarded_pts = num_points;
        num_discarded_seqs = num_seqs;
    }
    else
    {
        num_discarded_pts = 0;
        num_discarded_seqs = 0;
    }

    std::cout << "minTimestampSec  " << minTimestampSec << std::endl;
    std::cout << "minTimestampNSec " << minTimestampNSec << std::endl;

    std::cout << "num_discarded_pts  " << num_discarded_pts << std::endl;
    std::cout << "num_discarded_seqs " << num_discarded_seqs << std::endl;

    uint32_t num_remaining_pts = num_points - num_discarded_pts;
    uint32_t num_remaining_seqs = num_seqs - num_discarded_seqs;
    
    // resize target buffers
    m_bufHistoricPointSequenceRemainingPointIndices.resize(num_remaining_pts);
    m_bufHistoricPointSequenceRemainingSeqIndices.resize(num_remaining_seqs);
    m_bufHistoricPointSequencePointsB.resize(num_remaining_pts);
    m_bufHistoricPointSequenceSeqIdcsB.resize(num_remaining_pts);
    m_bufHistoricPointSequencePointsMaskB.resize(num_remaining_pts);
    m_bufHistoricPointSequencesB.resize(num_remaining_seqs);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // compute remaining point input indices
    m_bufBindings.bind(m_computeRollbufferRemainingPointsIndices.getGLProgram());
    m_computeRollbufferRemainingPointsIndices.use();
    m_computeRollbufferRemainingPointsIndices.num_items.set(num_remaining_pts);
    m_computeRollbufferRemainingPointsIndices.num_discarded_pts.set(num_discarded_pts);
    m_computeRollbufferRemainingPointsIndices.out_offset.set(0);
    m_computeRollbufferRemainingPointsIndices.dispatch();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // copy remaining points from input indices and update point sequence indices
    m_bufBindings.bind(m_computeRollbufferRemainingPointsCopyAndUpdate.getGLProgram());
    m_computeRollbufferRemainingPointsCopyAndUpdate.use();
    m_computeRollbufferRemainingPointsCopyAndUpdate.num_items.set(num_remaining_pts);
    m_computeRollbufferRemainingPointsCopyAndUpdate.num_discarded_seqs.set(num_discarded_seqs);
    m_computeRollbufferRemainingPointsCopyAndUpdate.in_remaining_idcs_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.in_points_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.in_seq_idcs_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.in_mask_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.out_points_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.out_seq_idcs_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.out_mask_offset.set(0);
    m_computeRollbufferRemainingPointsCopyAndUpdate.dispatch();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
    // compute remaining sequence input indices
    m_bufBindings.bind(m_computeRollbufferRemainingSeqsIndices.getGLProgram());
    m_computeRollbufferRemainingSeqsIndices.use();
    m_computeRollbufferRemainingSeqsIndices.num_items.set(num_remaining_seqs);
    m_computeRollbufferRemainingSeqsIndices.num_discarded_seqs.set(num_discarded_seqs);
    m_computeRollbufferRemainingSeqsIndices.out_offset.set(0);
    m_computeRollbufferRemainingSeqsIndices.dispatch();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // copy remaining sequences from input indices
    m_bufBindings.bind(m_computeRollbufferRemainingSeqsCopy.getGLProgram());
    m_computeRollbufferRemainingSeqsCopy.use();
    m_computeRollbufferRemainingSeqsCopy.num_items.set(num_remaining_seqs);
    m_computeRollbufferRemainingSeqsCopy.in_indices_offset.set(0);
    m_computeRollbufferRemainingSeqsCopy.in_data_offset.set(0);
    m_computeRollbufferRemainingSeqsCopy.out_data_offset.set(0);
    m_computeRollbufferRemainingSeqsCopy.dispatch();

    m_rollBufferNumPoints = num_remaining_pts;
    m_rollBufferNumSeqs = num_remaining_seqs;

    // std::cout << "m_rollBufferNumPoints " << m_rollBufferNumPoints << std::endl;
    // std::cout << "m_rollBufferNumSeqs   " << m_rollBufferNumSeqs << std::endl;



    // checkAllPointSequenceBuffers();
    // checkRollPointSequenceRollbuffer();
}

void GPUDepthmapFusion::selectPointSequenceTimespanCPU(
    uint32_t minTimestampSec, 
    uint32_t minTimestampNSec, 
    uint32_t maxTimestampSec, 
    uint32_t maxTimestampNSec)
{
    // static std::vector<PointSequence> vecHistoricPointSequencesA;
    static std::vector<PointSequence> vecHistoricPointSequencesB;
    // vecHistoricPointSequencesA.resize(m_bufHistoricPointSequencesA.size());
    vecHistoricPointSequencesB.resize(m_bufHistoricPointSequencesB.size());
    // m_bufHistoricPointSequencesA.bind().download(vecHistoricPointSequencesA.data(), 0 , vecHistoricPointSequencesA.size());
    m_bufHistoricPointSequencesB.bind().download(vecHistoricPointSequencesB.data(), 0 , vecHistoricPointSequencesB.size());


    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    uint32_t num_points = m_rollBufferNumPoints;
    uint32_t num_seqs = m_rollBufferNumSeqs;

    int sel_seq_start = num_seqs;
    int sel_seq_last = 0;
    int sel_seq_count = 0;
    int sel_point_start = 0;
    int sel_point_count = 0;

    for (int i = 0; i < vecHistoricPointSequencesB.size(); ++i)
    {
        const auto& seq = vecHistoricPointSequencesB[i];
        int cmpMinSeq = compareTime(minTimestampSec, minTimestampNSec, seq.timestampSec, seq.timestampNSec);
        int cmpSeqMax = compareTime(seq.timestampSec, seq.timestampNSec, maxTimestampSec, maxTimestampNSec);
        if ((cmpMinSeq <= 0) && (cmpSeqMax <= 0))
        {
            if (i < sel_seq_start) sel_seq_start = i;
            if (i > sel_seq_last)  sel_seq_last = i;
            sel_point_count += seq.numPoints;
        }
        
    }

    if (sel_seq_last < sel_seq_start) sel_seq_count = 0;
    else sel_seq_count = 1 + sel_seq_last - sel_seq_start;

    for (int i = 0; i < sel_seq_start; ++i)
    {
        const auto& seq = vecHistoricPointSequencesB[i];
        sel_point_start += seq.numPoints;
    }

    m_rollBufferSelectionPointStart = sel_point_start;
    m_rollBufferSelectionPointCount = sel_point_count;
    m_rollBufferSelectionSequenceStart = sel_seq_start;
    m_rollBufferSelectionSequenceCount = sel_seq_count;

    // std::cout << "m_rollBufferSelectionPointStart    " << m_rollBufferSelectionPointStart << std::endl;
    // std::cout << "m_rollBufferSelectionPointCount    " << m_rollBufferSelectionPointCount << std::endl;
    // std::cout << "m_rollBufferSelectionSequenceStart " << m_rollBufferSelectionSequenceStart << std::endl;
    // std::cout << "m_rollBufferSelectionSequenceCount " << m_rollBufferSelectionSequenceCount << std::endl;

}
void GPUDepthmapFusion::selectPointSequenceTimespan(
    uint32_t minTimestampSec, 
    uint32_t minTimestampNSec, 
    uint32_t maxTimestampSec, 
    uint32_t maxTimestampNSec)
{
    uint32_t num_points = m_rollBufferNumPoints;
    uint32_t num_seqs = m_rollBufferNumSeqs;

    uint32_t sel_point_start = 0;
    uint32_t sel_point_last = 0;
    uint32_t sel_point_count = 0;
    uint32_t sel_seq_start = 0;
    uint32_t sel_seq_last = 0;
    uint32_t sel_seq_count = 0;

    m_bufRollbufferSelected.resize(4);
    
    // find start index and count for points and sequences between min timestamp and max timestamp
    // m_bufRollbufferSelected;   0: point start, 1: point last, 2: seq start, 3: seq last
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    m_bufRollbufferSelected.bind();
    m_bufRollbufferSelected.upload(&sel_point_start, 0, 1);
    m_bufRollbufferSelected.upload(&sel_point_last, 1, 1);
    m_bufRollbufferSelected.upload(&sel_seq_start, 2, 1);
    m_bufRollbufferSelected.upload(&sel_point_last, 3, 1);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    m_bufBindings.bind(m_computeRollbufferSelectTimespanPoints.getGLProgram());
    m_computeRollbufferSelectTimespanPoints.use();
    m_computeRollbufferSelectTimespanPoints.num_items.set(num_points);
    m_computeRollbufferSelectTimespanPoints.min_timestamp_sec.set(minTimestampSec);
    m_computeRollbufferSelectTimespanPoints.min_timestamp_nsec.set(minTimestampNSec);
    m_computeRollbufferSelectTimespanPoints.max_timestamp_sec.set(maxTimestampSec);
    m_computeRollbufferSelectTimespanPoints.max_timestamp_nsec.set(maxTimestampNSec);
    m_computeRollbufferSelectTimespanPoints.in_seq_idcs_offset.set(0);
    m_computeRollbufferSelectTimespanPoints.in_sequences_offset.set(0);
    m_computeRollbufferSelectTimespanPoints.selection_offset.set(0);
    m_computeRollbufferSelectTimespanPoints.dispatch();

    m_bufBindings.bind(m_computeRollbufferSelectTimespanSequences.getGLProgram());
    m_computeRollbufferSelectTimespanSequences.use();
    m_computeRollbufferSelectTimespanSequences.num_items.set(num_seqs);
    m_computeRollbufferSelectTimespanSequences.min_timestamp_sec.set(minTimestampSec);
    m_computeRollbufferSelectTimespanSequences.min_timestamp_nsec.set(minTimestampNSec);
    m_computeRollbufferSelectTimespanSequences.max_timestamp_sec.set(maxTimestampSec);
    m_computeRollbufferSelectTimespanSequences.max_timestamp_nsec.set(maxTimestampNSec);
    m_computeRollbufferSelectTimespanSequences.in_seq_idcs_offset.set(0);
    m_computeRollbufferSelectTimespanSequences.in_sequences_offset.set(0);
    m_computeRollbufferSelectTimespanSequences.selection_offset.set(2);
    m_computeRollbufferSelectTimespanSequences.dispatch();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    m_bufRollbufferSelected.bind();
    m_bufRollbufferSelected.download(&sel_point_start, 0, 1);
    m_bufRollbufferSelected.download(&sel_point_last, 1, 1);
    m_bufRollbufferSelected.download(&sel_seq_start, 2, 1);
    m_bufRollbufferSelected.download(&sel_seq_last, 3, 1);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    if (sel_point_last <= sel_point_start) sel_point_count = 0;
    else  sel_point_count = sel_point_last + 1 - sel_point_start;
    if (sel_seq_last <= sel_seq_start) sel_seq_count = 0;
    else  sel_seq_count = sel_seq_last + 1 - sel_seq_start;

    m_rollBufferSelectionPointStart = sel_point_start;
    m_rollBufferSelectionPointCount = sel_point_count;
    m_rollBufferSelectionSequenceStart = sel_seq_start;
    m_rollBufferSelectionSequenceCount = sel_seq_count;

    std::cout << "m_rollBufferSelectionPointStart    " << m_rollBufferSelectionPointStart << std::endl;
    std::cout << "m_rollBufferSelectionPointCount    " << m_rollBufferSelectionPointCount << std::endl;
    std::cout << "m_rollBufferSelectionSequenceStart " << m_rollBufferSelectionSequenceStart << std::endl;
    std::cout << "m_rollBufferSelectionSequenceCount " << m_rollBufferSelectionSequenceCount << std::endl;

    // checkAllPointSequenceBuffers();
}
void GPUDepthmapFusion::preparePointAndMaskBuffers()
{

    m_numPointsTotal = m_depthmapsTotalElements + m_rollBufferSelectionPointCount;

    // warning: resize buffers so we can append will discard previous content
    m_bufPointsA.resize(m_numPointsTotal);
    m_bufPointsB.resize(m_numPointsTotal);
    m_bufPointsC.resize(m_numPointsTotal);
    m_bufMaskA.resize(m_numPointsTotal);
    m_bufMaskB.resize(m_numPointsTotal);
}
void GPUDepthmapFusion::insertSelectedPointSequence(
    const cv::Matx44f& tf_world_move,
    const cv::Matx44f& tf_crop_move
    )
{
    m_bufRollbufferSelectedTransformIdcs.resize(m_rollBufferSelectionPointCount);
    m_bufRollbufferSelectedTransformsWorld.resize(m_rollBufferSelectionSequenceCount);
    m_bufRollbufferSelectedTransformsCrop.resize(m_rollBufferSelectionSequenceCount);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // append selected mask to all points mask (B)
    m_bufBindings.bind(m_computeRollbufferTransferSelectedMask.getGLProgram());
    m_computeRollbufferTransferSelectedMask.use();
    m_computeRollbufferTransferSelectedMask.num_items.set(m_rollBufferSelectionPointCount);
    m_computeRollbufferTransferSelectedMask.in_offset.set(m_rollBufferSelectionPointStart);
    m_computeRollbufferTransferSelectedMask.out_offset.set(m_depthmapsTotalElements);
    m_computeRollbufferTransferSelectedMask.dispatch();    

    // select transform indices from sequences
    // the indices will be renumbered
    // from: seqidcs[sel_point_start]..seqidcs[sel_point_start]+N
    //   to: 0..N
    m_bufBindings.bind(m_computeRollbufferTransferSelectedTransformIndices.getGLProgram());
    m_computeRollbufferTransferSelectedTransformIndices.use();
    m_computeRollbufferTransferSelectedTransformIndices.num_items.set(m_rollBufferSelectionPointCount);
    m_computeRollbufferTransferSelectedTransformIndices.in_seq_idcs_offset.set(m_rollBufferSelectionPointStart);
    m_computeRollbufferTransferSelectedTransformIndices.out_tf_idcs_offset.set(0);
    m_computeRollbufferTransferSelectedTransformIndices.dispatch();

    // assert (m_bufHistoricPointSequenceSeqIdcs_[sel_point_start] == sel_seq_start) ;

    // select transforms from sequences
    m_bufBindings.bind(m_computeRollbufferTransferSelectedTransforms.getGLProgram());
    m_computeRollbufferTransferSelectedTransforms.use();
    m_computeRollbufferTransferSelectedTransforms.num_items.set(m_rollBufferSelectionSequenceCount);
    m_computeRollbufferTransferSelectedTransforms.in_seqs_offset.set(m_rollBufferSelectionSequenceStart);
    m_computeRollbufferTransferSelectedTransforms.out_tfs_world_offset.set(0);
    m_computeRollbufferTransferSelectedTransforms.out_tfs_crop_offset.set(0);
    m_computeRollbufferTransferSelectedTransforms.tf_world_move.set(tf_world_move);
    m_computeRollbufferTransferSelectedTransforms.tf_crop_move.set(tf_crop_move);
    m_computeRollbufferTransferSelectedTransforms.dispatch();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GPUDepthmapFusion::transformPointSequence()
{
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    uint32_t in_offset = m_rollBufferSelectionPointStart;
    uint32_t out_offset = m_depthmapsTotalElements;

    m_bufBindings.bind(m_computeTransformPointSequenceWorld.getGLProgram());
    m_computeTransformPointSequenceWorld.use();
    m_computeTransformPointSequenceWorld.num_items.set(m_rollBufferSelectionPointCount);
    m_computeTransformPointSequenceWorld.in_points_offset.set(in_offset);
    m_computeTransformPointSequenceWorld.in_mask_offset.set(in_offset);
    m_computeTransformPointSequenceWorld.in_transforms_offset.set(0);
    m_computeTransformPointSequenceWorld.in_transform_indices_offset.set(0);
    m_computeTransformPointSequenceWorld.out_points_offset.set(out_offset);
    m_computeTransformPointSequenceWorld.dispatch();

    m_bufBindings.bind(m_computeTransformPointSequenceCrop.getGLProgram());
    m_computeTransformPointSequenceCrop.use();
    m_computeTransformPointSequenceCrop.num_items.set(m_rollBufferSelectionPointCount);
    m_computeTransformPointSequenceCrop.in_points_offset.set(in_offset);
    m_computeTransformPointSequenceCrop.in_mask_offset.set(in_offset);
    m_computeTransformPointSequenceCrop.in_transforms_offset.set(0);
    m_computeTransformPointSequenceCrop.in_transform_indices_offset.set(0);
    m_computeTransformPointSequenceCrop.out_points_offset.set(out_offset);
    m_computeTransformPointSequenceCrop.dispatch();

}

void GPUDepthmapFusion::uploadDepthmaps()
{
    m_bufDepthPairs.resize(m_depthmapsTotalElements / 2);
    m_bufDepthPairs.bind();
    int offset = 0;
    for (const auto& item : m_depthmaps)
    {
        m_bufDepthPairs.upload(item.depthmap, offset, item.numItems / 2);
        offset += item.numItems / 2;
    }
}

void GPUDepthmapFusion::convertDepthmaps()
{
    // the buffers are resized in preparePointAndMaskBuffers
    assert(m_bufPointsA.size() >= m_depthmapsTotalElements);
    assert(m_bufPointsB.size() >= m_depthmapsTotalElements);
    assert(m_bufPointsC.size() >= m_depthmapsTotalElements);
    assert(m_bufMaskA.size() >= m_depthmapsTotalElements);
    assert(m_bufMaskB.size() >= m_depthmapsTotalElements);

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    int offset = 0;
    m_bufBindings.bind(m_computeConvertDepthmapToPoints.getGLProgram());
    m_computeConvertDepthmapToPoints.use();
    m_computeConvertDepthmapToPoints.rectify_offset.set(0);
    for (const auto& item : m_depthmaps)
    {
        m_computeConvertDepthmapToPoints.num_items.set(item.numItems);
        m_computeConvertDepthmapToPoints.width.set(item.width);
        m_computeConvertDepthmapToPoints.height.set(item.height);
        m_computeConvertDepthmapToPoints.depth_scale.set(item.depthScale);
        m_computeConvertDepthmapToPoints.fx.set(item.fx);
        m_computeConvertDepthmapToPoints.fy.set(item.fy);
        m_computeConvertDepthmapToPoints.cx.set(item.cx);
        m_computeConvertDepthmapToPoints.cy.set(item.cy);
        m_computeConvertDepthmapToPoints.transform_world.set(item.transform_world);
        m_computeConvertDepthmapToPoints.transform_crop.set(item.transform_crop);
        m_computeConvertDepthmapToPoints.depth_pair_offset.set(offset/2);
        m_computeConvertDepthmapToPoints.point_offset.set(offset);
        m_computeConvertDepthmapToPoints.mask_offset.set(offset);
        m_computeConvertDepthmapToPoints.dispatch();
        offset += item.numItems;
    }
    
}
void GPUDepthmapFusion::filterFlyingPixels(uint filter_size, float threshold, bool enable_rot45)
{
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    int offset = 0;
    m_bufBindings.bind(m_computeFilterFlyingPixels.getGLProgram());
    m_computeFilterFlyingPixels.use();
    m_computeFilterFlyingPixels.threshold_view_angle.set(threshold);
    m_computeFilterFlyingPixels.filter_size.set(filter_size);
    m_computeFilterFlyingPixels.enable_rot45.set(enable_rot45 ? 1 : 0);
    for (const auto& item : m_depthmaps)
    {
        m_computeFilterFlyingPixels.num_items.set(item.width*item.height);
        m_computeFilterFlyingPixels.width.set(item.width);
        m_computeFilterFlyingPixels.height.set(item.height);
        m_computeFilterFlyingPixels.point_offset.set(offset);
        m_computeFilterFlyingPixels.mask_offset.set(offset);
        m_computeFilterFlyingPixels.dispatch();
        offset += item.numItems;
    }
}
void GPUDepthmapFusion::cropPoints(glm::vec3 lower_bound, glm::vec3 upper_bound)
{
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_bufBindings.bind(m_computeCropPoints.getGLProgram());
    m_computeCropPoints.use();
    m_computeCropPoints.num_items.set(m_numPointsTotal);
    m_computeCropPoints.point_offset.set(0);
    m_computeCropPoints.mask_offset.set(0);
    m_computeCropPoints.lower_bound.set(lower_bound);
    m_computeCropPoints.upper_bound.set(upper_bound);
    m_computeCropPoints.dispatch();
}
void GPUDepthmapFusion::applyPointMask()
{
    uint zero = 0;
    m_bufScratchpad.bind().upload(&zero, 0, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_bufBindings.bind(m_computeApplyPointMask.getGLProgram());
    m_computeApplyPointMask.use();
    m_computeApplyPointMask.num_items.set(m_numPointsTotal);
    m_computeApplyPointMask.in_point_offset.set(0);
    m_computeApplyPointMask.out_point_offset.set(0);
    m_computeApplyPointMask.mask_offset.set(0);
    m_computeApplyPointMask.dispatch();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_numItemsAfterMask = 0;
    m_bufScratchpad.bind().download(&m_numItemsAfterMask, 0, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
}

void GPUDepthmapFusion::computeVoxelCoords(glm::vec3 lower_bound, glm::vec3 upper_bound, glm::vec3 cell_size)
{
    m_bufVoxelCoords.resize(m_numItemsAfterMask);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_bufBindings.bind(m_computeVoxelCoords.getGLProgram());
    m_computeVoxelCoords.use();
    m_computeVoxelCoords.num_items.set(m_numItemsAfterMask);
    m_computeVoxelCoords.in_point_offset.set(0);
    m_computeVoxelCoords.out_coord_offset.set(0);
    m_computeVoxelCoords.lower_bound.set(lower_bound);
    m_computeVoxelCoords.upper_bound.set(upper_bound);
    m_computeVoxelCoords.cell_size.set(cell_size);
    glm::vec3 fsize = (upper_bound - lower_bound) / cell_size;
    glm::uvec3 grid_size(
        (uint)ceil(fsize[0]),
        (uint)ceil(fsize[1]),
        (uint)ceil(fsize[2]));
    m_computeVoxelCoords.grid_size.set(grid_size);
    m_computeVoxelCoords.dispatch();
    m_voxelGridMeta.set(
        glm::vec4(lower_bound, 1), 
        glm::vec4(upper_bound, 1),
        glm::vec4(cell_size, 1),
        glm::bvec3(false,false,false)
        );
        // cv::Vec3f(lower_bound[0], lower_bound[1], lower_bound[2]),
        // cv::Vec3f(upper_bound[0], upper_bound[1], upper_bound[2]),
        // cv::Vec3f(cell_size[0], cell_size[1], cell_size[2]),
        // cv::Vec<bool,3>(false, false, false));

}
void GPUDepthmapFusion::downloadVoxelCoords()
{
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_voxelCoords.resize(m_numItemsAfterMask);
    m_bufVoxelCoords.bind().download(m_voxelCoords.data(), 0, m_voxelCoords.size());
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
}

glm::vec3 GPUDepthmapFusion::voxelCoordToWorldCoord(float x, float y, float z)
{
    auto lb = m_voxelGridMeta.lowerBound();
    auto cs = m_voxelGridMeta.cellSize();

    return glm::vec3(
        x*cs[0] + lb[0],
        y*cs[1] + lb[1],
        z*cs[2] + lb[2]
    );
}
glm::vec3 GPUDepthmapFusion::worldCoordToVoxelCoord(float x, float y, float z)
{
    auto lb = m_voxelGridMeta.lowerBound();
    auto cs = m_voxelGridMeta.cellSize();
    return glm::vec3(
        (x-lb[0])/cs[0],
        (y-lb[1])/cs[1],
        (z-lb[2])/cs[2]
    );

}

void GPUDepthmapFusion::voxelize(bool average_voxels)
{
    m_points_voxelized.resize(m_voxelCoords.size());
    ::voxelize<glm::vec4, VoxelGridMeta, 3, uint32_t>(
        m_voxelRadixGrouper,
        m_voxelGridMeta,
        &m_voxelCoords[0],
        m_voxelCoords.size(),
        m_points,
        m_points_voxelized,
        average_voxels,
        m_enableParallelVoxelFilter
    );
}
void GPUDepthmapFusion::voxelOccupancyGrid(uint32_t lifetime)
{
    static bool invoked_once = false;

    // clear historic occupancy grid on first invocation or on resize
    if (!invoked_once || m_bufHistoricVoxelOccupancyA.size() != m_voxelGridMeta.numCells())
    {
        m_bufHistoricVoxelOccupancyA.resize(m_voxelGridMeta.numCells());
        m_bufBindings.bind(m_computeHistoricVoxelGridOccupancyClear.getGLProgram());
        m_computeHistoricVoxelGridOccupancyClear.use();
        m_computeHistoricVoxelGridOccupancyClear.num_items.set(m_voxelGridMeta.numCells());
        m_computeHistoricVoxelGridOccupancyClear.offset.set(0);
        m_computeHistoricVoxelGridOccupancyClear.dispatch();
    }
    m_bufHistoricVoxelOccupancyB.resize(m_voxelGridMeta.numCells());
    
    invoked_once = true;

    // clear
    m_bufVoxelOccupancyA.resize(m_voxelGridMeta.numCells());
    m_bufVoxelOccupancyB.resize(m_voxelGridMeta.numCells()/4 + ((m_voxelGridMeta.numCells()%4 == 0) ? 0 : 1));

    m_bufBindings.bind(m_computeVoxelGridOccupancyClear.getGLProgram());
    m_computeVoxelGridOccupancyClear.use();
    // std::cout << "m_voxelGridMeta.numCells() " << m_voxelGridMeta.numCells() << std::endl;
    m_computeVoxelGridOccupancyClear.num_items.set(m_voxelGridMeta.numCells());
    m_computeVoxelGridOccupancyClear.offset.set(0);
    m_computeVoxelGridOccupancyClear.dispatch();

    // occupancy of points with uint32_t for each cell
    m_bufBindings.bind(m_computeVoxelGridOccupancyOfPoints.getGLProgram());
    m_computeVoxelGridOccupancyOfPoints.use();
    m_computeVoxelGridOccupancyOfPoints.num_items.set(m_numItemsAfterMask);
    m_computeVoxelGridOccupancyOfPoints.in_coord_offset.set(0);
    m_computeVoxelGridOccupancyOfPoints.out_occupancy_offset.set(0);
    m_computeVoxelGridOccupancyOfPoints.occupied_value.set(1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_computeVoxelGridOccupancyOfPoints.dispatch();

    // advance historic occupancy grid by decrementing cell values (min value=0)
    m_bufBindings.bind(m_computeHistoricVoxelGridOccupancyDec.getGLProgram());
    m_computeHistoricVoxelGridOccupancyDec.use();
    m_computeHistoricVoxelGridOccupancyDec.num_items.set(m_voxelGridMeta.numCells());
    m_computeHistoricVoxelGridOccupancyDec.offset.set(0);
    m_computeHistoricVoxelGridOccupancyDec.decrement.set(1); 
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_computeHistoricVoxelGridOccupancyDec.dispatch();

    // set current occupancy multiplied by lifetime to historic occupancy grid
    m_bufBindings.bind(m_computeHistoricVoxelGridOccupancySet.getGLProgram());
    m_computeHistoricVoxelGridOccupancySet.use();
    m_computeHistoricVoxelGridOccupancySet.num_items.set(m_voxelGridMeta.numCells());
    m_computeHistoricVoxelGridOccupancySet.offset.set(0);
    m_computeHistoricVoxelGridOccupancySet.multiplier.set(lifetime);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_computeHistoricVoxelGridOccupancySet.dispatch();
    
    // convert historic occupancy grid to occupancy uint8_t for each cell
    m_bufBindings.bind(m_computeVoxelGridOccupancyToChars.getGLProgram());
    m_computeVoxelGridOccupancyToChars.use();
    m_computeVoxelGridOccupancyToChars.num_items.set(m_voxelGridMeta.numCells()/4 + ((m_voxelGridMeta.numCells()%4 == 0) ? 0 : 1));
    m_computeVoxelGridOccupancyToChars.in_data_offset.set(0);
    m_computeVoxelGridOccupancyToChars.out_data_offset.set(0);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_computeVoxelGridOccupancyToChars.dispatch();

}
void GPUDepthmapFusion::downloadVoxelOccupancyGrid()
{
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    m_occupancyGrid.resize(m_voxelGridMeta.numCells());
    m_bufVoxelOccupancyB.bind().download(m_occupancyGrid.data(),0,m_voxelGridMeta.numCells()/4 + ((m_voxelGridMeta.numCells()%4 == 0) ? 0 : 1));
    int width = m_voxelGridMeta.gridSize()[0];
    int height = m_voxelGridMeta.gridSize()[1];
    int num_layers = m_voxelGridMeta.gridSize()[2];
    m_occupancyLayers.clear();
    for (int i = 0; i < num_layers; ++i)
    {
        m_occupancyLayers.push_back(
            cv::Mat_<uint8_t>(height, width, (&m_occupancyGrid[i*width*height]))
        );
    }
}

template <typename TLabel>
void colorizeLabels(const cv::Mat& labels, cv::Mat& output)
{
    static std::vector<cv::Vec3b> colors;
    if (colors.size() == 0)
    {
        colors.resize(100);
        colors[0] = cv::Vec3b(0, 0, 0);//background
        for(int label = 1; label < colors.size(); ++label)
        {
            colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
        }
    }

    static std::vector<uint8_t> coloredData;
    coloredData.resize(labels.rows, labels.cols);
    // cv::Mat labelsColor = cv::Mat(labels.rows, labels.cols, CV_8UC3);
    output.create(labels.rows, labels.cols, CV_8UC3);
    for(int r = 0; r < output.rows; ++r)
    {
        for(int c = 0; c < output.cols; ++c)
        {
            TLabel label = labels.at<TLabel>(r, c);
            cv::Vec3b &pixel = output.at<cv::Vec3b>(r, c);
            pixel = colors[label == 0 ? 0 : (1+(label-1)%(colors.size()-1))];
        }
    }
}

// typedef std::vector<cv::Point> ContourVector;
// typedef std::vector<std::vector<cv::Point>> ContoursVector;
void GPUDepthmapFusion::labelVoxels()
{
    int numLayers = m_occupancyLayers.size();
    // std::cout << "num_layers " << numLayers << std::endl;
    m_contoursPerLayer.clear();
    // m_bboxesPerLayer.clear();
    m_contoursPerLayer.resize(numLayers);
    m_labelsToContoursPerLayer.resize(numLayers);
    // m_bboxesPerLayer.resize(numLayers);
    int height = (numLayers == 0) ? 0 : m_occupancyLayers[0].cols;
    int width = (numLayers == 0) ? 0 : m_occupancyLayers[0].rows;

    m_ccLabeledLayersData.resize(numLayers * width * height);
    m_ccLabeledLayers.resize(numLayers);
    m_ccNumLabelsPerLayer.resize(numLayers);
    // std::cout << "m_ccLabeledLayersData.size() " << m_ccLabeledLayersData.size() << std::endl;

    // m_ccStatsData
    // m_ccCentroidsData

    std::vector<cv::Mat> stats;
    std::vector<cv::Mat> centroids;
    stats.resize(numLayers);
    centroids.resize(numLayers);


    #pragma omp parallel for
    for (int i = 0; i < numLayers; ++i)
    {
        // for (int j = 0; j < contours.size(); ++j)
        // {
        //     auto rect = cv::boundingRect(contours[j]);
        //     boxes.push_back(rect);
        //     // std::cout << "boundingRect " << rect << std::endl;
        // }
        // cv::Mat dst = cv::Mat(m_occupancyLayers[i].rows, m_occupancyLayers[i].cols, CV_8UC3);
        // cv::cvtColor(m_occupancyLayers[i], dst, CV_GRAY2RGB);
        // cv::drawContours(dst, contours, -1, cv::Scalar(0,0,255), 1);

        // cv::Mat labels = cv::Mat(m_occupancyLayers[i].rows, m_occupancyLayers[i].cols, CV_16U);
        // int numLabels = cv::connectedComponents(m_occupancyLayers[i], labels, 8, CV_16U);

        // m_ccLabeledLayers[i] = cv::Mat(height, width, CV_32S, &m_ccLabeledLayersData[i*width*height]);
        m_ccLabeledLayers[i] = cv::Mat(height, width, CV_16U, &m_ccLabeledLayersData[i*width*height]);
        // cv::Mat labels = m_ccLabeledLayers[i];
        cv::Mat _stats, _centroids;
        int numLabels = cv::connectedComponentsWithStats(
            m_occupancyLayers[i], 
            m_ccLabeledLayers[i], 
            // _stats, _centroids, //);
            stats[i], 
            centroids[i], 
            // 8, CV_32S);
            8, CV_16U);
        // _stats.copyTo(stats[i]);
        // _centroids.copyTo(centroids[i]);
        m_ccNumLabelsPerLayer[i] = numLabels;

        
        std::vector<int>& labelsToContours = m_labelsToContoursPerLayer[i];
        labelsToContours.resize(numLabels);
        std::vector<std::vector<cv::Point>>& contours = m_contoursPerLayer[i];
        // std::vector<cv::Rect>& boxes = m_bboxesPerLayer[i];
        cv::findContours(m_occupancyLayers[i], contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        
        // std::cout << "num labels in layer " << i << ": " << numLabels << std::endl;
        // std::cout << "num contours in layer " << i << ": " << contours.size() << std::endl;

        for (int j = 0; j < numLabels; ++j) labelsToContours[j] = -1; // no contour assigned
        for (int j = 0; j < contours.size(); ++j)
        {
            auto& contour = contours[j];
            if (contour.size() == 0) continue;
            cv::Point pt = contour[0];
            uint16_t label = m_ccLabeledLayers[i].at<uint16_t>(pt.y, pt.x);
            // assign this contour to the label, assumes non overlapping
            // contours which is the case for the labeled connected
            // components
            labelsToContours[label] = j;
        }


        // stats[i] = _stats;
        // centroids[i] = _centroids;
        // cv::Mat labelsColor;
        // colorizeLabels<uint16_t>(m_ccLabeledLayers[i], labelsColor);
        // for (int k=0; k<numLabels; ++k)
        // {
        //     std::cout << "label " << k << "\n";
        //     std::cout << "left " << stats.at<int32_t>(k, cv::CC_STAT_LEFT) << "\n";
        // }

        // cv::imwrite("layer_" + std::to_string(i) + ".png", m_occupancyLayers[i]);
        // cv::imwrite("contours_layer_" + std::to_string(i) + ".png", dst);
        // cv::imwrite("components_layer_" + std::to_string(i) + ".png", labelsColor);
    }
    int statsDataCount = 0;
    int centroidsDataCount = 0;


    m_ccStatsDataStarts.resize(numLayers);
    m_ccCentroidsDataStarts.resize(numLayers);
    for (int i = 0; i < numLayers; ++i)
    {
        statsDataCount += stats[i].total();
        centroidsDataCount += centroids[i].total();
        m_ccStatsDataStarts[i] = 
            (i == 0)
                ? 0 
                : (m_ccStatsDataStarts[i-1] + stats[i-1].total());
        m_ccCentroidsDataStarts[i] = 
            (i == 0) 
                ? 0 
                : (m_ccCentroidsDataStarts[i-1] + centroids[i-1].total());
    }
    m_ccStats.resize(numLayers);
    m_ccCentroids.resize(numLayers);
    m_ccStatsData.resize(statsDataCount);
    m_ccCentroidsData.resize(centroidsDataCount);
    #pragma omp parallel for
    for (int i = 0; i < numLayers; ++i)
    {
        memcpy(
            &m_ccStatsData[m_ccStatsDataStarts[i]],
            stats[i].data,
            stats[i].total() * stats[i].elemSize());
        memcpy(
            &m_ccCentroidsData[m_ccCentroidsDataStarts[i]],
            centroids[i].data,
            centroids[i].total() * centroids[i].elemSize());

        m_ccStats[i] = cv::Mat_<int32_t>((int)stats[i].total() / 5, 5,  
            &m_ccStatsData[m_ccStatsDataStarts[i]]);

        m_ccCentroids[i] = cv::Mat_<double>((int)centroids[i].total() / 2, 2,  
            &m_ccCentroidsData[m_ccCentroidsDataStarts[i]]);
    }

}

void GPUDepthmapFusion::uploadVoxelLabels()
{

    // upload cc label data

    // std::cout << "m_ccStatsData.size() " << m_ccStatsData.size() << std::endl;
    // std::cout << "m_ccCentroidsData.size() " << m_ccCentroidsData.size() << std::endl;
    m_bufCCLabeledLayersA.resize(m_ccLabeledLayersData.size());
    m_bufCCLabeledLayersB.resize(m_ccLabeledLayersData.size());
    // std::cout << "m_ccLabeledLayersData.size() " << m_ccLabeledLayersData.size() << std::endl;
    m_bufCCLabeledLayersA.bind().upload(m_ccLabeledLayersData.data(), 0, m_ccLabeledLayersData.size());

    // upload cc label stats

    m_bufCCNumLabelsPerLayer.resize(m_ccNumLabelsPerLayer.size());
    // m_bufCCStats.resize(m_ccStatsData.size());
    // m_bufCCCentroids.resize(m_ccCentroidsData.size());
    // m_bufCCStatsDataStarts.resize(m_ccStatsDataStarts.size());
    // m_bufCCCentroidsDataStarts.resize(m_ccCentroidsDataStarts.size());

    m_bufCCNumLabelsPerLayer.bind().upload(m_ccNumLabelsPerLayer.data(), 0, m_ccNumLabelsPerLayer.size());
    // m_bufCCStats.bind().upload(m_ccStatsData.data(), 0, m_ccStatsData.size());
    // m_bufCCCentroids.bind().upload(m_ccCentroidsData.data(), 0, m_ccCentroidsData.size());
    // m_bufCCStatsDataStarts.bind().upload(m_ccStatsDataStarts.data(), 0, m_ccStatsDataStarts.size());
    // m_bufCCCentroidsDataStarts.bind().upload(m_ccCentroidsDataStarts.data(), 0, m_ccCentroidsDataStarts.size());

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    m_bufBindings.bind(m_computeLabelWordsToUints.getGLProgram());
    m_computeLabelWordsToUints.use();
    m_computeLabelWordsToUints.num_items.set(m_ccLabeledLayersData.size()/2);
    m_computeLabelWordsToUints.in_data_offset.set(0);
    m_computeLabelWordsToUints.out_data_offset.set(0);    
    m_computeLabelWordsToUints.dispatch();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // static std::vector<std::vector<cv::Point3i>> intersections;
    // intersections.clear();
    // intersections.resize(m_bboxesPerLayer.size()-1);
    // #pragma omp parallel for
    // for (int i = 1; i < m_bboxesPerLayer.size(); ++i)
    // {
    //     for (int j = 0; j < m_bboxesPerLayer[i-1].size(); ++j)
    //     {
    //         cv::Rect& box1 = m_bboxesPerLayer[i-1][j];
    //         for (int k = 0; k < m_bboxesPerLayer[i].size(); ++k)
    //         {
    //             cv::Rect& box2 = m_bboxesPerLayer[i][k];
    //             cv::Rect intersection = box1 & box2;
    //             if (intersection.area()>0)
    //             {
    //                 intersections[i-1].push_back(cv::Point3i(i,j,k));
    //                 // std::cout << "intersection between layer " << i-1 << " and "<< i << " " << intersection << std::endl;
    //             }
    //         }

    //     }
    // }
    // m_occupancyGrid.size();
}

void GPUDepthmapFusion::prepareLayersConnections()
{
    if(m_occupancyLayers.size()<2) return;
    int numLayers = m_occupancyLayers.size();
    m_ccLayersConnectionsDataStarts.resize(numLayers-1);
    m_ccLayersConnections.resize(numLayers-1);
    std::vector<int> matrixSizes;
    matrixSizes.resize(numLayers-1);
    int numConnTotal = 0;
    // a connection matrix is between labeled objects of two adjacent layers
    for (int i=0; i<numLayers-1; ++i)
    {
        int numA = m_ccNumLabelsPerLayer[i];
        int numB = m_ccNumLabelsPerLayer[i+1];
        matrixSizes[i] = numA*numB;
        numConnTotal += matrixSizes[i];
    }
    if (numConnTotal > m_ccLayersConnectionsData.size())
        m_ccLayersConnectionsData.resize(numConnTotal);
    for (int i=0; i<numLayers-1; ++i)
    {
        int numA = m_ccNumLabelsPerLayer[i];
        int numB = m_ccNumLabelsPerLayer[i+1];
        m_ccLayersConnectionsDataStarts[i] =
            (i == 0)
                ? 0
                : (m_ccLayersConnectionsDataStarts[i-1] + matrixSizes[i-1]);
        // std::cout << "cv::Mat_<uint8_t>().elemSize() " << cv::Mat_<uint8_t>().elemSize() << std::endl;
        // std::cout << "step " << numB * sizeof(uint8_t) << std::endl;
        // uint8_t* ptr = &m_ccLayersConnectionsData[m_ccLayersConnectionsDataStarts[i]];
        // m_ccLayersConnections[i] = cv::Mat(
        //     numA, numB, CV_32S, ptr
        //     );
        m_ccLayersConnections[i] = cv::Mat_<uint8_t>(
            numA, numB,  
            &m_ccLayersConnectionsData[m_ccLayersConnectionsDataStarts[i]],
            numB * sizeof(uint8_t));
        // std::cout << "m_ccLayersConnections[i].elemSize() " << m_ccLayersConnections[i].elemSize() << std::endl;
        // std::cout << "m_ccLayersConnections[i] rowstep " << (&m_ccLayersConnections[i].at<uint8_t>(1,0)-&m_ccLayersConnections[i].at<uint8_t>(0,0)) << std::endl;
        // assert(numA*numB > 0);
        // if (numA*numB == 0)
        // {
        //     m_ccLayersConnections[i] = cv::Mat_<uint32_t>();
        // }
        // else
        // {

        // }
    }
    // for (int i=0; i<numLayers-1; ++i)
    // {
    //     for (int a=0; a<m_ccNumLabelsPerLayer[i]; ++a)
    //     {
    //         for (int b=0; b<m_ccNumLabelsPerLayer[i+1]; ++b)
    //         {
    //             // int idx = m_ccLayersConnectionsDataStarts[i] + a*m_ccNumLabelsPerLayer[i+1] + b;
    //             // m_ccLayersConnectionsData[idx] = 0;
    //             m_ccLayersConnections[i].at<uint8_t>(a,b) = 0;
    //         }
    //     }
    // }
    m_bufCCLayersConnectionsDataStarts.resize(m_ccLayersConnectionsDataStarts.size());
    m_bufCCLayersConnectionsDataA.resize(m_ccLayersConnectionsData.size());
    m_bufCCLayersConnectionsDataB.resize(m_ccLayersConnectionsData.size());

    m_bufCCLayersConnectionsDataStarts.bind().upload(
        m_ccLayersConnectionsDataStarts.data(), 0, m_ccLayersConnectionsDataStarts.size());
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    m_bufBindings.bind(m_computeClearLayersConnections.getGLProgram());
    m_computeClearLayersConnections.use();
    m_computeClearLayersConnections.num_items.set(m_ccLayersConnectionsData.size());
    m_computeClearLayersConnections.offset.set(0);
    m_computeClearLayersConnections.dispatch();

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void GPUDepthmapFusion::computeLayersConnectionsCPU()
{
    int numLayers = m_occupancyLayers.size();
    // std::cout << "num_layers " << numLayers << std::endl;
    // m_contoursPerLayer.clear();
    // m_bboxesPerLayer.clear();
    // m_contoursPerLayer.resize(numLayers);
    // m_bboxesPerLayer.resize(numLayers);

    #pragma omp parallel for
    for (int i=0; i<m_ccLayersConnections.size(); ++i)
    {
        int height = m_ccLabeledLayers[i].rows;
        int width = m_ccLabeledLayers[i].cols;
        auto& mat = m_ccLayersConnections[i];
        // std::cout << " m_ccNumLabelsPerLayer[i] " << m_ccNumLabelsPerLayer[i] << "\n";
        // std::cout << " m_ccNumLabelsPerLayer[i+1] " << m_ccNumLabelsPerLayer[i+1] << "\n";
        // std::cout << " m_ccLayersConnections[i].rows " << m_ccLayersConnections[i].rows << "\n";
        // std::cout << " m_ccLayersConnections[i].cols " << m_ccLayersConnections[i].cols << "\n";
        for (int labelA=0; labelA<m_ccNumLabelsPerLayer[i]; ++labelA)
        {
            for (int labelB=0; labelB<m_ccNumLabelsPerLayer[i+1]; ++labelB)
            {
                // std::cout << " write to " << labelA << "," << labelB << "\n";
                m_ccLayersConnections[i].at<uint8_t>(labelA,labelB) = 0;
            }
        }
        for (int y=0; y<height; ++y)
        {
            for (int x=0; x<width; ++x)
            {
                uint16_t labelA = m_ccLabeledLayers[i].at<uint16_t>(y,x);
                uint16_t labelB = m_ccLabeledLayers[i+1].at<uint16_t>(y,x);
                m_ccLayersConnections[i].at<uint8_t>(labelA,labelB) = 1;
            }
        }
    }

    // for (int i=0; i<m_ccLayersConnections.size(); ++i)
    // {
    //     cv::Mat img;
    //     colorizeLabels<uint8_t>(m_ccLayersConnections[i], img);
    //     cv::Mat dst;
    //     cv::resize(img, dst, cv::Size(), 32, 32, cv::INTER_NEAREST);
    //     cv::imwrite("conn_" + std::to_string(i) + ".png", dst);
    // }    
}
void GPUDepthmapFusion::computeLayersConnections()
{
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    m_bufBindings.bind(m_computeLayersConnections.getGLProgram());
    m_computeLayersConnections.use();
    m_computeLayersConnections.num_layers.set(m_voxelGridMeta.gridSize()[2]);
    m_computeLayersConnections.width.set(m_voxelGridMeta.gridSize()[0]);
    m_computeLayersConnections.layer_size.set(m_voxelGridMeta.gridSize()[0]*m_voxelGridMeta.gridSize()[1]);
    m_computeLayersConnections.num_items.set(
        (m_computeLayersConnections.num_layers.get()-1) * m_computeLayersConnections.layer_size.get());

    m_computeLayersConnections.dispatch();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    // execute compute program
}
void GPUDepthmapFusion::downloadLayersConnections()
{
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_bufBindings.bind(m_computeLayersConnectionsToChars.getGLProgram());
    m_computeLayersConnectionsToChars.use();
    m_computeLayersConnectionsToChars.num_items.set(m_ccLayersConnectionsData.size()/4);
    m_computeLayersConnectionsToChars.in_data_offset.set(0);
    m_computeLayersConnectionsToChars.out_data_offset.set(0);    
    m_computeLayersConnectionsToChars.dispatch();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_bufCCLayersConnectionsDataB
        .bind()
        .download(
            m_ccLayersConnectionsData.data(),
            0, m_ccLayersConnectionsData.size());

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // for (int i=0; i<m_ccLayersConnections.size(); ++i)
    // {
    //     cv::Mat img;
    //     colorizeLabels<uint8_t>(m_ccLayersConnections[i], img);
    //     cv::Mat dst;
    //     cv::resize(img, dst, cv::Size(), 32, 32, cv::INTER_NEAREST);
    //     cv::imwrite("conn_" + std::to_string(i) + ".png", dst);
    // }
}

void GPUDepthmapFusion::mergeLabelsAcrossLayers()
{
    // assign unique labels across all layers, simply numbered in increasing order
    // labels from lower levels will get lower unique labels
    // bottom-up pass across layers:
    //   for each layer: 
    //     assign lowest label from all connected labels from layer below
    // top-down pass across layers:
    //   for each layer:
    //     assign lowest label from all connected labels from layer above
    // now all labels are assigned to a merged unique label
    // (counting-)group them by this assigned unique-labels


    // assign unique labels across all layers, simply numbered in increasing order
    int numLayers = m_ccNumLabelsPerLayer.size();
    int numLabelsTotal = 0;
    for (int i=0; i<numLayers; ++i)
        numLabelsTotal += m_ccNumLabelsPerLayer[i];

    m_ccLabelsGlobal.resize(numLabelsTotal);
    m_ccLabelsLayer.resize(numLabelsTotal);
    m_ccLabelsLocal.resize(numLabelsTotal);
    m_ccLabelsLayerStarts.resize(numLayers);
    uint32_t next_global_label = 0;
    for (int i=0; i<numLayers; ++i)
    {
        if (i==0)
            m_ccLabelsLayerStarts[i] = 0;
        else
        {
            m_ccLabelsLayerStarts[i] = m_ccLabelsLayerStarts[i-1] + m_ccNumLabelsPerLayer[i-1];
        }
        for (int k=0; k<m_ccNumLabelsPerLayer[i]; ++k)
        {

            m_ccLabelsGlobal[next_global_label] = next_global_label;
            m_ccLabelsLayer[next_global_label] = i;
            m_ccLabelsLocal[next_global_label] = k;
            ++next_global_label;
        }
    }

    // bottom-up pass across layers:
    for (int i=0; i<numLayers-1; ++i)
    {
        int layer_a = i;
        int layer_b = i+1;
        if (layer_a < 0 || layer_a >= numLayers) continue;
        if (layer_b < 0 || layer_b >= numLayers) continue;
       
        // merge labels of layer_a into layer_b
        for (int label_b=0; label_b < m_ccNumLabelsPerLayer[layer_b]; ++label_b)
        {
            int idx = m_ccLabelsLayerStarts[layer_b] + label_b;
            for (int label_a=0; label_a < m_ccNumLabelsPerLayer[layer_a]; ++label_a)
            {
                // skip merge of layer-label=0 (background) with other layer-label>0, 
                // i.e. skip background with other stuff, 
                // but merge background with itself
                if (label_a * label_b == 0 && label_a * label_b != label_a + label_b) continue;
                if (m_ccLayersConnections[layer_a].at<uint8_t>(label_a,label_b) == 0) continue;
                // assign lower global_label to label_b in layer_b
                int other_idx = m_ccLabelsLayerStarts[layer_a] + label_a;
                if (m_ccLabelsGlobal[other_idx] < m_ccLabelsGlobal[idx])
                {
                    m_ccLabelsGlobal[idx] = m_ccLabelsGlobal[other_idx];
                }
            }
        }
    }

    // top-down pass across layers:
    for (int i=0; i<numLayers-1; ++i)
    {
        int layer_a = (numLayers-1)-i-1;
        int layer_b = (numLayers-1)-i;
        if (layer_a < 0 || layer_a >= numLayers) continue;
        if (layer_b < 0 || layer_b >= numLayers) continue;

        // merge labels of layer_b into layer_a
        for (int label_a=0; label_a < m_ccNumLabelsPerLayer[layer_a]; ++label_a)
        {
            int idx = m_ccLabelsLayerStarts[layer_a] + label_a;
            for (int label_b=0; label_b < m_ccNumLabelsPerLayer[layer_b]; ++label_b)
            {
                // skip merge of layer-label=0 (background) with other layer-label>0, 
                // i.e. skip background with other stuff, 
                // but merge background with itself
                if (label_a * label_b == 0 && label_a * label_b != label_a + label_b) continue;
                if (m_ccLayersConnections[layer_a].at<uint8_t>(label_a,label_b) == 0) continue;
                // assign lower global_label to label_b in layer_b
                int other_idx = m_ccLabelsLayerStarts[layer_b] + label_b;
                if (m_ccLabelsGlobal[other_idx] < m_ccLabelsGlobal[idx])
                {
                    m_ccLabelsGlobal[idx] = m_ccLabelsGlobal[other_idx];
                }
            }
        }
    }

    // now all labels are assigned to a merged unique label
    // (counting-)group them by this assigned unique-labels
    m_ccLabelsGlobalGrouper.group(m_ccLabelsGlobal);
    // assign new labels in increasing order
    std::vector<uint32_t> merged_labels;
    m_ccLabelsMerged.resize(m_ccLabelsGlobal.size());
    int next_merged_label_id=0;
    for(int i=0;i<m_ccLabelsGlobalGrouper.groupSizes.size();++i)
    {
        if (m_ccLabelsGlobalGrouper.groupSizes[i]==0) continue;
        for(int k=0;k<m_ccLabelsGlobalGrouper.groupSizes[i];++k)
        {
            m_ccLabelsMerged[m_ccLabelsGlobalGrouper.groupedIndices[m_ccLabelsGlobalGrouper.groupStarts[i] + k]] = next_merged_label_id;
        }
        next_merged_label_id++;
    }
    m_ccLabelsMergedGrouper.group(m_ccLabelsMerged);
}


void GPUDepthmapFusion::createCCObjects()
{

    int numObjects = m_ccLabelsMergedGrouper.groupSizes.size();
    m_ccObjects.clear();
    m_ccObjects.resize(numObjects);
    std::cout << "numObjects " << numObjects << std::endl;
    for(int i=0; i<m_ccObjects.size(); ++i)
    {
        CCObject& obj = m_ccObjects[i];
        obj.centroid.x = 0;
        obj.centroid.y = 0;
        obj.num_components = m_ccLabelsMergedGrouper.groupSizes[i];
        assert(obj.num_components > 0);
        obj.label = m_ccLabelsMerged[m_ccLabelsMergedGrouper.groupedIndices[m_ccLabelsMergedGrouper.groupStarts[i]]];

        
        obj.components.resize(obj.num_components);

        // find minima, maxima and centroid mean
        for(int k=0; k<obj.num_components; ++k)
        {
            uint idx = m_ccLabelsMergedGrouper.groupedIndices[m_ccLabelsMergedGrouper.groupStarts[i] + k];
            uint layer = m_ccLabelsLayer[idx];
            uint label_local = m_ccLabelsLocal[idx];

            assert(layer < m_labelsToContoursPerLayer.size());
            assert(label_local < m_labelsToContoursPerLayer[layer].size());
            assert(label_local < m_ccCentroids[layer].rows);

            double x = m_ccCentroids[layer].at<double>(label_local, 0);
            double y = m_ccCentroids[layer].at<double>(label_local, 1);
            auto& stats = m_ccStats[layer];
            auto stat_left   = stats(label_local, cv::CC_STAT_LEFT);
            auto stat_top    = stats(label_local, cv::CC_STAT_TOP);
            auto stat_width  = stats(label_local, cv::CC_STAT_WIDTH);
            auto stat_height = stats(label_local, cv::CC_STAT_HEIGHT);
            auto stat_area   = stats(label_local, cv::CC_STAT_AREA);
            // std::cout << "stat_left   " << stat_left << "\n";
            // std::cout << "stat_top    " << stat_top << "\n";
            // std::cout << "stat_width  " << stat_width << "\n";
            // std::cout << "stat_height " << stat_height << "\n";
            // std::cout << "stat_area   " << stat_area << "\n";
            auto right = stat_left + stat_width;
            auto bottom = stat_top + stat_height;
            obj.centroid.x += x / obj.num_components;
            obj.centroid.y += y / obj.num_components;
            if (k == 0 || stat_left < obj.min_coord.voxel[0]) obj.min_coord.voxel[0] = stat_left;
            if (k == 0 || stat_top < obj.min_coord.voxel[1])  obj.min_coord.voxel[1] = stat_top;
            if (k == 0 || layer < obj.min_coord.voxel[2])     obj.min_coord.voxel[2] = layer;

            if (k == 0 || right > obj.max_coord.voxel[0])     obj.max_coord.voxel[0] = right;
            if (k == 0 || bottom > obj.max_coord.voxel[1])    obj.max_coord.voxel[1] = bottom;
            if (k == 0 || layer > obj.max_coord.voxel[2])     obj.max_coord.voxel[2] = layer;

        }

        obj.center_coord.voxel = glm::vec3(obj.max_coord.voxel + obj.min_coord.voxel) * 0.5f;

        obj.center_coord.world = voxelCoordToWorldCoord(obj.center_coord.voxel[0], obj.center_coord.voxel[1], obj.center_coord.voxel[2]);
        obj.min_coord.world = voxelCoordToWorldCoord(obj.min_coord.voxel[0], obj.min_coord.voxel[1], obj.min_coord.voxel[2]);
        obj.max_coord.world = voxelCoordToWorldCoord(obj.max_coord.voxel[0], obj.max_coord.voxel[1], obj.max_coord.voxel[2]);

        obj.aabb_size.voxel = obj.max_coord.voxel - obj.min_coord.voxel;
        obj.aabb_size.world = obj.max_coord.world - obj.min_coord.world;

        obj.num_layers = 1+obj.aabb_size.voxel.z;

        // reserve memory for components and count contour points to reserve
        // memory for layers
        int num_contour_points = 0;
        std::vector<int> num_contour_points_per_layer;
        num_contour_points_per_layer.resize(obj.num_layers);


        for(int k=0; k<obj.num_components; ++k)
        {
            uint idx = m_ccLabelsMergedGrouper.groupedIndices[m_ccLabelsMergedGrouper.groupStarts[i] + k];
            uint layer = m_ccLabelsLayer[idx];
            uint layer_in_obj = layer - obj.min_coord.voxel.z;
            uint label_local = m_ccLabelsLocal[idx];
            int contour_idx = m_labelsToContoursPerLayer[layer][label_local];

            assert(layer < m_labelsToContoursPerLayer.size());
            assert(label_local < m_labelsToContoursPerLayer[layer].size());

            if (contour_idx >= 0)
            {
                const std::vector<cv::Point>& contour = m_contoursPerLayer[layer][contour_idx];
                num_contour_points += contour.size();
                num_contour_points_per_layer[layer_in_obj] += contour.size();

                obj.components[k].contour2d.world.resize(contour.size());
                obj.components[k].contour2d.voxel.resize(contour.size());
                obj.components[k].contour3d.world.resize(contour.size());
                obj.components[k].contour3d.voxel.resize(contour.size());
            }
        }
        
        // reserve memory for aggregated contour points
        obj.topview.points2d.world.reserve(num_contour_points);
        obj.topview.points2d.voxel.reserve(num_contour_points);
        obj.layers.resize(obj.num_layers);
        for(int layer=0; layer<obj.num_layers; ++layer)
        {
            obj.layers[layer].points2d.voxel.reserve(num_contour_points_per_layer[layer]);
            obj.layers[layer].points2d.world.reserve(num_contour_points_per_layer[layer]);
        }

        // copy contour points to obj.components and obj.layers 
        // and compute shapes for components. shape is computed here because the necessary
        // input points are here created and therefore already in hot cache
        for(int k=0; k<obj.num_components; ++k)
        {
            uint idx = m_ccLabelsMergedGrouper.groupedIndices[m_ccLabelsMergedGrouper.groupStarts[i] + k];
            uint layer = m_ccLabelsLayer[idx];
            uint layer_in_obj = layer - obj.min_coord.voxel.z;
            uint label_local = m_ccLabelsLocal[idx];
            int contour_idx = m_labelsToContoursPerLayer[layer][label_local];

            assert(layer < m_labelsToContoursPerLayer.size());
            assert(label_local < m_labelsToContoursPerLayer[layer].size());

            if (contour_idx >= 0)
            {
                const std::vector<cv::Point>& contour = m_contoursPerLayer[layer][contour_idx];
                num_contour_points += contour.size();
                num_contour_points_per_layer[layer_in_obj] += contour.size();

                for (int j = 0; j < contour.size(); ++j)
                {
                    // copy and transform contour point in 2d and 3d
                    obj.components[k].contour3d.voxel[j] = glm::ivec3(
                        contour[j].x, 
                        contour[j].y, 
                        layer);
                    // TODO: use averaged world position of voxel
                    obj.components[k].contour3d.world[j] = voxelCoordToWorldCoord(
                        contour[j].x, 
                        contour[j].y, 
                        layer);
                    obj.components[k].contour2d.voxel[j] = {
                        (float)contour[j].x, 
                        (float)contour[j].y
                    };
                    obj.components[k].contour2d.world[j] = {
                        (float)obj.components[k].contour3d.world[j].x,
                        (float)obj.components[k].contour3d.world[j].y
                    };
                    // aggregate points into layers and topview
                    obj.layers[layer_in_obj].points2d.voxel.push_back(obj.components[k].contour2d.voxel[j]);
                    obj.layers[layer_in_obj].points2d.world.push_back(obj.components[k].contour2d.world[j]);
                    obj.topview.points2d.world.push_back(obj.components[k].contour2d.world[j]);
                    obj.topview.points2d.voxel.push_back(obj.components[k].contour2d.voxel[j]);
                }
                // compute shapes of contour for world and voxel coordinates
                // if it turns out that this is to costly to do for both
                // coordinate frames, one could write code to convert the
                // shapes itself, but it wont work for circles due to
                // arbitrary aspect ratios between world and voxel. if that
                // becomes a problem, upgrade circle description to ellipse.
                obj.components[k].shapes.voxel = CCObject::MinShapes(obj.components[k].contour2d.voxel);
                obj.components[k].shapes.world = CCObject::MinShapes(obj.components[k].contour2d.world);
            }
        }

        // compute shapes for layers
        for(int layer=0; layer<obj.num_layers; ++layer)
        {
            obj.layers[layer].shapes.voxel = CCObject::MinShapes(obj.layers[layer].points2d.voxel);
            obj.layers[layer].shapes.world = CCObject::MinShapes(obj.layers[layer].points2d.world);
        }
        obj.topview.shapes.voxel = CCObject::MinShapes(obj.topview.points2d.voxel);
        obj.topview.shapes.world = CCObject::MinShapes(obj.topview.points2d.world);


        // std::cout << "obj.min_coord.voxel    " << obj.min_coord.voxel[0] << "\t" << obj.min_coord.voxel[1] << "\t" << obj.min_coord.voxel[2] << "\n";
        // std::cout << "obj.max_coord.voxel    " << obj.max_coord.voxel[0] << "\t" << obj.max_coord.voxel[1] << "\t" << obj.max_coord.voxel[2] << "\n";
        // std::cout << "obj.center_coord.voxel " << obj.center_coord.voxel[0] << "\t" << obj.center_coord.voxel[1] << "\t" << obj.center_coord.voxel[2] << "\n";
        // std::cout << "obj.center_coord.world " << obj.center_coord.world[0] << "\t" << obj.center_coord.world[1] << "\t" << obj.center_coord.world[2] << "\n";
        // std::cout << "obj.min_coord.world    " << obj.min_coord.world[0] << "\t" << obj.min_coord.world[1] << "\t" << obj.min_coord.world[2] << "\n";
        // std::cout << "obj.max_coord.world    " << obj.max_coord.world[0] << "\t" << obj.max_coord.world[1] << "\t" << obj.max_coord.world[2] << "\n";
        // std::cout << "obj.aabb_size.world    " << obj.aabb_size.world[0] << "\t" << obj.aabb_size.world[1] << "\t" << obj.aabb_size.world[2] << "\n";
        // std::cout << "obj.aabb_size_voxel    " << obj.aabb_size_voxel[0] << "\t" << obj.aabb_size_voxel[1] << "\t" << obj.aabb_size_voxel[2] << "\n";
    }

}

void GPUDepthmapFusion::objectSegmentation()
{
    labelVoxels();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    uploadVoxelLabels();
    
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    prepareLayersConnections();
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // downloadLayersConnections();
    // computeLayersConnectionsCPU();

    computeLayersConnections();
    downloadLayersConnections();

    // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    mergeLabelsAcrossLayers();
    createCCObjects();

    // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}



CCObjectTrack::CCObjectTrack()
    : initialized(false)
    , rrect_filter()
    , score_filter(
        0.5, 0.1,
        0.9, 0.1)
    , age(0)
{
}
CCObjectTrack::CCObjectTrack(const CCObject& object)
    : initialized(true)
    , rrect_filter(object.topview.shapes.world.box)
    , lastObject(object)
    , score_filter(
        0.25, 0.1,
        0.9, 0.1)
    , age(0)
{
    double new_track_score = 0.5;
    score_filter.correct(1, &new_track_score);
}

bool CCObjectTrack::isDead() const
{
    // return (age > 15) || ((age > 0.06) && (score_filter.values[0] < 0.1));
    return ((age > 0.06) && (score_filter.values[0] < 0.1));
    // return (age > 2) || (lastObject.topview.shapes.world.box.size.area() < 0.1);
}
bool CCObjectTrack::isAcceptable(const CCObjectTrackComparison& comp) const
{
    double area = comp.trackBox.size.area();
    float trackSize = comp.trackBox.size.width + comp.trackBox.size.height;
    float objectSize = comp.objectBox.size.width + comp.objectBox.size.height;
    float distUntilTrackBoundary = trackSize*0.5;
    float distUntilObjectBoundary = objectSize*0.5;
    // if (area > 0.5 && (comp.area_diff > area*0.1))
        // return false;
    if (area > 0.5 && (comp.area_diff > area*0.5))
        return false;
    if (comp.center_dist > (distUntilTrackBoundary + distUntilObjectBoundary))
        return false;
    if (comp.center_dist > 2.5)
        return false;
    // if (comp.center_dist > 5)
        // return false;
    // if (comp.area_diff > 0.5)
        // return false;
    if (comp.mean_box_point_dist > 2.5)
        return false;
    // if (comp.mean_box_point_dist > 5)
        // return false;
    return true;
}
void CCObjectTrack::advance(double dt)
{
    age += dt;
    double penalty_score = 0.0;
    // std::cout << "score before penalty " << score_filter.values[0] << std::endl;
    score_filter.predict(dt, &penalty_score);
    // std::cout << "score after penalty " << score_filter.values[0] << " dt " << dt << std::endl;
}
void CCObjectTrack::merge(double dt, const CCObject& obj, const CCObjectTrackComparison& comp)
{
    age += 1;
    rrect_filter.filter(dt, rolledRRect(obj.topview.shapes.world.box, comp.best_roll));
    lastObject = obj;
    double merge_score = 1.0;
    score_filter.correct(dt, &merge_score);
    
}

cv::RotatedRect rolledRRect(const cv::RotatedRect& rrect, int roll)
{
    if (roll % 2 == 0)
    {
        return cv::RotatedRect(
            rrect.center, rrect.size, rrect.angle + 90*roll
        );
    }
    else
    {
        return cv::RotatedRect(
            rrect.center, {rrect.size.height, rrect.size.width}, rrect.angle + 90*roll
        );
    }
}


CCObjectTrackComparison::CCObjectTrackComparison()
    : track(nullptr)
    , object(nullptr)
{}
CCObjectTrackComparison::CCObjectTrackComparison(
    const CCObjectTrack& track,
    const CCObject& object
)
    : track(&track)
    , object(&object)

{
    trackBox = track.rrect_filter.rrect;
    // trackBox = track.lastObject.topview.shapes.world.box;
    objectBox = object.topview.shapes.world.box;
    center_diff = trackBox.center - objectBox.center;
    center_dist = cv::norm(center_diff);    
    cv::Point2f ptsA[4];
    cv::Point2f ptsB[4];
    trackBox.points(ptsA);
    objectBox.points(ptsB);

    for (int roll=0; roll<4; ++roll)
    {
        cv::Point2f box_point_diffs_rolled[4];
        float box_point_dists_rolled[4];
        for (int i=0; i<4; ++i)
        {
            box_point_diffs_rolled[i] = ptsA[i] - ptsB[(i+roll)%4];
            box_point_dists_rolled[i] = cv::norm(box_point_diffs_rolled[i]);
        }
        float mean_box_point_dist_rolled = 0.25 * (
            box_point_dists_rolled[0] + 
            box_point_dists_rolled[1] + 
            box_point_dists_rolled[2] + 
            box_point_dists_rolled[3]
        );
        if ((roll == 0) || (mean_box_point_dist_rolled < mean_box_point_dist ))
        {
            best_roll = roll;
            mean_box_point_dist = mean_box_point_dist_rolled;
            for (int i=0; i<4; ++i)
            {
                box_point_diffs[i] = box_point_diffs_rolled[i];
                box_point_dists[i] = box_point_dists_rolled[i];
            }
        }
    }
    area_diff = abs(trackBox.size.area() - objectBox.size.area());
    float w_center = 0.0;
    float w_pts = 0.1;
    float w_area = 0.0;
    // score = -center_dist;
    score = -(
        + w_center * center_dist 
        + w_pts * mean_box_point_dist
        + w_area * area_diff
    );    
}

void GPUDepthmapFusion::objectTracking(float min_area)
{
    // TODO: get actual dt
    double dt = 1.0 / 30.0;

    // take objects from m_ccObjects and associate them with tracked ccObjects
    std::vector<int> assignedTracksPerObject;
    std::vector<int> numObjectsPerTrack;
    int num_objects = m_ccObjects.size();
    int num_tracks = m_ccObjectTracks.size();
    std::cout << "num_tracks " << num_tracks << std::endl;
    assignedTracksPerObject.resize(num_objects);
    numObjectsPerTrack.resize(num_tracks);
    for (int track_id=0; track_id<num_tracks; ++track_id)
    {
        numObjectsPerTrack[track_id] = 0;
    }
    static std::vector<CCObjectTrackComparison> comparisons;
    comparisons.resize(num_objects * num_tracks);

    static int frame_num = 0;
    ++frame_num;

    // assign best track for each object
    for (int object_id=0; object_id<num_objects; ++object_id)
    {
        if (false)
        { // output csv
            auto& rrect = m_ccObjects[object_id].topview.shapes.world.box;
            cv::Point2f pts[4];
            rrect.points(pts);
            std::cout << "csv"
                << ";" << frame_num
                << ";" << rrect.center.x 
                << ";" << rrect.center.y 
                << ";" << rrect.size.width 
                << ";" << rrect.size.height 
                << ";" << pts[0].x
                << ";" << pts[0].y
                << ";" << pts[1].x
                << ";" << pts[1].y
                << ";" << pts[2].x
                << ";" << pts[2].y
                << ";" << pts[3].x
                << ";" << pts[3].y
                << "\n";
        }

        assignedTracksPerObject[object_id] = -2; // mark object as ignore
        if (object_id == 0) continue; // skip background object
        if (m_ccObjects[object_id].topview.shapes.world.box.size.area() < min_area) continue; // skip very small objects, as they are likely clutter
        assignedTracksPerObject[object_id] = -1; // mark object as new track
        float bestScore = 0; // best track score for object
        for (int track_id=0; track_id<num_tracks; ++track_id)
        {
            CCObjectTrackComparison& comp = comparisons[object_id * num_tracks + track_id];
            comp = CCObjectTrackComparison(m_ccObjectTracks[track_id], m_ccObjects[object_id]);
            if (!m_ccObjectTracks[track_id].isAcceptable(comp)) continue;
            float score = comp.score;
            // std::cout << "#" << object_id << "," << track_id << ": " << score << "\n";
            if ((assignedTracksPerObject[object_id] < 0) || (score > bestScore))
            {
                bestScore = score;
                assignedTracksPerObject[object_id] = track_id;
            }
        }
        if (assignedTracksPerObject[object_id] >= 0)
        {
            numObjectsPerTrack[assignedTracksPerObject[object_id]] += 1;
        }
    }
    // tracks could be assigned multiple objects, 
    // only use the very best
    std::vector<int> assignedObjectsPerTrack;
    std::vector<float> bestObjectScorePerTrack;
    assignedObjectsPerTrack.resize(num_tracks);
    bestObjectScorePerTrack.resize(num_tracks);
    for (int j=0; j<num_tracks; ++j)
    {
        assignedObjectsPerTrack[j] = -1;
        bestObjectScorePerTrack[j] = 0;
    }
    
    for (int object_id=1; object_id<num_objects; ++object_id)
    {
        int track_id = assignedTracksPerObject[object_id];
        // skip objects which build new tracks
        if (track_id < 0) continue;
        // skip objects which are the only ones assigned to its track
        // if (numObjectsPerTrack[track_id] == 1) continue;

        CCObjectTrackComparison& comp = comparisons[object_id * num_tracks + track_id];
        // std::cout << "#" << object_id << "," << track_id << ": " << comp.score << "\n";
        if ((assignedObjectsPerTrack[track_id] == -1) || (comp.score > bestObjectScorePerTrack[track_id]))
        {
            assignedObjectsPerTrack[track_id] = object_id;
            bestObjectScorePerTrack[track_id] = comp.score;
        }
    }
    // now that the best object per track is known, reset
    // assignedTracksPerObject for objects which didn't 
    // win the competition for its track
    for (int object_id=1; object_id<num_objects; ++object_id)
    {
        int track_id = assignedTracksPerObject[object_id];
        // skip objects which build new tracks
        if (track_id < 0) continue;
        // skip the winning objects
        if (assignedObjectsPerTrack[track_id] == object_id) continue;

        // mark object to be ignored
        // assignedTracksPerObject[object_id] = -2; 
        // mark object as new track
        assignedTracksPerObject[object_id] = -1; 
    }
    // count new tracks
    int num_new_tracks = 0;
    for (int object_id=1; object_id<num_objects; ++object_id)
    {
        int track_id = assignedTracksPerObject[object_id];
        if (track_id == -1) ++num_new_tracks;
    }
    m_ccObjectTracks.resize(num_tracks + num_new_tracks);
    std::vector<bool> doTrackAdvance;
    doTrackAdvance.resize(num_tracks + num_new_tracks);
    for (int track_id=0; track_id<num_tracks + num_new_tracks; ++track_id)
    {
        doTrackAdvance[track_id] = true;
    }
    int next_new_track = num_tracks;
    int num_updated_tracks = 0;
    for (int object_id=1; object_id<num_objects; ++object_id)
    {
        int track_id = assignedTracksPerObject[object_id];
        if (track_id == -2)
        {
            // ignore object
            continue;
        }
        else if (track_id == -1)
        {
            // new track
            m_ccObjectTracks[next_new_track] = CCObjectTrack(m_ccObjects[object_id]);
            doTrackAdvance[next_new_track] = false;
            ++next_new_track;
        }
        else
        {
            // merge object to track
            CCObjectTrackComparison& comp = comparisons[object_id * num_tracks + track_id];
            m_ccObjectTracks[track_id].merge(dt, m_ccObjects[object_id], comp);
            doTrackAdvance[track_id] = false;
            ++num_updated_tracks;
        }
    }
    // advance other tracks
    for (int track_id=0; track_id<num_tracks + num_new_tracks; ++track_id)
    {
        if (doTrackAdvance[track_id])
        {
            m_ccObjectTracks[track_id].advance(dt);
        }
    }
    std::cout << "num_new_tracks " << num_new_tracks << std::endl;
    std::cout << "num_updated_tracks " << num_updated_tracks << std::endl;
    num_tracks += num_new_tracks;

    // remove dead tracks
    int num_dead_tracks = 0;
    for (int track_id=0; track_id<num_tracks; ++track_id)
    {
        if (m_ccObjectTracks[track_id].isDead())
        {
            ++num_dead_tracks;
        }
    }
    std::vector<int> remainingTrackIds;
    int num_remaining_tracks = num_tracks - num_dead_tracks;
    std::cout << "num_dead_tracks " << num_dead_tracks << std::endl;
    std::cout << "num_remaining_tracks " << num_remaining_tracks << std::endl;
    remainingTrackIds.resize(num_remaining_tracks);
    int next_remaining_track = 0;
    for (int track_id=0; track_id<num_tracks; ++track_id)
    {
        if (m_ccObjectTracks[track_id].isDead()) continue;
        remainingTrackIds[next_remaining_track] = track_id;
        ++next_remaining_track;
    }
    std::vector<int> scoreHistogram;
    scoreHistogram.resize(10);
    for(int i = 0; i < 10; ++i) scoreHistogram[i] = 0;
    int max_age = -10;
    float max_score = -10;
    for (int track_id=0; track_id<num_remaining_tracks; ++track_id)
    {
        m_ccObjectTracks[track_id] = m_ccObjectTracks[remainingTrackIds[track_id]];
        if (m_ccObjectTracks[track_id].age > max_age)
            max_age = m_ccObjectTracks[track_id].age;
        double score = m_ccObjectTracks[track_id].score_filter.values[0];
        if (score > max_score)
            max_score = score;
        if (0 <= score && score <= 1)
        {
            int score_bin = (int)(score / 0.1);
            if (score_bin < 0) score_bin = 0;
            if (score_bin > 9) score_bin = 9;
            ++scoreHistogram[score_bin];
        }
    }
    for(int i = 0; i < 10; ++i) 
    {
        std::cout << "# score in [" << (i*0.1) << "," << ((i+1)*0.1) <<  "] " << scoreHistogram[i] << std::endl;
    }
    m_ccObjectTracks.resize(next_remaining_track);
    std::cout << "m_ccObjectTracks.size() " << m_ccObjectTracks.size() << std::endl;
    std::cout << "max_age " << max_age << std::endl;
    std::cout << "max_score " << max_score << std::endl;
}

void GPUDepthmapFusion::downloadPoints()
{
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    m_points.resize(m_numItemsAfterMask);
    m_bufPointsA.bind().download(m_points.data(),0,m_numItemsAfterMask);
}
void* GPUDepthmapFusion::downloadPoints(int& count)
{
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    // count = m_numItemsAfterMask;
    count = m_bufPointsA.m_numItems;
    // count = m_depthmapsTotalElements;
    return m_bufPointsA.bind().map_ro();
}

void GPUDepthmapFusion::unmap()
{
    m_bufPointsB.unmap();
}

/*
void GPUDepthmapFusion::convertDepthmap(
    const cv::Mat_<uint16_t>& depthmap,
    float depthScale,
    float fx, float fy,
    float cx, float cy,
    cv::Matx44f transform
)
{
    m_measureTime.begin();

    glfwMakeContextCurrent(m_glWindow);

    if (m_mappedPoints != nullptr)
        m_bufPoints.unmap();

    // upload data to gpu
    // execute compute shaders:
    // - transform depthmap pixels to 3d points
    // - filter out flying points
    // download data from gpu
    // insert into point buffer
    m_bufDepthmap.resize(depthmap.total());
    m_bufPoints.resize(depthmap.total());

    m_measureTime.nextSection();
    
    m_bufDepthmap.bind().upload(depthmap.data);
    
    m_compute.depthScale(depthScale);
    m_compute.width(depthmap.cols);
    m_compute.fx(fx);
    m_compute.fy(fy);
    m_compute.cx(cx);
    m_compute.cy(cy);
    
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    // glMemoryBarrier(GL_ALL_BARRIER_BITS);

    
    m_measureTime.nextSection();

    m_bufBindings.bind(m_compute.getGLProgram());
    m_compute.use();

    for (int i=0; i<1000; ++i)
    {
        m_compute.dispatch(depthmap.cols, depthmap.rows, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }


    // static std::vector<glm::vec4> points;
    m_points.resize(depthmap.total());

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    // glMemoryBarrier(GL_ALL_BARRIER_BITS);

    m_measureTime.nextSection();

    m_bufPoints.bind().download(m_points.data());
    // m_mappedPoints = static_cast<glm::vec4*>(m_bufPoints.mapr_ro(0,1));
    // m_mappedPoints = static_cast<glm::vec4*>(m_bufPoints.bind().mapr_ro(0,m_bufPoints.m_numItems));
    // m_mappedPoints = static_cast<glm::vec4*>(m_bufPoints.map_ro());
    m_numPoints = depthmap.total();
    m_measureTime.end();


    // static std::vector<uint16_t> depths;
    // depths.resize(depthmap.total());
    // m_bufDepthmap.download(depths.data());

    // glfwPollEvents();
    // drawGL();
    
    // glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_glBufDepthmap);
    // glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint16_t) * depthmap.total(), depthmap.data, GL_STATIC_DRAW);

    // cv::Mat4f points;
    // points.create(depthmap.cols, depthmap.rows);
    // glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_glBufPoints);
    // glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(glm::vec4) * depthmap.total(), points.data);

    // uploadDepthmap();
    // computeDepthmapToPoints();
    // computeFilterFlyingPoints();
    // downloadPoints();
}
*/

void GPUDepthmapFusion::drawGL()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(m_glWindow);
}

void GPUDepthmapFusion::startGLFWEventLoop()
{
    glfwMakeContextCurrent(0);
    m_glfwThread = std::thread(
        &GPUDepthmapFusion::glfwEventLoop,
        std::ref(*this)
    );
}

void GPUDepthmapFusion::endGLFWEventLoop()
{
    m_glfwThread.join();
}
void GPUDepthmapFusion::glfwEventLoop()
{
    glfwMakeContextCurrent(m_glWindow);
    while (ros::ok() && (glfwWindowShouldClose(m_glWindow) == 0))
    {
        glfwPollEvents();
        drawGL();
    }
    ros::shutdown();
}