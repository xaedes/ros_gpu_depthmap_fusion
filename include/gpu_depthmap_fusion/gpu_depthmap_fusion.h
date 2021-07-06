#pragma once
#include <thread> 
#include <vector> 
#include <string> 
#include <memory> 

#include <opencv2/opencv.hpp>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "sensor_msgs/PointCloud2.h"

#include "gpu_depthmap_fusion/storage_buffer.h"
#include "gpu_depthmap_fusion/measure_time.h"

#include "gpu_depthmap_fusion/compute.h"

#include "gpu_depthmap_fusion/radix_grouper.h"
#include "gpu_depthmap_fusion/grid_meta.h"
#include "gpu_depthmap_fusion/uint_grouper.h"
#include "gpu_depthmap_fusion/buffer_index_bindings.h"

#include "gpu_depthmap_fusion/filter/const_local_velocity_filter.h"
#include "gpu_depthmap_fusion/filter/roll_pitch_yaw_filter.h"
#include "gpu_depthmap_fusion/filter/orientation_2d_filter.h"
#include "gpu_depthmap_fusion/filter/rotated_rect_filter.h"

cv::RotatedRect rolledRRect(const cv::RotatedRect& rrect, int roll);

class CCObject
{
public:
    template <typename TWorld, typename TVoxel>
    struct Pair
    {
        TWorld world;
        TVoxel voxel;
    };
    template <typename TWorld, typename TVoxel>
    struct VectorPair
    {
        std::vector<TWorld> world;
        std::vector<TVoxel> voxel;
    };

    struct EnclosingCircle
    {
        cv::Point2f center;
        float radius;
    };
    struct MinShapes
    {
        cv::RotatedRect box;
        EnclosingCircle circle;
        MinShapes(){}
        MinShapes(const std::vector<cv::Point>& points)
        {
            if (points.size() > 0)
            {
                box = cv::minAreaRect(points);
                cv::minEnclosingCircle(points, circle.center, circle.radius);
            }
        }
        MinShapes(const std::vector<cv::Point2f>& points)
        {
            if (points.size() > 0)
            {
                box = cv::minAreaRect(points);
                cv::minEnclosingCircle(points, circle.center, circle.radius);
            }
        }
    };

    typedef Pair<glm::vec3,glm::vec3> Pair_3f3f;
    typedef Pair<glm::vec3,glm::ivec3> Pair_3f3i;
    typedef VectorPair<cv::Point2f,cv::Point2f> VectorPair_Points;
    typedef VectorPair<glm::vec3,glm::vec3> VectorPair_3f3f;
    typedef Pair<MinShapes,MinShapes> Pair_Shapes;

    struct Component
    {
        VectorPair_Points contour2d;
        VectorPair_3f3f contour3d;
        Pair_Shapes shapes;
    };
    struct Layer
    {
        VectorPair_Points points2d;
        // VectorPair_Points convex_hull;
        Pair_Shapes shapes;
    };

    CCObject(){}
    ~CCObject(){}

    cv::Point2f centroid;
    uint32_t label;
    uint num_components;
    uint num_layers;

    Pair_3f3f center_coord;
    Pair_3f3i min_coord;
    Pair_3f3i max_coord;
    Pair_3f3i aabb_size; // aabb = axis aligned bounding box

    std::vector<Component> components;
    std::vector<Layer> layers;
    Layer topview;

};
class CCObjectTrack;
class CCObjectTrackComparison
{
public:
    CCObjectTrackComparison();
    CCObjectTrackComparison(
        const CCObjectTrack& track,
        const CCObject& object
    );
    const CCObjectTrack* track;
    const CCObject* object;
    cv::RotatedRect trackBox;
    cv::RotatedRect objectBox;
    cv::Point2f center_diff;
    cv::Point2f box_point_diffs[4];
    float center_dist;
    float box_point_dists[4];
    int best_roll;
    float mean_box_point_dist;
    float area_diff;
    float score;
};

class CCObjectTrack
{
public:
    CCObjectTrack();
    CCObjectTrack(const CCObject& object);

    bool initialized;
    CCObject lastObject;
    float age;

    RotatedRectFilter rrect_filter;
    ObservePredictFilter<double,1> score_filter;

    void advance(double dt);
    void merge(double dt, const CCObject& other, const CCObjectTrackComparison& comp);

    bool isAcceptable(const CCObjectTrackComparison& comp) const;

    bool isDead() const;
};


class GPUDepthmapFusion
{
protected:

    struct DepthmapConversion
    {
        void* depthmap;
        uint numItems;
        int width;
        int height;
        float depthScale;
        float fx; 
        float fy; 
        float cx; 
        float cy; 
        cv::Matx44f transform_world;
        cv::Matx44f transform_crop;
    };

    #pragma pack(push, 1)
    struct PointSequence
    {
        uint32_t timestampSec;
        uint32_t timestampNSec;
        uint32_t start;
        uint32_t numPoints;
        
        uint32_t padding[12];

        cv::Matx44f transform_move; 

        // to easily map PointSequence to GPU with std430 layout we add padding
        // to achieve 64byte alignment. we put the padding before the transforms
        // this way we can add more useful members without changing position of
        // transforms
        //
        // From the std430 Layout Rules: Structure alignment is the same as the
        // alignment for the biggest structure member, where three-component
        // vectors are not rounded up to the size of four-component vectors.
        // Each structure will start on this alignment, and its size will be the
        // space needed by its members, according to the previous rules, rounded
        // up to a multiple of the structure alignment. biggest element is
        // cv::Matx44f / mat4 with 16*4=64 byte alignment, i.e. 16 values of each 32bit 
        // 
    };
    #pragma pack(pop)

    struct PointSequences
    {
        int num_points_total;
        std::vector<PointSequence> sequences;
        std::vector<float> pointData;
        void clear()
        {
            num_points_total = 0;
            sequences.clear();
            pointData.clear();
        }
    };

public:
    GPUDepthmapFusion();
    ~GPUDepthmapFusion();

    void init(
        const std::string& shaderPath
    );

    void clear();

    void addDepthmap(
        // const cv::Mat& depthmap,
        const cv::Mat_<uint16_t>& depthmap,
        float depthScale,
        float fx, float fy, 
        float cx, float cy, 
        cv::Matx44f transform_world,
        cv::Matx44f transform_crop
    );

    void addPointSequence(
        const sensor_msgs::PointCloud2& pointcloud, 
        uint32_t timestampSec,
        uint32_t timestampNSec,
        cv::Matx44f transform_move
    );


    void uploadPointSequences();


    void filterNewPointSequences(float threshold, uint filter_size);

    void insertNewPointSequencesInRollbuffer();
    void rollPointSequenceRollbufferCPU(uint32_t minTimestampSec, uint32_t minTimestampNSec);
    void rollPointSequenceRollbuffer(uint32_t minTimestampSec, uint32_t minTimestampNSec);
    void selectPointSequenceTimespan(
        uint32_t minTimestampSec, 
        uint32_t minTimestampNSec, 
        uint32_t maxTimestampSec, 
        uint32_t maxTimestampNSec);
    void selectPointSequenceTimespanCPU(
        uint32_t minTimestampSec, 
        uint32_t minTimestampNSec, 
        uint32_t maxTimestampSec, 
        uint32_t maxTimestampNSec);
    
    void preparePointAndMaskBuffers();

    void insertSelectedPointSequence(
        const cv::Matx44f& tf_world_move,
        const cv::Matx44f& tf_crop_move
    );
    void transformPointSequence();

    void uploadDepthmaps();
    void convertDepthmaps();

    void checkAllPointSequenceBuffers();

    void filterFlyingPixels(uint filter_size, float threshold, bool enable_rot45);
    void cropPoints(glm::vec3 lower_bound, glm::vec3 upper_bound);
    void applyPointMask();
    void computeVoxelCoords(glm::vec3 lower_bound, glm::vec3 upper_bound, glm::vec3 cell_size);
    void downloadVoxelCoords();
    void voxelize(bool average_voxels);

    void voxelOccupancyGrid(uint32_t lifetime);

    void downloadVoxelOccupancyGrid();
    
    void objectSegmentation();

    void labelVoxels();
    void uploadVoxelLabels();
    void prepareLayersConnections();
    void computeLayersConnectionsCPU();
    void computeLayersConnections();
    void downloadLayersConnections();
    void mergeLabelsAcrossLayers();
    void createCCObjects();

    void objectTracking(float min_area);

    void downloadPoints();
    void* downloadPoints(int& count);

    glm::vec3 voxelCoordToWorldCoord(float x, float y, float z);
    glm::vec3 worldCoordToVoxelCoord(float x, float y, float z);

    void unmap();

    // void convertDepthmap(
    //     const cv::Mat_<uint16_t>& depthmap,
    //     float depthScale,
    //     float fx, float fy, 
    //     float cx, float cy, 
    //     cv::Matx44f transform
    // );
    int m_voxelGroupSize;
    bool m_enableParallelVoxelFilter;

    std::vector<cv::Mat_<uint8_t>> m_occupancyLayers;
    std::vector<uint8_t> m_occupancyGrid;
    std::vector<glm::vec4> m_points;
    std::vector<glm::vec4> m_points_voxelized;
    std::vector<uint32_t> m_voxelCoords;
    int m_numPoints;

    std::vector<cv::Mat_<uint16_t>> m_ccLabeledLayers;

    std::vector<cv::Mat_<int32_t>> m_ccStats;
    std::vector<cv::Mat_<double>> m_ccCentroids;
    
    std::vector<cv::Mat_<uint8_t>> m_ccLayersConnections;


    std::vector<uint32_t> m_ccLabelsLocal;
    std::vector<uint32_t> m_ccLabelsLayer;
    std::vector<uint32_t> m_ccLabelsGlobal;
    std::vector<uint32_t> m_ccLabelsMerged;
    UIntGrouper<uint32_t> m_ccLabelsGlobalGrouper;
    UIntGrouper<uint32_t> m_ccLabelsMergedGrouper;
    std::vector<CCObject> m_ccObjects;
    std::vector<CCObjectTrack> m_ccObjectTracks;
    
    std::vector<uint32_t> m_ccLabelsLayerStarts;

    MeasureTime m_measureTime;

    glm::vec4* m_mappedPoints;

    uint32_t m_rollBufferNumPoints;
    uint32_t m_rollBufferNumSeqs;
    uint32_t m_rollBufferSelectionPointStart;
    uint32_t m_rollBufferSelectionPointCount;
    uint32_t m_rollBufferSelectionSequenceStart;
    uint32_t m_rollBufferSelectionSequenceCount;
    uint32_t m_rollBufferEarliestTimeSec;
    uint32_t m_rollBufferEarliestTimeNSec;
    uint32_t m_rollBufferLastTimeSec;
    uint32_t m_rollBufferLastTimeNSec;

    int numCollectedPointSequencePoints() {return m_pointSequencesCollect->num_points_total;};

protected:
    void initGL();
    void cleanupGL();

    // void uploadDepthmap();
    // void downloadPoints();
    // void computeDepthmapToPoints();
    // void computeFilterFlyingPoints();

    void drawGL();

    void startGLFWEventLoop();
    void endGLFWEventLoop();
    void glfwEventLoop();

    GLFWwindow* m_glWindow;
    std::thread m_glfwThread;

    // StorageBuffer<uint16_t> m_bufDepthmap;
    // StorageBuffer<glm::vec4> m_bufPoints;
    // ComputeDepthToPoints m_compute;



    StorageBuffer<uint32_t> m_bufDepthPairs;
    StorageBuffer<uint32_t> m_bufMaskA;
    StorageBuffer<uint32_t> m_bufMaskB;
    StorageBuffer<uint32_t> m_bufMaskNewPointSequencePointsA;
    StorageBuffer<uint32_t> m_bufMaskNewPointSequencePointsB;
    StorageBuffer<glm::vec4> m_bufPointsA; // inbetween buffer for camera coordinates, final output in world coordinates
    StorageBuffer<glm::vec4> m_bufPointsB; // world coordinates
    StorageBuffer<glm::vec4> m_bufPointsC; // crop coordinates

    StorageBuffer<glm::vec4>     m_bufNewPointSequencesPoints;
    StorageBuffer<PointSequence> m_bufNewPointSequences;
    StorageBuffer<uint32_t>      m_bufNewPointSequencesPointsMaskA;
    StorageBuffer<uint32_t>      m_bufNewPointSequencesPointsMaskB;

    StorageBuffer<glm::vec4>     m_bufHistoricPointSequencePointsA; // 1-to-1 with mask and seqidcs buffers
    StorageBuffer<glm::vec4>     m_bufHistoricPointSequencePointsB; // 1-to-1 with mask and seqidcs buffers
    StorageBuffer<uint32_t>      m_bufHistoricPointSequencePointsMaskA;
    StorageBuffer<uint32_t>      m_bufHistoricPointSequencePointsMaskB;
    StorageBuffer<uint32_t>      m_bufHistoricPointSequenceSeqIdcsA;
    StorageBuffer<uint32_t>      m_bufHistoricPointSequenceSeqIdcsB;
    StorageBuffer<uint32_t>      m_bufHistoricPointSequenceRemainingPointIndices; // 1-to-1 with next buffer
    StorageBuffer<uint32_t>      m_bufHistoricPointSequenceRemainingSeqIndices;
    StorageBuffer<PointSequence> m_bufHistoricPointSequencesA;
    StorageBuffer<PointSequence> m_bufHistoricPointSequencesB;
    StorageBuffer<uint32_t>      m_bufRollbufferScratchpad; // 0: CountDiscardedRollbufferPoints, 1: CountDiscardedRollbufferSeqs
    StorageBuffer<uint32_t>      m_bufRollbufferSelected;   // 0: point start, 1: point last, 2: seq start, 3: seq last
    StorageBuffer<uint32_t>      m_bufRollbufferSelectedTransformIdcs;
    StorageBuffer<glm::mat4>     m_bufRollbufferSelectedTransformsWorld;
    StorageBuffer<glm::mat4>     m_bufRollbufferSelectedTransformsCrop;


    StorageBuffer<glm::vec2> m_bufRectifyMap;
    StorageBuffer<uint32_t> m_bufScratchpad; // 0: valid point count
    StorageBuffer<uint32_t> m_bufVoxelCoords;

    // connected components
    StorageBuffer<uint16_t> m_bufCCLabeledLayersA; // for upload to gpu
    StorageBuffer<uint32_t> m_bufCCLabeledLayersB; // for use on gpu
    StorageBuffer<uint32_t> m_bufCCNumLabelsPerLayer; 
    StorageBuffer<uint32_t> m_bufCCStats; 
    StorageBuffer<uint32_t> m_bufCCCentroids; 
    StorageBuffer<uint16_t> m_bufCCStatsDataStarts; 
    StorageBuffer<uint16_t> m_bufCCCentroidsDataStarts; 
    StorageBuffer<uint32_t> m_bufCCLayersConnectionsDataStarts;
    StorageBuffer<uint32_t> m_bufCCLayersConnectionsDataA; // for use on gpu
    StorageBuffer<uint8_t> m_bufCCLayersConnectionsDataB;  // for download from gpu

    StorageBuffer<uint32_t> m_bufVoxelOccupancyA;
    StorageBuffer<uint32_t> m_bufVoxelOccupancyB;
    StorageBuffer<uint32_t> m_bufHistoricVoxelOccupancyA;
    StorageBuffer<uint32_t> m_bufHistoricVoxelOccupancyB;

    BufferIndexBindings m_bufBindings;

    std::vector<DepthmapConversion> m_depthmaps;
    int m_depthmapsTotalElements;
    int m_numPointsTotal;
    uint m_numItemsAfterMask;
    
    uint m_numFilteredPointSeqPoints;



    PointSequences m_pointSequencesA;
    PointSequences m_pointSequencesB;

    void swapPointSequencesBuffers();
    PointSequences* m_pointSequencesCollect;
    PointSequences* m_pointSequencesUpload;



    ComputeConvertDepthmapToPoints    m_computeConvertDepthmapToPoints;
    ComputeFilterFlyingPixels         m_computeFilterFlyingPixels;

    ComputeSetUints                                   m_computeInitNewPointSequencesPointsMask;
    ComputeFilterPointSequence                        m_computeFilterNewPointSequencePoints;
    ComputeTransferData                               m_computeRollbufferTransferOldPoints;
    ComputeTransferData                               m_computeRollbufferTransferOldPointsMask;
    ComputeTransferData                               m_computeRollbufferTransferOldSeqs;
    ComputeTransferData                               m_computeRollbufferTransferOldSeqIdcs;
    ComputeTransferData                               m_computeRollbufferTransferNewPoints;
    ComputeTransferData                               m_computeRollbufferTransferNewPointsMask;
    ComputeTransferData                               m_computeRollbufferTransferNewSeqs;
    ComputeSetUints                                   m_computeRollbufferSetNewPointsSeqIdcs;

    ComputeRollbufferCountDiscardedPoints             m_computeRollbufferCountDiscardedPoints;
    ComputeRollbufferCountDiscardedSeqs               m_computeRollbufferCountDiscardedSeqs;
    ComputeRollbufferRemainingPointsIndices           m_computeRollbufferRemainingPointsIndices;
    ComputeRollbufferRemainingPointsCopyAndUpdate     m_computeRollbufferRemainingPointsCopyAndUpdate;
    ComputeRollbufferRemainingSeqsIndices             m_computeRollbufferRemainingSeqsIndices;
    ComputeTransferDataFrom                           m_computeRollbufferRemainingSeqsCopy;
    ComputeRollbufferSelectTimespanPoints             m_computeRollbufferSelectTimespanPoints;
    ComputeRollbufferSelectTimespanSequences          m_computeRollbufferSelectTimespanSequences;
    ComputeTransferData                               m_computeRollbufferTransferSelectedMask;
    ComputeRollbufferTransferSelectedTransformIndices m_computeRollbufferTransferSelectedTransformIndices;
    ComputeRollbufferTransferSelectedTransforms       m_computeRollbufferTransferSelectedTransforms;

    ComputeTransformPointsIndirect                    m_computeTransformPointSequenceWorld;
    ComputeTransformPointsIndirect                    m_computeTransformPointSequenceCrop;

    ComputeCropPoints                 m_computeCropPoints;
    ComputeApplyPointMask             m_computeApplyPointMask;
    ComputeVoxelCoords                m_computeVoxelCoords;

    ComputeZeroUints                  m_computeVoxelGridOccupancyClear;
    ComputeVoxelGridOccupancyOfPoints m_computeVoxelGridOccupancyOfPoints;
    ComputeUintsToChars               m_computeVoxelGridOccupancyToChars;
    
    ComputeZeroUints                  m_computeHistoricVoxelGridOccupancyClear;
    ComputeDecrementUints             m_computeHistoricVoxelGridOccupancyDec;
    ComputeMaxWithUintsTimesScalar    m_computeHistoricVoxelGridOccupancySet;

    ComputeWordsToUints               m_computeLabelWordsToUints;
    ComputeZeroUints                  m_computeClearLayersConnections;
    ComputeLayersConnections          m_computeLayersConnections;
    ComputeUintsToChars               m_computeLayersConnectionsToChars;

    typedef RadixGrouper<uint32_t, cv::Vec<uint32_t,4>> VoxelGrouper;
    typedef GridMeta<glm::vec4, glm::uvec3, glm::bvec3, 3, uint32_t> VoxelGridMeta;
    VoxelGrouper m_voxelRadixGrouper;
    VoxelGridMeta m_voxelGridMeta;

    std::vector<std::vector<std::vector<cv::Point>>> m_contoursPerLayer;
    std::vector<std::vector<int>> m_labelsToContoursPerLayer;

    std::vector<std::vector<cv::Rect>> m_bboxesPerLayer;
    std::vector<uint32_t> m_ccNumLabelsPerLayer;
    std::vector<uint32_t> m_ccStatsDataStarts;
    std::vector<uint32_t> m_ccCentroidsDataStarts;
    std::vector<uint16_t> m_ccLabeledLayersData;
    std::vector<int32_t>  m_ccStatsData;
    std::vector<double>   m_ccCentroidsData;
    std::vector<uint32_t> m_ccLayersConnectionsDataStarts;
    std::vector<uint8_t>  m_ccLayersConnectionsData;

    std::string m_shaderPath;

};
