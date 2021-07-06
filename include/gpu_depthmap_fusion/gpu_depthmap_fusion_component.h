#pragma once
#include <vector>
#include <memory>
#include <thread>
#include <mutex>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>

#include <tf2_ros/transform_listener.h>

#include "ros_component/ros_component.h"
#include "ros_topic_sync/advanced_sync_policy.h"

#include "gpu_depthmap_fusion/gpu_depthmap_fusion.h"

#include "std_msgs/Float32.h"
#include "std_msgs/Int32.h"
#include "std_msgs/Bool.h"

#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/CameraInfo.h"

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"


typedef ros_topic_sync::AdvancedSyncPolicy<
    sensor_msgs::Image,
    sensor_msgs::Image
> SyncPolicy2;

typedef ros_topic_sync::AdvancedSyncPolicy<
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image
> SyncPolicy3;

typedef ros_topic_sync::AdvancedSyncPolicy<
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image
> SyncPolicy4;

typedef ros_topic_sync::AdvancedSyncPolicy<
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image
> SyncPolicy5;

typedef ros_topic_sync::AdvancedSyncPolicy<
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image,
    sensor_msgs::Image
> SyncPolicy6;

class GPUDepthmapFusionComponent : public mft_ros::component::RosComponent
{
public:
    GPUDepthmapFusionComponent();
    virtual ~GPUDepthmapFusionComponent();

protected:
    virtual void onInit();

    std::mutex m_mutexCallbackN;
    std::mutex m_mutexCallbackPointSequence;
    std::mutex m_mutexCollectedDepthFrames;

    void processDepthmaps(
        const std::vector<sensor_msgs::Image::ConstPtr>& msgs);

    void callbackResample(const ros::TimerEvent& event);

    void callbackN(
        const std::vector<sensor_msgs::Image::ConstPtr>& msgs);
    void callback1(
        const sensor_msgs::Image::ConstPtr msg0);
    void callback2(
        const sensor_msgs::Image::ConstPtr msg0, 
        const sensor_msgs::Image::ConstPtr msg1);
    void callback3(
        const sensor_msgs::Image::ConstPtr msg0, 
        const sensor_msgs::Image::ConstPtr msg1,
        const sensor_msgs::Image::ConstPtr msg2);
    void callback4(
        const sensor_msgs::Image::ConstPtr msg0, 
        const sensor_msgs::Image::ConstPtr msg1,
        const sensor_msgs::Image::ConstPtr msg2,
        const sensor_msgs::Image::ConstPtr msg3);
    void callback5(
        const sensor_msgs::Image::ConstPtr msg0, 
        const sensor_msgs::Image::ConstPtr msg1,
        const sensor_msgs::Image::ConstPtr msg2,
        const sensor_msgs::Image::ConstPtr msg3,
        const sensor_msgs::Image::ConstPtr msg4);
    void callback6(
        const sensor_msgs::Image::ConstPtr msg0, 
        const sensor_msgs::Image::ConstPtr msg1,
        const sensor_msgs::Image::ConstPtr msg2,
        const sensor_msgs::Image::ConstPtr msg3,
        const sensor_msgs::Image::ConstPtr msg4,
        const sensor_msgs::Image::ConstPtr msg5);
    
    void callbackCameraInfo(
        const sensor_msgs::CameraInfo::ConstPtr msg
    );

    void callbackCfgFilterFlyingPixelsThreshold(
        const std_msgs::Float32::ConstPtr msg    
    );
    void callbackCfgFilterFlyingPixelsSize(
        const std_msgs::Int32::ConstPtr msg    
    );
    void callbackCfgFilterFlyingPixelsEnableRot45(
        const std_msgs::Bool::ConstPtr msg    
    );

    void callbackPointSequence(
        const sensor_msgs::PointCloud2::ConstPtr msg
    );

    cv::Matx44f lookupTransform(const std::string &target_frame, const std::string &source_frame, const ros::Time& time);
    cv::Matx44f lookupTransform(const std::string &target_frame, const std::string &source_frame);

    bool canTransform(const std::string &target_frame, const std::string &source_frame, const ros::Time& time);
    bool canTransform(const std::string &target_frame, const std::string &source_frame);

    const sensor_msgs::CameraInfo& getCameraInfo(const std::string& frame_id) const;
    bool isCameraInfoAvailable(const std::string& frame_id) const;


    void visualizeObjects(const ros::Time& time);


    std::shared_ptr<GPUDepthmapFusion> m_fusion;
    int m_numSubcribedDepthmaps;
    int m_numSubcribedPointSequences;
    ros::Publisher m_pub;
    ros::Publisher m_pubViz;
    ros::Publisher m_pubVizPoints;

    ros::Subscriber m_subFilterFlyingPixelsThreshold;
    ros::Subscriber m_subFilterFlyingPixelsSize;
    ros::Subscriber m_subFilterFlyingPixelsEnableRot45;
    std::vector<ros::Subscriber> m_subsPointSequences;

    ros::Subscriber m_singleSub;
    std::vector<ros::Subscriber> m_subCameraInfo;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> m_syncSubs;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy2>> m_sync2;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy3>> m_sync3;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy4>> m_sync4;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy5>> m_sync5;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy6>> m_sync6;

    tf2_ros::Buffer m_tfBuffer;
    std::shared_ptr<tf2_ros::TransformListener> m_tfListener;

    std::string m_worldFrame;
    std::string m_cropFrame;
    std::string m_moveFrame;
    std::string m_exportFrame;

    std::vector<sensor_msgs::CameraInfo> m_cameraInfos;

    std::vector<sensor_msgs::Image::ConstPtr> m_collectedDepthFrames;

    bool m_resample;
    float m_resampleRate;
    ros::Timer m_resampleTimer;

    float m_depthScale;

    float m_flyingPixelFilterThreshold;
    int m_flyingPixelFilterSize;
    bool m_flyingPixelFilterEnableRot45;

    float m_psFlyingPixelFilterThreshold;
    int m_psfFlyingPixelFilterSize;

    
    float m_cropMinX, m_cropMinY, m_cropMinZ, m_cropMaxX, m_cropMaxY, m_cropMaxZ;
    float m_voxelMinX, m_voxelMinY, m_voxelMinZ, m_voxelMaxX, m_voxelMaxY, m_voxelMaxZ;
    float m_voxelSizeX, m_voxelSizeY, m_voxelSizeZ;
    bool m_voxelEnableAverage;

    float m_radiusMinX, m_radiusMinY, m_radiusMinZ, m_radiusMaxX, m_radiusMaxY, m_radiusMaxZ;
    float m_radiusFilterRadius;
    int m_radiusFilterMinNeighbors;

    int m_voxelOccupancyLifetime;

    float m_objectMinArea;

    float m_ptSeqTimespan;

    bool m_enableFlyingPixelsFilter;
    bool m_enableVoxelFilter;
    bool m_enableRadiusFilter;

    bool m_enableDebugOutput;

    bool m_enableParallelAddPointcloud;
    bool m_enableParallelTransformations;
    bool m_enableParallelCrop;
    bool m_enableParallelVoxelFilter;
    bool m_enableParallelRadiusFilter;

    std::string m_shaderPath;

};