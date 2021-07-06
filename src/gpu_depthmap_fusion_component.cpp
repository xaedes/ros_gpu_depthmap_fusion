#include "gpu_depthmap_fusion/gpu_depthmap_fusion_component.h"
#include "gpu_depthmap_fusion/tf2_cv.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "std_msgs/Header.h"

#include "geometry_msgs/Point.h"
#include "std_msgs/ColorRGBA.h"

#include <chrono>
#include <cv_bridge/cv_bridge.h>

#include "gpu_depthmap_fusion/gl_compute_test.h"
#include "ros_viz/create_marker.h"
#include "ros_viz/marker_set_values.h"
#include "ros_viz/wire_frame.h"

GPUDepthmapFusionComponent::GPUDepthmapFusionComponent()
{}

GPUDepthmapFusionComponent::~GPUDepthmapFusionComponent()
{}

cv::Matx44f GPUDepthmapFusionComponent::lookupTransform(const std::string &target_frame, const std::string &source_frame,
    const ros::Time& time)
{
    geometry_msgs::TransformStamped transformStamped = m_tfBuffer.lookupTransform(
        target_frame, 
        source_frame,
        time
    );            
    cv::Matx44f tfMat;
    tf2::transformToCv(transformStamped.transform, tfMat);
    return tfMat;
}

cv::Matx44f GPUDepthmapFusionComponent::lookupTransform(const std::string &target_frame, const std::string &source_frame)
{
    return lookupTransform(target_frame, source_frame, ros::Time(0));
}

bool GPUDepthmapFusionComponent::canTransform(const std::string &target_frame, const std::string &source_frame,
    const ros::Time& time)
{
    return m_tfBuffer.canTransform(
        target_frame, 
        source_frame,
        time,
        ros::Duration(0)
    );            
}

bool GPUDepthmapFusionComponent::canTransform(const std::string &target_frame, const std::string &source_frame)
{
    return canTransform(target_frame, source_frame, ros::Time(0));
}

void GPUDepthmapFusionComponent::callbackN(
    const std::vector<sensor_msgs::Image::ConstPtr>& msgs)
{
    if (m_resample)
    {
        const std::lock_guard<std::mutex> lock(m_mutexCollectedDepthFrames);
        for (int i=0; i<msgs.size(); ++i)
        {
            m_collectedDepthFrames.push_back(msgs[i]);
        }
    }
    else
    {
        processDepthmaps(msgs);
    }
}

void GPUDepthmapFusionComponent::callbackResample(const ros::TimerEvent& event)
{
    if (m_resample)
    {
        std::vector<sensor_msgs::Image::ConstPtr> msgs;
        {
            const std::lock_guard<std::mutex> lock(m_mutexCollectedDepthFrames);

            for (int i=0; i<m_collectedDepthFrames.size(); ++i)
            {
                msgs.push_back(m_collectedDepthFrames[i]);
            }
            m_collectedDepthFrames.clear();
        }
        processDepthmaps(msgs);
    }
}

void GPUDepthmapFusionComponent::processDepthmaps(
    const std::vector<sensor_msgs::Image::ConstPtr>& msgs)
{
    const std::lock_guard<std::mutex> lock(m_mutexCallbackN);

    // std::cout << "callbackN" << std::endl;
    std_msgs::Header header;
    auto t0 = std::chrono::high_resolution_clock::now();
    // m_fusion.clear();
    auto t1 = std::chrono::high_resolution_clock::now();
    int numAdded = 0;

    m_fusion->clear();
    m_fusion->m_measureTime.beginFrame();

    ros::Time overallLatestTime;

    // std::cout << "msgs.size() " << msgs.size() << std::endl;
    std::vector<cv_bridge::CvImageConstPtr> images; // store cv_bridge images so they can be uploaded later
    for (int i=0; i<msgs.size(); ++i)
    {
        const auto& msg = msgs[i];
        if (!msg) continue;

        const auto& frame_id = msg->header.frame_id;

        if (!isCameraInfoAvailable(frame_id)) continue;

        auto cvImgPtr = cv_bridge::toCvShare(msg);
        if (!cvImgPtr) continue;
        const cv::Mat& img = cvImgPtr->image;
        images.push_back(cvImgPtr);

        if(canTransform(m_worldFrame, frame_id, msg->header.stamp)
            && canTransform(m_cropFrame, frame_id, msg->header.stamp))
        {
            cv::Matx44f transform_world = lookupTransform(m_worldFrame, frame_id, msg->header.stamp);
            cv::Matx44f transform_crop = lookupTransform(m_cropFrame, frame_id, msg->header.stamp);

            const auto& camInfo = getCameraInfo(frame_id);
            float focal_length_x = camInfo.K[0+0*3];
            float focal_length_y = camInfo.K[1+1*3];
            float center_x = camInfo.K[2+0*3];
            float center_y = camInfo.K[2+1*3];

            m_fusion->addDepthmap(
                img, 
                m_depthScale, 
                focal_length_x, focal_length_y, 
                center_x, center_y, 
                transform_world,
                transform_crop
            );

            ++numAdded;
            header = msgs[i]->header;

            if (msg->header.stamp > overallLatestTime)
                overallLatestTime = msg->header.stamp;
        }
        
    }
    
    m_fusion->m_measureTime.endFrame();

    auto t2 = std::chrono::high_resolution_clock::now();
    if ((numAdded>0) || (m_fusion->numCollectedPointSequencePoints()>0))
    {
        {
            // not sure if lock is necessary here
            const std::lock_guard<std::mutex> point_seq_lock(m_mutexCallbackPointSequence);
            m_fusion->uploadPointSequences();
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
        }
        m_fusion->filterNewPointSequences(m_psFlyingPixelFilterThreshold, m_psfFlyingPixelFilterSize);

        m_fusion->insertNewPointSequencesInRollbuffer();

        ros::Duration timespan(m_ptSeqTimespan);
        // std::cout << " m_fusion->m_rollBufferLastTimeSec " << m_fusion->m_rollBufferLastTimeSec << std::endl;
        // std::cout << " m_fusion->m_rollBufferLastTimeNSec " << m_fusion->m_rollBufferLastTimeNSec << std::endl;
        
        ros::Time latestTime, earliestTime;

        if (m_fusion->m_rollBufferLastTimeSec != 0|| m_fusion->m_rollBufferLastTimeNSec != 0)
        {
            latestTime = ros::Time(
                m_fusion->m_rollBufferLastTimeSec, 
                m_fusion->m_rollBufferLastTimeNSec);
            earliestTime = latestTime - timespan;
            if (latestTime > overallLatestTime)
                overallLatestTime = latestTime;
        }
        m_fusion->rollPointSequenceRollbufferCPU(
            earliestTime.sec, earliestTime.nsec
        );

        if(canTransform(m_worldFrame, m_moveFrame)
            && canTransform(m_cropFrame, m_moveFrame))
        {
            cv::Matx44f tf_world_move = lookupTransform(m_worldFrame, m_moveFrame);
            cv::Matx44f tf_crop_move = lookupTransform(m_cropFrame, m_moveFrame);
            m_fusion->selectPointSequenceTimespanCPU(
                earliestTime.sec, earliestTime.nsec, 
                latestTime.sec, latestTime.nsec
            );
            m_fusion->preparePointAndMaskBuffers();
            m_fusion->insertSelectedPointSequence(
                tf_world_move, 
                tf_crop_move
            );
            m_fusion->transformPointSequence();
        }
        else
        {
            // tf not available, --> no selection
            // selection is cleared by m_fusion->clear() at each frame begin
            // just prepare buffers for depthmaps
            m_fusion->preparePointAndMaskBuffers();
        }

        header.stamp = overallLatestTime;

        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        auto t3 = std::chrono::high_resolution_clock::now();

        m_fusion->uploadDepthmaps();

        glMemoryBarrier(GL_ALL_BARRIER_BITS);


        auto t4 = std::chrono::high_resolution_clock::now();
        
        m_fusion->convertDepthmaps();
        // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        // // m_fusion.transformPoints(m_cropFrame, tf_crop);

        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        auto t5 = std::chrono::high_resolution_clock::now();

        m_fusion->filterFlyingPixels(m_flyingPixelFilterSize, m_flyingPixelFilterThreshold, m_flyingPixelFilterEnableRot45);
        // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        
        auto t6 = std::chrono::high_resolution_clock::now();

        m_fusion->cropPoints(
            glm::vec3(m_cropMinX, m_cropMinY, m_cropMinZ), 
            glm::vec3(m_cropMaxX, m_cropMaxY, m_cropMaxZ));

        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        auto t7 = std::chrono::high_resolution_clock::now();

        m_fusion->applyPointMask();
        // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        // // m_fusion.finishCrop(m_cropFrame);

        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        auto t8 = std::chrono::high_resolution_clock::now();

        if (m_enableVoxelFilter)
        {
            m_fusion->computeVoxelCoords(
                glm::vec3(m_voxelMinX,m_voxelMinY,m_voxelMinZ),
                glm::vec3(m_voxelMaxX,m_voxelMaxY,m_voxelMaxZ),
                glm::vec3(m_voxelSizeX,m_voxelSizeY,m_voxelSizeZ));
            m_fusion->downloadVoxelCoords();
        }
        // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        // // m_fusion.finishCrop(m_cropFrame);
        // 
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        auto t9 = std::chrono::high_resolution_clock::now();

        // if (!m_enableVoxelFilter || m_voxelEnableAverage)
        {
            // glMemoryBarrier(GL_ALL_BARRIER_BITS);
            m_fusion->downloadPoints();
        }
        // glMemoryBarrier(GL_ALL_BARRIER_BITS);

        // glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        // // m_fusion.finishCrop(m_cropFrame);
        // 
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        auto t10 = std::chrono::high_resolution_clock::now();

        if (m_enableVoxelFilter)
        {
            m_fusion->voxelize(m_voxelEnableAverage);
        }

        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        auto t11 = std::chrono::high_resolution_clock::now();

        if (m_enableVoxelFilter)
        {
            m_fusion->voxelOccupancyGrid(m_voxelOccupancyLifetime);
        }

        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        auto t12 = std::chrono::high_resolution_clock::now();

        if (m_enableVoxelFilter)
        {
            m_fusion->downloadVoxelOccupancyGrid();
        }

        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        auto t13 = std::chrono::high_resolution_clock::now();

        if (m_enableVoxelFilter)
        {
            m_fusion->objectSegmentation();
        }

        auto t14 = std::chrono::high_resolution_clock::now();

        if (m_enableVoxelFilter)
        {
            m_fusion->objectTracking(m_objectMinArea);
        }

        auto t15 = std::chrono::high_resolution_clock::now();

        if (m_enableVoxelFilter)
        {
            visualizeObjects(overallLatestTime);



            {
                std::vector<float> points;

                for(int i=0; i < m_fusion->m_ccCentroids.size(); ++i)
                {
                    for(int k=1; k < m_fusion->m_ccCentroids[i].rows; ++k)
                    {
                        auto worldCoord = m_fusion->voxelCoordToWorldCoord(
                            m_fusion->m_ccCentroids[i](k, 0),
                            m_fusion->m_ccCentroids[i](k, 1),
                            i
                        );
                        points.push_back(worldCoord[0]);
                        points.push_back(worldCoord[1]);
                        points.push_back(worldCoord[2]);
                    }
                }

                static sensor_msgs::PointCloud2 pcl_out;
                pcl_out.height = 1;
                pcl_out.width = points.size() / 3;
                pcl_out.fields.resize(3);
                pcl_out.fields[0].name = "x";
                pcl_out.fields[0].offset = 0;
                pcl_out.fields[0].datatype = pcl_out.fields[0].FLOAT32;
                pcl_out.fields[0].count = 1;
                pcl_out.fields[1].name = "y";
                pcl_out.fields[1].offset = 4;
                pcl_out.fields[1].datatype = pcl_out.fields[1].FLOAT32;
                pcl_out.fields[1].count = 1;
                pcl_out.fields[2].name = "z";
                pcl_out.fields[2].offset = 8;
                pcl_out.fields[2].datatype = pcl_out.fields[2].FLOAT32;
                pcl_out.fields[2].count = 1;
                // pcl_out.fields[3].name = "w";
                // pcl_out.fields[3].offset = 8;
                // pcl_out.fields[3].datatype = pcl_out.fields[3].FLOAT32;
                // pcl_out.fields[3].count = 1;
                pcl_out.is_bigendian = (__BYTE_ORDER != __LITTLE_ENDIAN);
                pcl_out.point_step = sizeof(glm::vec3);
                pcl_out.row_step = pcl_out.point_step * pcl_out.width;
                pcl_out.is_dense = true;
                size_t size = pcl_out.height * pcl_out.row_step;
                pcl_out.data.resize(size);

                pcl_out.header = header;
                // pcl_out.header.frame_id = "cam1_depth_optical_frame";
                pcl_out.header.frame_id = m_worldFrame;
                memcpy(&pcl_out.data[0], points.data(), size);

                m_pubVizPoints.publish(pcl_out);
            }
        }

        auto t16 = std::chrono::high_resolution_clock::now();

        int count = 0;
        void* ptr = 0;
        // ptr = m_fusion->downloadPoints(count);
        if (m_enableVoxelFilter)
        {
            count = m_fusion->m_points_voxelized.size();
            ptr = m_fusion->m_points_voxelized.data();
        }
        else
        {
            count = m_fusion->m_points.size();
            ptr = m_fusion->m_points.data();
        }
        auto vec4ptr = static_cast<cv::Vec4f*>(ptr);
        // cv::Mat4f pointMat(480,848,static_cast<cv::Vec4f*>(ptr));
        // std::cout << ptr << std::endl;
        
        // if (m_enableVoxelFilter)
        //     m_fusion.voxelize(
        //         cv::Vec3f(m_voxelMinX,m_voxelMinY,m_voxelMinZ),
        //         cv::Vec3f(m_voxelMaxX,m_voxelMaxY,m_voxelMaxZ),
        //         cv::Vec3f(m_voxelSizeX,m_voxelSizeY,m_voxelSizeZ),
        //         m_voxelEnableAverage);
        // else
        //     m_fusion.bypassVoxelFilter();
        // if (m_enableRadiusFilter)
        //     m_fusion.radiusFilter(
        //         cv::Vec3f(m_radiusMinX,m_radiusMinY,m_radiusMinZ),
        //         cv::Vec3f(m_radiusMaxX,m_radiusMaxY,m_radiusMaxZ),
        //         m_radiusFilterRadius, 
        //         m_radiusFilterMinNeighbors);
        // else
        //     m_fusion.bypassRadiusFilter();
        

        if (ptr != nullptr && count > 0)
        {
            static sensor_msgs::PointCloud2 pcl_out;
            pcl_out.height = 1;
            pcl_out.width = count;
            pcl_out.fields.resize(3);
            pcl_out.fields[0].name = "x";
            pcl_out.fields[0].offset = 0;
            pcl_out.fields[0].datatype = pcl_out.fields[0].FLOAT32;
            pcl_out.fields[0].count = 1;
            pcl_out.fields[1].name = "y";
            pcl_out.fields[1].offset = 4;
            pcl_out.fields[1].datatype = pcl_out.fields[1].FLOAT32;
            pcl_out.fields[1].count = 1;
            pcl_out.fields[2].name = "z";
            pcl_out.fields[2].offset = 8;
            pcl_out.fields[2].datatype = pcl_out.fields[2].FLOAT32;
            pcl_out.fields[2].count = 1;
            // pcl_out.fields[3].name = "w";
            // pcl_out.fields[3].offset = 8;
            // pcl_out.fields[3].datatype = pcl_out.fields[3].FLOAT32;
            // pcl_out.fields[3].count = 1;
            pcl_out.is_bigendian = (__BYTE_ORDER != __LITTLE_ENDIAN);
            pcl_out.point_step = sizeof(glm::vec4);
            pcl_out.row_step = pcl_out.point_step * pcl_out.width;
            pcl_out.is_dense = true;
            size_t size = pcl_out.height * pcl_out.row_step;
            pcl_out.data.resize(size);

            pcl_out.header = header;
            // pcl_out.header.frame_id = "cam1_depth_optical_frame";
            pcl_out.header.frame_id = m_worldFrame;
            memcpy(&pcl_out.data[0], ptr, size);

            m_pub.publish(pcl_out);

            // m_fusion->unmap();
        }
        

        auto t17 = std::chrono::high_resolution_clock::now();

        if (m_enableDebugOutput)
        {
            // m_fusion->m_measureTime.print("GPUDepthmapFusion::addDepthmap");


            auto d0 = std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 ).count();
            auto d1 = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
            auto d2 = std::chrono::duration_cast<std::chrono::microseconds>( t3 - t2 ).count();
            auto d3 = std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
            auto d4 = std::chrono::duration_cast<std::chrono::microseconds>( t5 - t4 ).count();
            auto d5 = std::chrono::duration_cast<std::chrono::microseconds>( t6 - t5 ).count();
            auto d6 = std::chrono::duration_cast<std::chrono::microseconds>( t7 - t6 ).count();
            auto d7 = std::chrono::duration_cast<std::chrono::microseconds>( t8 - t7 ).count();
            auto d8 = std::chrono::duration_cast<std::chrono::microseconds>( t9 - t8 ).count();
            auto d9 = std::chrono::duration_cast<std::chrono::microseconds>( t10 - t9 ).count();
            auto d10 = std::chrono::duration_cast<std::chrono::microseconds>( t11 - t10 ).count();
            auto d11 = std::chrono::duration_cast<std::chrono::microseconds>( t12 - t11 ).count();
            auto d12 = std::chrono::duration_cast<std::chrono::microseconds>( t13 - t12 ).count();
            auto d13 = std::chrono::duration_cast<std::chrono::microseconds>( t14 - t13 ).count();
            auto d14 = std::chrono::duration_cast<std::chrono::microseconds>( t15 - t14 ).count();
            auto d15 = std::chrono::duration_cast<std::chrono::microseconds>( t16 - t15 ).count();
            auto d16 = std::chrono::duration_cast<std::chrono::microseconds>( t17 - t16 ).count();
            auto dt = std::chrono::duration_cast<std::chrono::microseconds>( t17 - t0 ).count();

            std::cout << "num point clouds " << numAdded << std::endl;
            std::cout << "num points total " << count << std::endl;
            // std::cout << "num points after crop " << m_fusion.m_points_cropped.size() << std::endl;
            // std::cout << "num points after voxel filter "     << m_fusion.m_points_voxelized.size() << std::endl;
            // std::cout << "num points after radius filter "      << m_fusion.m_points_filtered.size() << std::endl;
            
            // std::cout << "duration0 " << d0 << " clear" << std::endl;
            std::cout << "duration1 " << d1 <<   " add" << std::endl;
            std::cout << "duration2 " << d2 <<   " upload and process point sequence" << std::endl;
            std::cout << "duration3 " << d3 <<   " upload depthmap" << std::endl;
            std::cout << "duration4 " << d4 <<   " convert depthmap" << std::endl;
            std::cout << "duration5 " << d5 <<   " filter flying pixels" << std::endl;
            std::cout << "duration6 " << d6 <<   " crop" << std::endl;
            std::cout << "duration7 " << d7 <<   " apply mask" << std::endl;
            std::cout << "duration8 " << d8 <<   " voxel coords" << std::endl;
            std::cout << "duration9 " << d9 <<   " download points" << std::endl;
            std::cout << "duration10 " << d10 << " voxelize" << std::endl;
            std::cout << "duration11 " << d11 << " voxel occupancy grid" << std::endl;
            std::cout << "duration12 " << d12 << " download occupancy grid" << std::endl;
            std::cout << "duration13 " << d13 << " object segmentation" << std::endl;
            std::cout << "duration14 " << d14 << " object tracking" << std::endl;
            std::cout << "duration15 " << d15 << " pub obj" << std::endl;
            std::cout << "duration16 " << d16 << " pub" << std::endl;
            std::cout << "duration total " << dt << " " << std::endl;
            std::cout << std::endl;
        }
    }
}
void GPUDepthmapFusionComponent::visualizeObjects(const ros::Time& time)
{
    visualization_msgs::MarkerArray markers;

    markers.markers.push_back(createMarker(
        time, m_worldFrame, "all", 0,
        visualization_msgs::Marker::LINE_LIST,
        visualization_msgs::Marker::DELETEALL));

    // visualize layer component centroids
    if(false)
    {

        visualization_msgs::Marker centroidsMarker;
        centroidsMarker = createMarker(
            time, m_worldFrame, "gpucc", 1,
            visualization_msgs::Marker::POINTS,
            visualization_msgs::Marker::ADD,
            0, 0, 0,
            0, 0, 0, 1,
            5, 5, 0,
            0, 1, 1, 1
        );
        for(int i=0; i < m_fusion->m_ccCentroids.size(); ++i)
        {
            for(int k=1; k < m_fusion->m_ccCentroids[i].rows; ++k)
            {
                auto worldCoord = m_fusion->voxelCoordToWorldCoord(
                    m_fusion->m_ccCentroids[i](k, 0),
                    m_fusion->m_ccCentroids[i](k, 1),
                    i);
                geometry_msgs::Point point;
                point.x = worldCoord[0];
                point.y = worldCoord[1];
                point.z = worldCoord[2];
                centroidsMarker.points.push_back(point);
                std_msgs::ColorRGBA color;
                color.r = 0.0;
                color.g = 1.0;
                color.b = 1.0;
                color.a = 1.0;
                centroidsMarker.colors.push_back(color);
            }
        }
        if (centroidsMarker.points.size()>0)
            markers.markers.push_back(centroidsMarker);
    }
    // visualize layer component ids
    if(false)
    {
        visualization_msgs::Marker textsMarker;
        textsMarker = createMarker(
            time, m_worldFrame, "gpucc", 10,
            visualization_msgs::Marker::TEXT_VIEW_FACING,
            visualization_msgs::Marker::ADD,
            0, 0, 0,
            0, 0, 0, 1,
            0, 0, 0.1,
            1, 1, 0, 1
        );
        for(int i=0; i < m_fusion->m_ccCentroids.size(); ++i)
        {
            for(int k=0; k < m_fusion->m_ccCentroids[i].rows; ++k)
            {
                auto worldCoord = m_fusion->voxelCoordToWorldCoord(
                    m_fusion->m_ccCentroids[i](k, 0),
                    m_fusion->m_ccCentroids[i](k, 1),
                    i);
                textsMarker.id++;
                textsMarker.pose.position.x = worldCoord[0];
                textsMarker.pose.position.y = worldCoord[1];
                textsMarker.pose.position.z = worldCoord[2];
                textsMarker.text = "# " + std::to_string(i) + " id " + std::to_string(k);
                markers.markers.push_back(visualization_msgs::Marker(textsMarker));
            }
        }
    }

    // visualize layer connections
    if(false)
    {
        visualization_msgs::Marker linesMarker;
        linesMarker = createMarker(
            time, m_worldFrame, "gpu_cc_layer_connections", 0,
            visualization_msgs::Marker::LINE_LIST,
            visualization_msgs::Marker::ADD,
            0, 0, 0,
            0, 0, 0, 1,
            0.02, 0, 0,
            1, 170/255., 0, 1
        );        

        linesMarker.points.clear();
        linesMarker.colors.clear();
        for(int i=0; i < m_fusion->m_ccCentroids.size()-1; ++i)
        {
            int numA = m_fusion->m_ccCentroids[i].rows;
            int numB = m_fusion->m_ccCentroids[i+1].rows;
            for(int a=1; a<numA; ++a)
            {
                auto worldCoordA = m_fusion->voxelCoordToWorldCoord(
                    m_fusion->m_ccCentroids[i](a, 0),
                    m_fusion->m_ccCentroids[i](a, 1),
                    i);
                geometry_msgs::Point pointA;
                pointA.x = worldCoordA[0];
                pointA.y = worldCoordA[1];
                pointA.z = worldCoordA[2];
                for(int b=1; b<numB; ++b)
                {
                    auto worldCoordB = m_fusion->voxelCoordToWorldCoord(
                        m_fusion->m_ccCentroids[i+1](b, 0),
                        m_fusion->m_ccCentroids[i+1](b, 1),
                        i+1);
                    if (m_fusion->m_ccLayersConnections[i].at<uint8_t>(a,b) != 0)
                    {
                        geometry_msgs::Point pointB;
                        pointB.x = worldCoordB[0];
                        pointB.y = worldCoordB[1];
                        pointB.z = worldCoordB[2];
                        linesMarker.points.push_back(pointA);
                        linesMarker.points.push_back(pointB);
                        linesMarker.colors.push_back(linesMarker.color);
                        linesMarker.colors.push_back(linesMarker.color);
                    }
                }
            }
        }
        if (linesMarker.points.size()>0)
            markers.markers.push_back(linesMarker);
    }



    // visualize object with label and bbox
    if(false)
    {
        visualization_msgs::Marker textsMarker;
        visualization_msgs::Marker boxLinesMarker;
        textsMarker = createMarker(
            time, m_worldFrame, "gpu_cc_obj_labels", 0,
            visualization_msgs::Marker::TEXT_VIEW_FACING,
            visualization_msgs::Marker::ADD,
            0, 0, 0,
            0, 0, 0, 1,
            0, 0, 0.1,
            1, 1, 0, 1
        );
        boxLinesMarker = createMarker(
            time, m_worldFrame, "gpu_cc_obj_boxes", 0,
            visualization_msgs::Marker::LINE_LIST,
            visualization_msgs::Marker::ADD,
            0, 0, 0,
            0, 0, 0, 1,
            0.0025, 0, 0,
            1, 1, 0, 1
        );

        float a=0.5;
        // WireFrame wireBox;
        // wireBox.addPoints(
        //     {-a,-a,-a},
        //     {+a,-a,-a},
        //     {+a,+a,-a},
        //     {-a,+a,-a},
        //     {-a,-a,+a},
        //     {+a,-a,+a},
        //     {+a,+a,+a},
        //     {-a,+a,+a}
        // );
        // wireBox.addQuad({0,1,2,3}); // bottom
        // wireBox.addQuad({4,5,6,7}); // top
        // // connect bottom and top
        // wireBox.addLine({0,4}); wireBox.addLine({1,5}); 
        // wireBox.addLine({2,6}); wireBox.addLine({3,7}); 
        std::vector<cv::Vec3f> boxPoints;
        // connect bottom quad
        boxPoints.push_back(cv::Vec3f(-a,-a,-a)); // new line
        boxPoints.push_back(cv::Vec3f(+a,-a,-a));
        boxPoints.push_back(cv::Vec3f(+a,-a,-a)); // new line
        boxPoints.push_back(cv::Vec3f(+a,+a,-a));
        boxPoints.push_back(cv::Vec3f(+a,+a,-a)); // new line
        boxPoints.push_back(cv::Vec3f(-a,+a,-a));
        boxPoints.push_back(cv::Vec3f(-a,+a,-a)); // new line
        boxPoints.push_back(cv::Vec3f(-a,-a,-a));
        // connect top quad
        boxPoints.push_back(cv::Vec3f(-a,-a,+a)); // new line
        boxPoints.push_back(cv::Vec3f(+a,-a,+a));
        boxPoints.push_back(cv::Vec3f(+a,-a,+a)); // new line
        boxPoints.push_back(cv::Vec3f(+a,+a,+a));
        boxPoints.push_back(cv::Vec3f(+a,+a,+a)); // new line
        boxPoints.push_back(cv::Vec3f(-a,+a,+a));
        boxPoints.push_back(cv::Vec3f(-a,+a,+a)); // new line
        boxPoints.push_back(cv::Vec3f(-a,-a,+a));
        // connect top and bottom points
        boxPoints.push_back(cv::Vec3f(-a,-a,-a)); // new line
        boxPoints.push_back(cv::Vec3f(-a,-a,+a));
        boxPoints.push_back(cv::Vec3f(+a,-a,-a)); // new line
        boxPoints.push_back(cv::Vec3f(+a,-a,+a));
        boxPoints.push_back(cv::Vec3f(+a,+a,-a)); // new line
        boxPoints.push_back(cv::Vec3f(+a,+a,+a));
        boxPoints.push_back(cv::Vec3f(-a,+a,-a)); // new line
        boxPoints.push_back(cv::Vec3f(-a,+a,+a));
        // add reverse lines
        boxPoints.resize(boxPoints.size()*2);
        for (int i=boxPoints.size()/2; i<boxPoints.size();++i)
        {
            if (i % 2 == 0)
                boxPoints[i] = boxPoints[i-boxPoints.size()/2+1];
            else
                boxPoints[i] = boxPoints[i-boxPoints.size()/2-1];
        }

        for(int i=1; i < m_fusion->m_ccObjects.size(); ++i)
        {
            auto& obj = m_fusion->m_ccObjects[i];

            // visualization_msgs::MarkerArray linesMarker;
            // m_ccLayersConnections
            // m_ccStats
            // m_ccCentroids

            textsMarker.id++;
            textsMarker.text = "# " + std::to_string(obj.label);
            markers.markers.push_back(visualization_msgs::Marker(textsMarker));

            // boxLinesMarker.id++;
            // boxLinesMarker.points.resize(boxPoints.size());
            // boxLinesMarker.colors.resize(boxPoints.size());
            // for (int i=0; i < boxPoints.size(); ++i)
            // {
            //     boxLinesMarker.points[i].x = obj.center_coord.world[0] + boxPoints[i][0] * obj.aabb_size.world[0];
            //     boxLinesMarker.points[i].y = obj.center_coord.world[1] + boxPoints[i][1] * obj.aabb_size.world[1];
            //     boxLinesMarker.points[i].z = obj.center_coord.world[2] + boxPoints[i][2] * obj.aabb_size.world[2];
            // }
            // for(int i=0; i < boxLinesMarker.points.size(); ++i)
            // {
            //     boxLinesMarker.colors[i] = boxLinesMarker.color;
            // }
            // markers.markers.push_back(visualization_msgs::Marker(boxLinesMarker));

        }
    }
    // visualize object minimal bbox
    if (false)
    {
        visualization_msgs::Marker minboxLinesMarker;

        minboxLinesMarker = createMarker(
            time, m_worldFrame, "gpu_cc_obj_min_boxes", 0,
            visualization_msgs::Marker::LINE_LIST,
            visualization_msgs::Marker::ADD,
            0, 0, 0,
            0, 0, 0, 1,
            0.02, 0, 0,
            // 0.0025, 0, 0,
            0, 0.2, 1, 1
        );

        std::vector<int> boxLinePoints{
            // connect bottom quad
            0, 1,
            1, 2,
            2, 3,
            3, 0,
            // connect top quad
            4, 5,
            5, 6,
            6, 7,
            7, 4,
            // connect top and bottom points
            0, 4,
            1, 5,
            2, 6,
            3, 7
        };
        // add reverse lines
        boxLinePoints.resize(boxLinePoints.size()*2);
        for (int i=boxLinePoints.size()/2; i<boxLinePoints.size();++i)
        {
            if (i % 2 == 0)
                boxLinePoints[i] = boxLinePoints[i-boxLinePoints.size()/2+1];
            else
                boxLinePoints[i] = boxLinePoints[i-boxLinePoints.size()/2-1];
        }

        for(int i=1; i < m_fusion->m_ccObjects.size(); ++i)
        {
            auto& obj = m_fusion->m_ccObjects[i];
            if (cv::norm(obj.topview.shapes.world.box.center) < 1) continue;
            
            cv::Point2f rrect_points[4];
            rrect_points; // The order is bottomLeft, topLeft, topRight, bottomRight. 
            obj.topview.shapes.world.box.points(rrect_points);

            cv::Point3f box_points[8];
            for (int i = 0; i < 4; i++)
            {
                box_points[i]   = cv::Point3f(rrect_points[i].x, rrect_points[i].y, obj.min_coord.world.z);
                box_points[4+i] = cv::Point3f(rrect_points[i].x, rrect_points[i].y, obj.max_coord.world.z);
            }

            minboxLinesMarker.id++;
            minboxLinesMarker.points.resize(boxLinePoints.size());
            // minboxLinesMarker.colors.resize(boxLinePoints.size());
            for (int i=0; i < boxLinePoints.size(); ++i)
            {
                minboxLinesMarker.points[i].x = box_points[boxLinePoints[i]].x;
                minboxLinesMarker.points[i].y = box_points[boxLinePoints[i]].y;
                minboxLinesMarker.points[i].z = box_points[boxLinePoints[i]].z;
            }
            // for(int i=0; i < minboxLinesMarker.points.size(); ++i)
            // {
            //     minboxLinesMarker.colors[i] = minboxLinesMarker.color;
            // }
            markers.markers.push_back(visualization_msgs::Marker(minboxLinesMarker));

        }
    }
    // visualize 
    if (false)
    {
        visualization_msgs::Marker contourLinesMarker;
        contourLinesMarker = createMarker(
            time, m_worldFrame, "gpu_cc_obj_component_contours", 0,
            visualization_msgs::Marker::LINE_LIST,
            visualization_msgs::Marker::ADD,
            0, 0, 0,
            0, 0, 0, 1,
            0.0025, 0, 0,
            1, 1, 0, 1
        );
        contourLinesMarker.points.clear();
        contourLinesMarker.colors.clear();
        for(int i=1; i < m_fusion->m_ccObjects.size(); ++i)
        {
            auto& obj = m_fusion->m_ccObjects[i];
            for (int k=0; k<obj.components.size(); ++k)
            {
                const auto& component = obj.components[k];
                const auto& contour = component.contour3d.world;
                geometry_msgs::Point pointA;
                geometry_msgs::Point pointB;
                for (int j=0; j<contour.size(); ++j)
                {
                    pointA.x = contour[j].x;
                    pointA.y = contour[j].y;
                    pointA.z = contour[j].z;
                    pointB.x = contour[(j+1)%contour.size()].x;
                    pointB.y = contour[(j+1)%contour.size()].y;
                    pointB.z = contour[(j+1)%contour.size()].z;
                    contourLinesMarker.points.push_back(pointA);
                    contourLinesMarker.points.push_back(pointB);
                    contourLinesMarker.colors.push_back(contourLinesMarker.color);
                    contourLinesMarker.colors.push_back(contourLinesMarker.color);
                }
            }
        }
        markers.markers.push_back(visualization_msgs::Marker(contourLinesMarker));

    }
    // visualize object minimal bbox
    {
        visualization_msgs::Marker minboxLinesMarker;

        minboxLinesMarker = createMarker(
            time, m_worldFrame, "gpu_cc_obj_track_min_boxes", 0,
            visualization_msgs::Marker::LINE_LIST,
            visualization_msgs::Marker::ADD,
            0, 0, 0,
            0, 0, 0, 1,
            0.02, 0, 0,
            // 0.0025, 0, 0,
            1, 1, 0, 1
        );

        std::vector<int> boxLinePoints{
            // connect bottom quad
            0, 1,
            1, 2,
            2, 3,
            3, 0,
            // connect top quad
            4, 5,
            5, 6,
            6, 7,
            7, 4,
            // connect top and bottom points
            0, 4,
            1, 5,
            2, 6,
            3, 7
        };
        // add reverse lines
        boxLinePoints.resize(boxLinePoints.size()*2);
        for (int i=boxLinePoints.size()/2; i<boxLinePoints.size();++i)
        {
            if (i % 2 == 0)
                boxLinePoints[i] = boxLinePoints[i-boxLinePoints.size()/2+1];
            else
                boxLinePoints[i] = boxLinePoints[i-boxLinePoints.size()/2-1];
        }

        for(int i=0; i < m_fusion->m_ccObjectTracks.size(); ++i)
        {
            if (i > 500) break;
            auto& track = m_fusion->m_ccObjectTracks[i];
            // skip when too close at origin
            if (cv::norm(track.rrect_filter.rrect.center) < 1) continue;
            double score = track.score_filter.values[0];
            if (score < 0.65) continue; 
            auto& obj = track.lastObject;
            cv::Point2f rrect_points[4];

            // to understand the order of points:
            // split the area around rrect center of angle=0 in 4 mathematical quadrants:
            // Q1(x>0,y>0),Q2(x<0,y>0),Q3(x<0,y<0),Q4(x>0,y<0)
            // The order of points is then Q2,Q3,Q4,Q1
            // applying angle is rotation in mathematical positive direction (ccw)
            // obj.topview.shapes.world.box.points(rrect_points);
            track.rrect_filter.rrect.points(rrect_points);

            cv::Point3f box_points[8];
            for (int i = 0; i < 4; i++)
            {
                box_points[i]   = cv::Point3f(rrect_points[i].x, rrect_points[i].y, obj.min_coord.world.z);
                box_points[4+i] = cv::Point3f(rrect_points[i].x, rrect_points[i].y, obj.max_coord.world.z);
            }

            minboxLinesMarker.id++;
            minboxLinesMarker.points.resize(boxLinePoints.size());
            // minboxLinesMarker.colors.resize(boxLinePoints.size());
            for (int i=0; i < boxLinePoints.size(); ++i)
            {
                minboxLinesMarker.points[i].x = box_points[boxLinePoints[i]].x;
                minboxLinesMarker.points[i].y = box_points[boxLinePoints[i]].y;
                minboxLinesMarker.points[i].z = box_points[boxLinePoints[i]].z;
            }
            // for(int i=0; i < minboxLinesMarker.points.size(); ++i)
            // {
            //     minboxLinesMarker.colors[i] = minboxLinesMarker.color;
            // }
            minboxLinesMarker.color.a = score;
            markers.markers.push_back(visualization_msgs::Marker(minboxLinesMarker));

        }
    }

    m_pubViz.publish(markers);
}


void GPUDepthmapFusionComponent::callbackCfgFilterFlyingPixelsThreshold(
    const std_msgs::Float32::ConstPtr msg    
)
{
    if (!msg) return;
    m_flyingPixelFilterThreshold = msg->data;
}
void GPUDepthmapFusionComponent::callbackCfgFilterFlyingPixelsSize(
    const std_msgs::Int32::ConstPtr msg    
)
{
    if (!msg) return;
    m_flyingPixelFilterSize = msg->data;
}
void GPUDepthmapFusionComponent::callbackCfgFilterFlyingPixelsEnableRot45(
    const std_msgs::Bool::ConstPtr msg    
)
{
    if (!msg) return;
    m_flyingPixelFilterEnableRot45 = msg->data;
}
void GPUDepthmapFusionComponent::callbackPointSequence(
    const sensor_msgs::PointCloud2::ConstPtr msg
)
{
    if (!msg) return;
    
    const std::lock_guard<std::mutex> lock(m_mutexCallbackPointSequence);

    const auto& frame_id = msg->header.frame_id;
    // std::cout << "receive other pcl with frame " << frame_id << std::endl;

    if(canTransform(m_moveFrame, frame_id, msg->header.stamp))
    {
        cv::Matx44f transform_move = lookupTransform(m_moveFrame, frame_id, msg->header.stamp);
        // std::cout << "can transform " << std::endl;

        m_fusion->addPointSequence(
            *msg, 
            msg->header.stamp.sec, msg->header.stamp.nsec,
            transform_move);
    }
    // else std::cout << "can not transform " << std::endl;
}
void GPUDepthmapFusionComponent::callbackCameraInfo(
    const sensor_msgs::CameraInfo::ConstPtr msg
)
{
    if (!msg) return;
    // only use camera infos for known frame ids
    // if (!m_fusion.frameIds.contains(msg->header.frame_id)) return;
    // if (isCameraInfoAvailable(msg->header.frame_id)) return;

    int id = 0;
    // int id = m_fusion.frameIds.map(msg->header.frame_id);
    if (m_cameraInfos.size() < id+1)
        m_cameraInfos.resize(id+1);
    m_cameraInfos[id] = *msg;
}

bool GPUDepthmapFusionComponent::isCameraInfoAvailable(const std::string& frame_id) const
{
    // if (!m_fusion.frameIds.contains(frame_id)) return false;
    // int id = m_fusion.frameIds.mapConst(frame_id);
    int id = 0;
    if (m_cameraInfos.size() < id+1) return false;
    return true;
}
const sensor_msgs::CameraInfo& GPUDepthmapFusionComponent::getCameraInfo(const std::string& frame_id) const
{
    assert(isCameraInfoAvailable(frame_id));
    // int id = m_fusion.frameIds.mapConst(frame_id);
    int id = 0;
    return m_cameraInfos[id];
}


void GPUDepthmapFusionComponent::callback1(
    const sensor_msgs::Image::ConstPtr msg0)
{
    // std::vector<sensor_msgs::Image::ConstPtr> msgs{msg0,msg0,msg0,msg0,msg0,msg0}; 
    // std::vector<sensor_msgs::Image::ConstPtr> msgs{msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0,msg0}; 
    std::vector<sensor_msgs::Image::ConstPtr> msgs{msg0};
    callbackN(msgs);
}

void GPUDepthmapFusionComponent::callback2(
    const sensor_msgs::Image::ConstPtr msg0, 
    const sensor_msgs::Image::ConstPtr msg1)
{
    std::vector<sensor_msgs::Image::ConstPtr> msgs{msg0, msg1};
    callbackN(msgs);
}

void GPUDepthmapFusionComponent::callback3(
    const sensor_msgs::Image::ConstPtr msg0, 
    const sensor_msgs::Image::ConstPtr msg1,
    const sensor_msgs::Image::ConstPtr msg2)
{
    std::vector<sensor_msgs::Image::ConstPtr> msgs{msg0, msg1, msg2};
    callbackN(msgs);
}

void GPUDepthmapFusionComponent::callback4(
    const sensor_msgs::Image::ConstPtr msg0, 
    const sensor_msgs::Image::ConstPtr msg1,
    const sensor_msgs::Image::ConstPtr msg2,
    const sensor_msgs::Image::ConstPtr msg3)
{
    std::vector<sensor_msgs::Image::ConstPtr> msgs{msg0, msg1, msg2, msg3};
    callbackN(msgs);
}

void GPUDepthmapFusionComponent::callback5(
    const sensor_msgs::Image::ConstPtr msg0, 
    const sensor_msgs::Image::ConstPtr msg1,
    const sensor_msgs::Image::ConstPtr msg2,
    const sensor_msgs::Image::ConstPtr msg3,
    const sensor_msgs::Image::ConstPtr msg4)
{
    std::vector<sensor_msgs::Image::ConstPtr> msgs{msg0, msg1, msg2, msg3, msg4};
    callbackN(msgs);
}

void GPUDepthmapFusionComponent::callback6(
    const sensor_msgs::Image::ConstPtr msg0, 
    const sensor_msgs::Image::ConstPtr msg1,
    const sensor_msgs::Image::ConstPtr msg2,
    const sensor_msgs::Image::ConstPtr msg3,
    const sensor_msgs::Image::ConstPtr msg4,
    const sensor_msgs::Image::ConstPtr msg5)
{
    std::vector<sensor_msgs::Image::ConstPtr> msgs{msg0, msg1, msg2, msg3, msg4, msg5};
    callbackN(msgs);
}


void GPUDepthmapFusionComponent::onInit()
{
    ros::NodeHandle& nh = getNodeHandle();

    m_fusion = std::make_shared<GPUDepthmapFusion>();
    
    m_tfListener = std::make_shared<tf2_ros::TransformListener>(m_tfBuffer);
    
    int in_dm_queue_size;
    int in_ps_queue_size;
    int out_queue_size;
    nh.param<int>("in_dm_queue_size", in_dm_queue_size, 1);
    nh.param<int>("in_ps_queue_size", in_ps_queue_size, 1);
    nh.param<int>("out_queue_size", out_queue_size, 1);
    nh.param<int>("num_maps", m_numSubcribedDepthmaps, 1);
    nh.param<int>("num_seqs", m_numSubcribedPointSequences, 1);

    nh.param<bool>("enable_debug_output", m_enableDebugOutput, true);

    nh.param<std::string>("world_frame", m_worldFrame, "world");
    nh.param<std::string>("crop_frame", m_cropFrame, "crop");
    nh.param<std::string>("move_frame", m_moveFrame, "move");
    nh.param<std::string>("obj_export_frame", m_exportFrame, "world");


    nh.param<float>("depth_scale", m_depthScale, 0.001);

    nh.param<float>("resample_rate", m_resampleRate, 0);
    m_resample = (m_resampleRate > 0);

    nh.param<float>("crop_min_x", m_cropMinX, -1);
    nh.param<float>("crop_min_y", m_cropMinY, -1);
    nh.param<float>("crop_min_z", m_cropMinZ, -1);
    nh.param<float>("crop_max_x", m_cropMaxX, +1);
    nh.param<float>("crop_max_y", m_cropMaxY, +1);
    nh.param<float>("crop_max_z", m_cropMaxZ, +1);

    nh.param<bool>("enable_flyingpixels_filter", m_enableFlyingPixelsFilter, true);
    nh.param<bool>("enable_voxel_filter", m_enableVoxelFilter, true);
    nh.param<bool>("enable_radius_filter", m_enableRadiusFilter, true);

    // nh.param<int>("flyingpixels_group_size", m_fusion.m_flyingpixelsGroupSize, 1024);
    nh.param<int>("voxel_group_size", m_fusion->m_voxelGroupSize, 1024);
    // nh.param<int>("radius_group_size", m_fusion.m_radiusGroupSize, 1024);

    nh.param<float>("flyingpixels_filter_threshold", m_flyingPixelFilterThreshold, 0.5);
    nh.param<int>("flyingpixels_filter_size", m_flyingPixelFilterSize, 1);
    nh.param<bool>("flyingpixels_filter_enable_rot45", m_flyingPixelFilterEnableRot45, 1);
    
    nh.param<float>("point_sequence_flying_pixel_filter_threshold", m_psFlyingPixelFilterThreshold, 0.5);
    nh.param<int>("point_sequence_flying_pixel_filter_size", m_psfFlyingPixelFilterSize, 1);



    
    nh.param<float>("point_sequence_aggregation_timespan", m_ptSeqTimespan, 0.1);

    nh.param<float>("voxel_filter_min_x", m_voxelMinX, -1);
    nh.param<float>("voxel_filter_min_y", m_voxelMinY, -1);
    nh.param<float>("voxel_filter_min_z", m_voxelMinZ, -1);
    nh.param<float>("voxel_filter_max_x", m_voxelMaxX, +1);
    nh.param<float>("voxel_filter_max_y", m_voxelMaxY, +1);
    nh.param<float>("voxel_filter_max_z", m_voxelMaxZ, +1);

    nh.param<float>("voxel_filter_size_x", m_voxelSizeX, 0.1);
    nh.param<float>("voxel_filter_size_y", m_voxelSizeY, 0.1);
    nh.param<float>("voxel_filter_size_z", m_voxelSizeZ, 0.1);
    nh.param<bool>("voxel_filter_enable_average", m_voxelEnableAverage, true);
    
    nh.param<int>("voxel_occupancy_lifetime", m_voxelOccupancyLifetime, 1);
    nh.param<float>("object_min_area", m_objectMinArea, 0.2*0.2);

    nh.param<float>("radius_filter_min_x", m_radiusMinX, -1);
    nh.param<float>("radius_filter_min_y", m_radiusMinY, -1);
    nh.param<float>("radius_filter_min_z", m_radiusMinZ, -1);
    nh.param<float>("radius_filter_max_x", m_radiusMaxX, +1);
    nh.param<float>("radius_filter_max_y", m_radiusMaxY, +1);
    nh.param<float>("radius_filter_max_z", m_radiusMaxZ, +1);
    
    nh.param<float>("radius_filter_radius", m_radiusFilterRadius, 0.1);
    nh.param<std::string>("shader_path", m_shaderPath, "");

    m_fusion->init(m_shaderPath);

    // nh.param<bool>("enable_parallel_add_pointcloud", m_fusion.m_enableParallelAddPointcloud, false);
    // nh.param<bool>("enable_parallel_transformations", m_fusion.m_enableParallelTransformations, false);
    // nh.param<bool>("enable_parallel_crop", m_fusion.m_enableParallelCrop, false);
    // nh.param<bool>("enable_parallel_voxel_filter", m_fusion.m_enableParallelVoxelFilter, false);
    // nh.param<bool>("enable_parallel_radius_filter", m_fusion.m_enableParallelRadiusFilter, false);

    m_pub = nh.advertise<sensor_msgs::PointCloud2>("out/Points", out_queue_size);

    m_pubViz = nh.advertise<visualization_msgs::MarkerArray>("out/Viz", out_queue_size);
    m_pubVizPoints = nh.advertise<sensor_msgs::PointCloud2>("out/VizPcl", out_queue_size);

    if (m_resample)
    {
        m_resampleTimer = nh.createTimer(
            ros::Duration(1.0 / m_resampleRate),
            &GPUDepthmapFusionComponent::callbackResample,
            this
        );

    }
    for(int i=0; i<m_numSubcribedPointSequences; ++i)
    {
        m_subsPointSequences.push_back(
            nh.subscribe(
                "in/Pointcloud/" + std::to_string(i),
                in_ps_queue_size,
                &GPUDepthmapFusionComponent::callbackPointSequence,
                this
            )
        );
    }
    if (m_numSubcribedDepthmaps == 1)
    {
        m_singleSub = nh.subscribe(
            "in/Depthmap/0",
            in_dm_queue_size,
            &GPUDepthmapFusionComponent::callback1,
            this
        );
    }
    if (m_numSubcribedDepthmaps > 1)
    {
        for (int i=0; i<m_numSubcribedDepthmaps; ++i)
        {
            m_syncSubs.push_back(
                std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(
                    nh, 
                    "in/Depthmap/" + std::to_string(i),
                    in_dm_queue_size
                )
            );
        }
        if (m_numSubcribedDepthmaps == 2)
        {
            m_sync2 = std::make_shared<message_filters::Synchronizer<SyncPolicy2>>(
                SyncPolicy2(&nh),
                *m_syncSubs[0],
                *m_syncSubs[1]
            );
            m_sync2->m_config[0].trigger  = true;
            m_sync2->m_config[0].delay    = false;
            m_sync2->m_config[0].optional = false;
            m_sync2->m_config[0].clear    = true;

            m_sync2->m_config[1].trigger  = false;
            m_sync2->m_config[1].delay    = false;
            m_sync2->m_config[1].optional = false;
            m_sync2->m_config[1].clear    = true;

            m_sync2->registerCallback(
                boost::bind(
                    &GPUDepthmapFusionComponent::callback2, 
                    this, 
                    _1, _2
                )
            );
        }
        if (m_numSubcribedDepthmaps == 3)
        {
            m_sync3 = std::make_shared<message_filters::Synchronizer<SyncPolicy3>>(
                SyncPolicy3(&nh),
                *m_syncSubs[0],
                *m_syncSubs[1],
                *m_syncSubs[2]
            );

            m_sync3->m_config[0].trigger  = true;
            m_sync3->m_config[0].delay    = false;
            m_sync3->m_config[0].optional = false;
            m_sync3->m_config[0].clear    = true;

            for(int i=1; i<m_numSubcribedDepthmaps; ++i)
            {
                m_sync3->m_config[i].trigger  = false;
                m_sync3->m_config[i].delay    = false;
                m_sync3->m_config[i].optional = false;
                m_sync3->m_config[i].clear    = true;
            }

            m_sync3->registerCallback(
                boost::bind(
                    &GPUDepthmapFusionComponent::callback3,
                    this, 
                    _1, _2, _3
                )
            );

        }
        if (m_numSubcribedDepthmaps == 4)
        {
            m_sync4 = std::make_shared<message_filters::Synchronizer<SyncPolicy4>>(
                SyncPolicy4(&nh),
                *m_syncSubs[0],
                *m_syncSubs[1],
                *m_syncSubs[2],
                *m_syncSubs[3]
            );

            m_sync4->m_config[0].trigger  = true;
            m_sync4->m_config[0].delay    = false;
            m_sync4->m_config[0].optional = false;
            m_sync4->m_config[0].clear    = true;

            for(int i=1; i<m_numSubcribedDepthmaps; ++i)
            {
                m_sync4->m_config[i].trigger  = false;
                m_sync4->m_config[i].delay    = false;
                m_sync4->m_config[i].optional = false;
                m_sync4->m_config[i].clear    = true;
            }

            m_sync4->registerCallback(
                boost::bind(
                    &GPUDepthmapFusionComponent::callback4,
                    this, 
                    _1, _2, _3, _4
                )
            );
        }
        if (m_numSubcribedDepthmaps == 5)
        {
            m_sync5 = std::make_shared<message_filters::Synchronizer<SyncPolicy5>>(
                SyncPolicy5(&nh),
                *m_syncSubs[0],
                *m_syncSubs[1],
                *m_syncSubs[2],
                *m_syncSubs[3],
                *m_syncSubs[4]
            );

            m_sync5->m_config[0].trigger  = true;
            m_sync5->m_config[0].delay    = false;
            m_sync5->m_config[0].optional = false;
            m_sync5->m_config[0].clear    = true;

            for(int i=1; i<m_numSubcribedDepthmaps; ++i)
            {
                m_sync5->m_config[i].trigger  = false;
                m_sync5->m_config[i].delay    = false;
                m_sync5->m_config[i].optional = false;
                m_sync5->m_config[i].clear    = true;
            }

            m_sync5->registerCallback(
                boost::bind(
                    &GPUDepthmapFusionComponent::callback5,
                    this, 
                    _1, _2, _3, _4, _5
                )
            );

        }
        if (m_numSubcribedDepthmaps == 6)
        {
            m_sync6 = std::make_shared<message_filters::Synchronizer<SyncPolicy6>>(
                SyncPolicy6(&nh),
                *m_syncSubs[0],
                *m_syncSubs[1],
                *m_syncSubs[2],
                *m_syncSubs[3],
                *m_syncSubs[4],
                *m_syncSubs[5]
            );

            m_sync6->m_config[0].trigger  = true;
            m_sync6->m_config[0].delay    = false;
            m_sync6->m_config[0].optional = false;
            m_sync6->m_config[0].clear    = true;

            for(int i=1; i<m_numSubcribedDepthmaps; ++i)
            {
                m_sync6->m_config[i].trigger  = false;
                m_sync6->m_config[i].delay    = false;
                m_sync6->m_config[i].optional = false;
                m_sync6->m_config[i].clear    = true;
            }

            m_sync6->registerCallback(
                boost::bind(
                    &GPUDepthmapFusionComponent::callback6,
                    this, 
                    _1, _2, _3, _4, _5, _6
                )
            );

        }
    }
    m_subCameraInfo.resize(m_numSubcribedDepthmaps);
    for (int i=0; i<m_numSubcribedDepthmaps; ++i)
    {
        m_subCameraInfo[i] = nh.subscribe(
            "in/CameraInfo/" + std::to_string(i), 
            in_dm_queue_size, 
            &GPUDepthmapFusionComponent::callbackCameraInfo, 
            this);
    }
    m_subFilterFlyingPixelsThreshold = nh.subscribe(
        "in/Config/FilterFlyingPixels/Threshold", in_dm_queue_size,
        &GPUDepthmapFusionComponent::callbackCfgFilterFlyingPixelsThreshold, this
    );
    m_subFilterFlyingPixelsSize = nh.subscribe(
        "in/Config/FilterFlyingPixels/Size", in_dm_queue_size,
        &GPUDepthmapFusionComponent::callbackCfgFilterFlyingPixelsSize, this
    );
    m_subFilterFlyingPixelsEnableRot45 = nh.subscribe(
        "in/Config/FilterFlyingPixels/EnableRot45", in_dm_queue_size,
        &GPUDepthmapFusionComponent::callbackCfgFilterFlyingPixelsEnableRot45, this
    );
    // glComputeTest(
    //     "/home/xaedes/ros/ws/current/build/gpu_depthmap_fusion/compute.glsl",
    //     "/home/xaedes/ros/ws/current/build/gpu_depthmap_fusion/compute2.glsl",
    //     "/home/xaedes/ros/ws/current/build/gpu_depthmap_fusion/shader.vert",
    //     "/home/xaedes/ros/ws/current/build/gpu_depthmap_fusion/shader.frag"
    // );
}
