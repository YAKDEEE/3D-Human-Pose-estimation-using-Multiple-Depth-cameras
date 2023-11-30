#include <stdio.h>
#include <stdlib.h>
#include "build/dtw.hpp"
#include <k4abttypes.h>
#include <k4a/k4a.h>
#include <k4abt.h>
#include <iostream>
#include<vector>
#include <sstream>
#include <fstream>
#include <memory>
#include <thread>
#include <time.h>
#include <numeric>
#include <fmt/core.h>
#include <cmath>
#include "open3d/Open3D.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include<windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")


using namespace open3d;

using namespace std;
using namespace cv;

static float e = 10e-4;
static float Pi = 3.141592;

#define VERIFY(result, error)                                                                            \
    if(result != K4A_RESULT_SUCCEEDED)                                                                   \
    {                                                                                                    \
        printf("%s \n - (File: %s, Function: %s, Line: %d)\n", error, __FILE__, __FUNCTION__, __LINE__); \
        exit(1);                                                                                         \
    }                                                                                                    \

// 
//             -  18(HIP_LEFT)   -19(KNEE_LEFT)   - 20(ANKLE_LEFT) - 21(FOOT_LEFT)
// 
//   -1(False_PELVIS) - 0(PELVIS) -  1(SPINE_NAVAL)  - 2 (SPINE_CHEST) 
//             
//              -  22(HIP_RIGHT) - 23(KNEE_RIGHT) -  24(ANKLE_RIGHT) - 25(FOOT_RIGHT)
// 
/////////////////////////////////////////////////////////////////////////////////////
// 
//           -4(CLAVICLE_LEFT) - 5(SHOULDER_LEFT) - 6(ELBOW_LEFT) - 7(WRIST_LEFT) - 8(HAND_LEFT)
// 
//  2(SPINE_CHEST)  -   3(NECK)  - 26(HEAD) - 27(NOSE)
//
//       -  11(CLAVICLE_RIGHT)   - 12(SHOULDER_RIGHT) - 13(ELBOW_RIGHT) - 14(WRIST_RIGHT) - 15(HAND_RIGHT)


//                                  27
//                                  I
//                                  26
//                                  I
//                                  3
//                                  I
//  8 - - 7 - - 6 - - 5 - - 4 - -   2  - - 11 - - 12 - - 13 - - 14 - - 15
//                                 II   
//                
//                                  1
// 
//                                  II
//                            22- - 0 - - 18
//                           //            ＼＼
//                          23               19
//                          II               II
//                          24               20
//                          II               II
//                          25               21
#define MOTION_NUM 19
static vector<int> joints_hierarchy;
const char* intrinsicArr[15] = { "cx", "cy", "fx","fy","k1","k2","k3","k4","k5","k6","codx","cody","p2","p1","metric radius" };
const char* jointArr[28] = { "Pelvis", "Spine 1", "Spine 2","Neck","Clavicle_L","Shoulder_L","Elbow_L","Wrist_L","Hand_L"," "," ","Clavicle_R",
"Shoulder_R","Elbow_R","Wrist_R","Hand_R"," "," ","Hip_L","Knee_L","Ankle_L","Foot_L","Hip_R", "Knee_R","Ankle_R","Foot_R" ,"Head", "Nose" };
const char* instruction2[19] = { "dummy.wav" , "기본준비서기", "아래막기","반대지르기" ,"아래막기2","지르기","아래막으며지르기","안막기","바로지르기","안막기2","바로지르기2"
                        ,"아래막으며지르기2","얼굴막기","앞차며지르기","얼굴막기2","앞차며반대지르기","아래막기3","반대지르기2" ,"바로" };
const char* motion_name[19] = { "dummy.png", "0.png", "1.png" , "2.png" , "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png" ,"10.png" ,"11.png" ,"12.png",
                            "13.png","14.png","15.png","16.png","17.png" }; //2023_11_19
const char* instruction[8] = { "dummy.wav" , "아래막" , "반대지르기" , "안막기" , "바로지르기" , "얼굴막기", "앞차기" }; 
const int vsize[2] = { 56, 3 };// vector size
const int rsize[3] = { 56, 3 ,3 }; //matrix size
// 추가 0407
const int vsize_2D[2] = { 56,2 }; // vector 2D size
Scalar colors[28];
Scalar colors0[28];

void  static Hierarchy_set();
float pre_score;
//check 2points
void Print2p(float x, float y);
//check 3points
void Print3p(float x, float y, float z);
//check vector
void vector_print(vector<vector<vector<float>>>& v);
void Print_joint_error(Mat pset1, Mat pset2, int i_body, string c);
void CheckIntrinsicParam(k4a_calibration_t sensor_calibration);

void Make_vector(int index, Mat& vectors, Mat& p_set, int i_body);
void Make_vector2D(int index, Mat& p_set, int i_body);

void Get_theta(int index, Mat& vectors, float theta[], int i_body);
void Get_n_vector(int index, Mat& vectors, Mat& nv_set, int i_body);
void Get_rotation_matrix(int index, Mat& nv_set, float th_set[], Mat& rot_set, Mat& inv_rot_set, int i_body);

static void create_xy_table(const k4a_calibration_t* calibration, k4a_image_t xy_table);


static void generate_point_cloud(const k4a_image_t depth_image,
    const k4a_image_t xy_table,
    k4a_image_t point_cloud,
    int* point_count);
static void write_point_cloud(const char* file_name, const k4a_image_t point_cloud, int point_count);
void MakeExtrinsicMatrix(string filename1, string filename2, Mat& transform);

void static color();
void static color0();
void Multiply_extrinsic_matrix(Mat& pset1, Mat& pset2, Mat transform, int i_body);

static void Get_3dPoints(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set);
static void  Drawing_circle(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set, k4a_calibration_t sensor_calibration, k4a_float2_t xy_color[], Mat& p_set_2D, k4abt_frame_t body_frame, Mat& colorMat,
    k4a_image_t color_image, uint8_t* color_buffer, Scalar colors[]);
static void Drawing_marker(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set, k4a_calibration_t sensor_calibration, k4a_float2_t xy_color[], Mat& p_set_2D, k4abt_frame_t body_frame, Mat& colorMat,
    k4a_image_t color_image, uint8_t* color_buffer, Scalar colors[]);
static void Setting_confidence_level(k4abt_skeleton_t& skeleton, double confidence[]);
void Fusion_points(Mat& p_set2, Mat& newpset1, Mat& fusion_pset, double confidence_level[], double confidence_level2[]);
void Icp_go();

class VisualizerWithDepthCapture
    : public visualization::VisualizerWithCustomAnimation {
protected:
    void KeyPressCallback(GLFWwindow* window,
        int key,
        int scancode,
        int action,
        int mods) override {
        if (action == GLFW_RELEASE) {
            return;
        }
        if (key == GLFW_KEY_S) {
            CaptureDepthImage("depth.png");
            CaptureDepthPointCloud("depth.ply");
            camera::PinholeCameraTrajectory camera;
            camera.parameters_.resize(1);
            view_control_ptr_->ConvertToPinholeCameraParameters(
                camera.parameters_[0]);
            io::WriteIJsonConvertible("camera.json", camera);
        }
        else if (key == GLFW_KEY_L) {
            if (utility::filesystem::FileExists("depth.png") &&
                utility::filesystem::FileExists("camera.json")) {
                camera::PinholeCameraTrajectory camera;
                io::ReadIJsonConvertible("camera.json", camera);
                auto image_ptr = io::CreateImageFromFile("depth.png");
                auto pointcloud_ptr =
                    geometry::PointCloud::CreateFromDepthImage(
                        *image_ptr, camera.parameters_[0].intrinsic_,
                        camera.parameters_[0].extrinsic_);
                AddGeometry(pointcloud_ptr);
            }
        }
        else if (key == GLFW_KEY_K) {
            if (utility::filesystem::FileExists("depth.ply")) {
                auto pointcloud_ptr = io::CreatePointCloudFromFile("depth.ply");
                AddGeometry(pointcloud_ptr);
            }
        }
        else if (key == GLFW_KEY_P) {
            if (utility::filesystem::FileExists("depth.png") &&
                utility::filesystem::FileExists("camera.json")) {
                camera::PinholeCameraTrajectory camera;
                io::ReadIJsonConvertible("camera.json", camera);
                view_control_ptr_->ConvertFromPinholeCameraParameters(
                    camera.parameters_[0]);
            }
        }
        else {
            visualization::VisualizerWithCustomAnimation::KeyPressCallback(
                window, key, scancode, action, mods);
        }
        UpdateRender();
    }
};

void static Comparison_Rod(int num_bodies, Mat rod_l[]);


//PointCloud_visualization
void VisualizeRegistration(const open3d::geometry::PointCloud& source,
    const open3d::geometry::PointCloud& target,
    const Eigen::Matrix4d& Transformation) {
    std::shared_ptr<geometry::PointCloud> source_transformed_ptr(
        new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> target_ptr(new geometry::PointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->Transform(Transformation);
    visualization::DrawGeometries({ source_transformed_ptr, target_ptr },
        "Registration result");
}


int main()
{
    //Hierarchy_initalize
    Hierarchy_set();
    color();
    color0();

    //save video 
    int mode = 1; //mode 0 : saving ans frame, mode  1: comparing with ans frame 
    int icp_mode = 0; //icp_mode 0 : 평가 모드 동작 icp_mode 1 : icp모드 동작
    if (icp_mode == 1) {
        // icp 후 matrix 반환
        Icp_go();
        return 0;
    }

    //4*4 tf matirx 가져오기
    cv::FileStorage fsRead("extrinsic_matrix.xml", cv::FileStorage::READ);
    cv::Mat trans(4, 4, CV_32FC1);
    fsRead["trans"] >> trans;
    fsRead.release();

    int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');
    VideoWriter writer[MOTION_NUM];
    VideoWriter test_writer;
    // motion var
    float motion_frames_divider[MOTION_NUM] = { 0, 2.85,2.44,2.7,1.75,1.67,1.78,1.82,2.13,2.31,1.82,1.91,2.43,1.76,2.67, 1.67,2.23,0.8,2.52 };
    int motion_frames_ref_start[MOTION_NUM] = { 0, 0, 130, 160, 186, 226, 262, 307, 340, 373, 412, 445 , 487, 520, 577, 607, 667, 703, 766 };
    
    int motion_frames_limit[MOTION_NUM] = { 0 ,50,40 ,40 ,40 ,40 ,40,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40, 40 }; // 18동작
    int motion_frames_limit_ans[MOTION_NUM] = { 0 ,367, 73, 70, 70, 60, 80 , 60 , 70 , 90 , 60 , 80 , 80 , 100 , 80 , 100 ,80 , 60 ,320 }; //18동작
    

    //기본준비서기 129       	0~129 		div 2.85       0 + matched_index * div(2.85)
  //      아래막기 30 	       	130~159		div 2.44       130 + matched_index * div(2.44)
 //       반대지르기 26 	   	160~185		div 2.7
 //       아래막기2 40 	     	186~225		div 1.75
 //       지르기 36 	        226~261		div 1.67
 //       아래막으며지르기 45  	262~306		div 1.78
 //       안막기 33     		307~339		div 1.82
 //       바로지르기 33  		340~372		div 2.13
 //       안막기2 39    		373~411		div 2.31
 //       바로지르기2 33  	 	412~444		div 1.82
 //       아래막으며지르기2 42  445~486		div 1.91
 //       얼굴막기 33  		    487~519		div 2.43
 //       앞차며지르기 57  		520~576		div 1.76
 //       얼굴막기2 30  		577~606		div 2.67
 //       앞차며반대지르기 60   607~666		div 1.67
 //       아래막기3 36  		667~702		div 2.23
 //        반대지르기2 75 	 	703~777		div 0.8
 //        바로 127   		    778~904		div 2.52


    VideoCapture cap("reference2.mp4");
    Mat ref[900];
    int fcnt = 0;
    //cout  << "total video frame : " << int(cap.get(CAP_PROP_FRAME_COUNT)) << '\n';

    while (fcnt < 900) {
        
        cap.read(ref[fcnt]);
        fcnt++;
    }
    //motion text image 불러오기
    Mat motion_text[19];
    for (int i = 1; i < MOTION_NUM; i++) {
        string text_path = "./instruction_text/"; 
        text_path += motion_name[i]; 
        motion_text[i] = imread(text_path);
    }
    
    // legacy motion_frames_limit 11.10 
	//int motion_frames_limit[MOTION_NUM] = { 0 ,50,40 ,40 ,40 ,40 ,40,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40 ,40, 40 }; // 18동작
    // int motion_frames_limit_ans[MOTION_NUM] = { 0,368, 73, 70, 70, 60, 80 , 60 , 70 , 90 , 60 , 80 , 80 , 100 , 80 , 100 ,80 , 60 ,320 }; //18동작

    int motion_frame_limit = 50; //자체 비교모드 test frame 10_!9
    int max_frame_count = 0; //10동작 frames 더하면됨.
    for (int i = 0; i < MOTION_NUM; i++) {
        max_frame_count += motion_frames_limit[i];
    }

    int motion_index = 1;
    // motion_clk => frames % motion_frames_limit[motion_index]
    // motion 0 -> motion_frames_limit[motion_index] ==  (motion_clk + 1) --> motion_index++, motion 1
    string score = "100";
    string score2 = "Score: ";

    string c1 = "";
    string c2 = "";
    string c3 = "";
    if (mode == 1) {
        test_writer.open("test_video.avi", fourcc, 15, Size(1280, 720), 1);
    }
    // string file_rod = "rod_l.csv";
    // std::ofstream ofs2(file_rod);

    // string file_gv = "global_vector.csv";
    // std::ofstream ofs3(file_gv);
    
    /////2023-11-16 Saving feedback video 사용자
    vector<Mat> total_image;
    total_image.assign(max_frame_count,Mat());
    vector<vector<int>> all_aligned_path;
    all_aligned_path.assign(MOTION_NUM,vector<int>(50,0));
    
    k4a_device_t device = NULL;
    k4a_device_t device2 = NULL;
    VERIFY(k4a_device_open(0, &device), "Open K4A Device 1 failed!");
    VERIFY(k4a_device_open(1, &device2), "Open K4A Device 2 failed!");

    const int32_t TIMEOUT_IN_MS = 1000000;
    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    deviceConfig.camera_fps = K4A_FRAMES_PER_SECOND_30;
    deviceConfig.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_720P;

    k4a_device_configuration_t deviceConfig2 = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig2.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    deviceConfig2.camera_fps = K4A_FRAMES_PER_SECOND_30;
    deviceConfig2.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
    deviceConfig2.color_resolution = K4A_COLOR_RESOLUTION_720P;
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras 1 failed!");
    VERIFY(k4a_device_start_cameras(device2, &deviceConfig2), "Start K4A cameras 2 failed!");

    // sensor calibration
    k4a_calibration_t sensor_calibration;
    k4a_calibration_t sensor_calibration2;
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensor_calibration),
        "Get depth camera 1 calibration failed!");
    VERIFY(k4a_device_get_calibration(device2, deviceConfig2.depth_mode, deviceConfig2.color_resolution, &sensor_calibration2),
        "Get depth camera 2 calibration failed!");

    //Check intrinsics info
    cout << "Kinect 1 intrinsic: " << '\n';
    CheckIntrinsicParam(sensor_calibration);
    cout << "Kinect 2 intrinsic: " << '\n';
    CheckIntrinsicParam(sensor_calibration2);

    //Body traker config
    k4abt_tracker_t tracker = NULL;
    k4abt_tracker_t tracker2 = NULL;
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker), "Body tracker 1 initialization failed!");
    VERIFY(k4abt_tracker_create(&sensor_calibration2, tracker_config, &tracker2), "Body tracker2 initialization failed!");




    //frame start
    int frame_count = 0;
    //814 frame local rodrigues 50frame단위   
    vector<vector<vector<vector<double>>>> frame_rod;  //motion_clk * frames * joints * xyz
    vector<vector<vector<vector<double>>>> frame_vec;
    

    // motion 간의 frames 폭 저장   
    int motion_depth[MOTION_NUM] = { 0, };
    for (int i = 1; i < MOTION_NUM; i++) {
        motion_depth[i] = motion_frames_limit[i] + motion_depth[i - 1];
    }
    //int test_joint = 27;
    string file_name_c = "confidence_all.csv";//0 9 10 16 17 제외
    std::ofstream ofs_c(file_name_c);


    //kincet 3d 정보 저장 10_21
	string file_name = "kinec3DInfo.csv";
	std::ofstream ofs(file_name);

    do
    {

        //int motion_count = (frame_count) / motion_frames_limit[motion_index];

        int motion_clk = (frame_count - motion_depth[motion_index - 1]) % motion_frames_limit[motion_index];

        //cout << "frame : " << frame_count << " motion_index : " << motion_index << "motion clk : " << motion_clk << '\n';

        //motion_clk 
        if (motion_clk == 0) {
            frame_rod.assign(MOTION_NUM, vector<vector<vector<double>>>(motion_frames_limit[motion_index], vector<vector<double>>(28, vector<double>(3))));
            frame_vec.assign(MOTION_NUM, vector<vector<vector<double>>>(motion_frames_limit[motion_index], vector<vector<double>>(28, vector<double>(3))));
        }
        k4a_capture_t sensor_capture;
        k4a_capture_t sensor_capture2;

        k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &sensor_capture, K4A_WAIT_INFINITE);
        k4a_wait_result_t get_capture_result2 = k4a_device_get_capture(device2, &sensor_capture2, K4A_WAIT_INFINITE);
        if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED && get_capture_result2 == K4A_WAIT_RESULT_SUCCEEDED)
        {

            k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, sensor_capture, K4A_WAIT_INFINITE);
            k4a_wait_result_t queue_capture_result2 = k4abt_tracker_enqueue_capture(tracker2, sensor_capture2, K4A_WAIT_INFINITE);

            k4a_capture_release(sensor_capture);
            k4a_capture_release(sensor_capture2);// Remember to release the sensor capture once you finish using it
            if (queue_capture_result == K4A_WAIT_RESULT_TIMEOUT || queue_capture_result2 == K4A_WAIT_RESULT_TIMEOUT) {
                // It should never hit timeout when K4A_WAIT_INFINITE is set.
                printf("Error! Add capture to tracker process queue timeout!\n");
                continue;
            }
            else if (queue_capture_result == K4A_WAIT_RESULT_FAILED || queue_capture_result2 == K4A_WAIT_RESULT_FAILED)
            {
                printf("Error! Add capture to tracker process queue failed!\n");
                continue;
            }

            k4abt_frame_t body_frame = NULL;
            k4abt_frame_t body_frame2 = NULL;
            k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);
            k4a_wait_result_t pop_frame_result2 = k4abt_tracker_pop_result(tracker2, &body_frame2, K4A_WAIT_INFINITE);

            if (pop_frame_result == K4A_WAIT_RESULT_SUCCEEDED && pop_frame_result2 == K4A_WAIT_RESULT_SUCCEEDED)
            {
                // Successfully popped the body tracking result. Start your processing
                k4a_image_t depth = k4a_capture_get_depth_image(sensor_capture);
                k4a_image_t color_image = k4a_capture_get_color_image(sensor_capture);
                size_t num_bodies = k4abt_frame_get_num_bodies(body_frame);

                k4a_image_t depth2 = k4a_capture_get_depth_image(sensor_capture2);
                k4a_image_t color_image2 = k4a_capture_get_color_image(sensor_capture2);
                size_t num_bodies2 = k4abt_frame_get_num_bodies(body_frame2);


                //cout << "num_bodies1 is " << num_bodies << '\n'; 727 test
                //cout << "num_bodies2 is " << num_bodies2 << '\n';

                //initalize position & store vecotr of parnet, child
                //::Mat joints_vector(24, 2, 3);
                Mat v_set(2, vsize, CV_32FC1, Scalar(0)); // vectors set 28*3
                v_set.at<float>(0, 0) = 0; //false pelvis vector set
                v_set.at<float>(0, 1) = -10; // 10.05 fbx파일 0 joint vector (0 0 1 )기준으로 만들어서 변경
                v_set.at<float>(0, 2) = 0;

                v_set.at<float>(28, 0) = 0; //false pelvis vector set -->수정예정
                v_set.at<float>(28, 1) = -10; // 10.05 fbx파일 0 joint vector (0 0 1 )기준으로 만들어서  변경
                v_set.at<float>(28, 2) = 0;
                // 0407 추가
                Mat p_set_2D(2, vsize_2D, CV_32FC1, Scalar(0)); // vectors set 28*2

                Mat p_set(2, vsize, CV_32FC1, Scalar(0)); //3d points set 28*3
                float th_set[56] = { 0, }; //theta set 28*1 // 0407 변경 27->28
                Mat rot_set[56]; // rotation matrix set 28 * 3 * 3
                Mat inv_rot_set[56]; //inverse rotation matrix set 28 * 3 * 3
                Mat nv_set(2, vsize, CV_32FC1, Scalar(0)); //n_vector set 28*3
                Mat frames[56]; //좌표계 28개
                Mat rod_l[56]; //로드리게스 벡터 local
                Mat rod_g[56]; //로드리게스 벡터 global
                Mat new_pset(2, vsize, CV_32FC1, Scalar(0)); //3d points set result of multiplying extrinsic matrix

                ///////////////////
                //kinect2 joints///
                Mat v_set2(2, vsize, CV_32FC1, Scalar(0)); // vectors set 28*3
                v_set2.at<float>(0, 0) = 0; //false pelvis vector set
                v_set2.at<float>(0, 1) = -10; // 10.05 fbx파일 0 joint vector (0 0 1 )기준으로 만들어서  변경
                v_set2.at<float>(0, 2) = 0;

                v_set2.at<float>(28, 0) = 0; //false pelvis vector set -->수정예정
                v_set2.at<float>(28, 1) = -10; // 10.05 fbx파일 0 joint vector (0 0 1 )기준으로 만들어서 변경
                v_set2.at<float>(28, 2) = 0;
                // 0407 추가
                Mat p_set_2D2(2, vsize_2D, CV_32FC1, Scalar(0)); // vectors set 28*2

                Mat new_pset2(2, vsize, CV_32FC1, Scalar(0)); //3d points set result of multiplying extrinsic matrix

                Mat p_set2(2, vsize, CV_32FC1, Scalar(0)); //3d points set 28*3
                float th_set2[56] = { 0, }; //theta set 28*1 // 0407 변경 27->28
                Mat rot_set2[56]; // rotation matrix set 28 * 3 * 3
                Mat inv_rot_set2[56]; //inverse rotation matrix set 28 * 3 * 3
                Mat nv_set2(2, vsize, CV_32FC1, Scalar(0)); //n_vector set 28*3
                Mat frames2[56]; //좌표계 28개
                Mat rod_l2[56]; //로드리게스 벡터 local
                Mat rod_g2[56]; //로드리게스 벡터 global


                


                ////727 fusion 3d joint points
                Mat fusion_pset(2, vsize, CV_32FC1, Scalar(0)); // vectors set 28*3

                /* printf("%zu bodies are detected!\n", num_bodies);*/
                if (num_bodies == num_bodies2 ) {
                    uint8_t* color_buffer = k4a_image_get_buffer(color_image);
                    uint8_t* color_buffer2 = k4a_image_get_buffer(color_image2);
                    //cv::namedWindow("image"); // 이미지를 보여주기 위한 빈 창
                    cv::Mat colorMat(720, 1280, CV_8UC4, (void*)color_buffer, cv::Mat::AUTO_STEP);
                    cv::Mat colorMat2(720, 1280, CV_8UC4, (void*)color_buffer2, cv::Mat::AUTO_STEP);// color camera 이미지
                    vector<k4abt_skeleton_t> skeleton(num_bodies);
                    vector<k4abt_skeleton_t> skeleton2(num_bodies2);
                    k4a_float2_t xy_color[56];
                    k4a_float2_t xy_color2[56];
                    string wav = ".wav";
                    for (size_t i_body = 0; i_body < num_bodies; i_body++)
                    {

                        //motion instruction 823
                        if (motion_clk % motion_frames_limit[motion_index] == 0) {
                            PlaySound(instruction2[motion_index], NULL, SND_FILENAME | SND_ASYNC);
                            cout << "sleep" << '\n';
                            Sleep(2000);
                        }
                        k4abt_frame_get_body_skeleton(body_frame, i_body, &skeleton[i_body]);
                        k4abt_frame_get_body_skeleton(body_frame2, i_body, &skeleton2[i_body]);
                        // kinect 1
                        cv::Mat cpColorMat(720, 1280, CV_8UC3); // copy 이미지
                        colorMat.copyTo(cpColorMat);
                        cv::Mat grayMat(720, 1280, CV_8UC1); // gray 이미지
                        cv::cvtColor(cpColorMat, grayMat, COLOR_BGR2GRAY);
                        //kinect 2
                        cv::Mat cpColorMat2(720, 1280, CV_8UC3); // copy2 이미지
                        colorMat2.copyTo(cpColorMat2);
                        cv::Mat grayMat2(720, 1280, CV_8UC1); // gray2 이미지
                        cv::cvtColor(cpColorMat2, grayMat2, COLOR_BGR2GRAY);


                        //get 3d body joint points
                        Get_3dPoints(skeleton, i_body, p_set);
                        Get_3dPoints(skeleton2, i_body, p_set2);

                        // calibration & drawing joints                   
                        /*Drawing_circle(skeleton, i_body, p_set, sensor_calibration, xy_color, p_set_2D, body_frame, colorMat, color_image, color_buffer, colors);*/
                        /*Drawing_circle(skeleton2, i_body, p_set2, sensor_calibration2, xy_color2, p_set_2D2, body_frame2, colorMat2, color_image2, color_buffer2, colors);*/

                        //// 추가
                        //Projection kinect2 points to kinect1 points
                        //Multiply_extrinsic_matrix(p_set2, new_pset, trans.inv(), i_body);
                        Drawing_marker(skeleton, i_body, new_pset, sensor_calibration, xy_color, p_set_2D, body_frame, colorMat, color_image, color_buffer, colors0);

                        //Projection kinect1 points to kinect2 points
                        Multiply_extrinsic_matrix(p_set, new_pset2, trans, i_body);
                        //Drawing_marker(skeleton2, i_body, new_pset2, sensor_calibration2, xy_color2, p_set_2D2, body_frame2, colorMat2, color_image2, color_buffer2, colors0);

                        double confidence_level[28];
                        double confidence_level2[28];

                        Setting_confidence_level(skeleton[0], confidence_level);
                        Setting_confidence_level(skeleton2[0], confidence_level2);
                        Fusion_points(p_set2, new_pset2, fusion_pset, confidence_level, confidence_level2);

                        //drawing skeleton on image 
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            /*cv::line(colorMat, Point(xy_color[28 * i_body + j].xy.x, xy_color[28 * i_body + j].xy.y), Point(xy_color[28 * i_body + joints_hierarchy[j]].xy.x, xy_color[28 * i_body + joints_hierarchy[j]].xy.y), colors[j], 3, 1, 0);*/
                        }
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            /* cv::line(colorMat2, Point(xy_color2[28 * i_body + j].xy.x, xy_color2[28 * i_body + j].xy.y), Point(xy_color2[28 * i_body + joints_hierarchy[j]].xy.x, xy_color2[28 * i_body + joints_hierarchy[j]].xy.y), colors[j], 3, 1, 0);*/
                        }


                        /*cout << "kinect 1 : 3D  info :" << '\n';*/
                        //ofs << "kinect 1 : 3D  info :" << '\n';
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            Make_vector(j, v_set, fusion_pset, i_body);//인덱스의 부모 좌표에서 인덱스 좌표로 가는 3차원 벡터
                            // cout << "joint number : " << j << " joint name : " << jointArr[j] << " point info :" << p_set.row(j) << '\n';
                                //ofs << "joint number , " << j << "," << "point info ," << p_set.row(j) << '\n';
                        }
                        //cout << "kinect 1->kinect2 : 3D  info :" << '\n';
                        //ofs << "kinect 1->kinect2 : 3D  info :" << '\n';
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10) { continue; }
                            //cout << "joint number : " << j << " joint name : " << jointArr[j] << "point info :" << new_pset2.row(j) << '\n';
                            //ofs << "joint number , " << j << "," << "point info ," << new_pset2.row(j) << '\n';
                        } //727 test

                        /*cout << '\n' << "kinect 1 : 2d  info :" << '\n';*/
                        //ofs << "kinect 1 : 2d  info :" << '\n';
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10) { continue; }
                            //Make_vector2D(j, p_set_2D, i_body);
                            //cout << "joint number : " << j << " " << "point info :" << p_set_2D.row(j) << '\n';
                            //ofs << "joint number , " << j << "," << "point info ," << p_set_2D.row(j) << '\n';
                        }
                        //cout << "kinect 2->kinect1 : 3D  info :" << '\n';
                        //ofs << "kinect 2->kinect1 : 3D  info :" << '\n';
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10) { continue; }
                            //cout << "joint number : " << j << " joint name : " << jointArr[j] << "point info :" << new_pset.row(j) << '\n';
                            //ofs << "joint number ," << j << "," << "point info ," << new_pset.row(j) << '\n';
                        }
                        //cout << "kinect 2 : 3D  info :" << '\n';
                        //ofs << "kinect 2 : 3D  info :" << '\n';
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            Make_vector(j, v_set2, p_set2, i_body);
                            //cout << "joint number : " << j << " " << "point info :" << p_set2.row(j) << '\n';
                            //ofs << "joint number ," << j << "," << "point info ," << p_set2.row(j) << '\n';
                        }

                        //cout << '\n' << "kinect 2 : 2D  info :" << '\n';
                        //ofs << "kinect 2 : 2D  info :" << '\n';
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10) { continue; }
                            //Make_vector2D(j, p_set_2D2, i_body);
                            //cout << "joint number : " << j << " " << "point info :" << p_set_2D2.row(j) << '\n';
                                //ofs << "joint number ," << j << "," << "point info ," << p_set_2D2.row(j) << '\n';
                        } //727 test
                        //ofs << trans << '\n';
                        //ofs.close(); //727 test


                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            Get_theta(j, v_set, th_set, i_body);  //인덱스의 부모, 인덱스의 부모의 부모 참조해서 만드는   ,radian 값         
                            Get_n_vector(j, v_set, nv_set, i_body); //로드리게스 벡터 만들기
                            Get_rotation_matrix(j, nv_set, th_set, rot_set[28 * i_body + j], inv_rot_set[28 * i_body + j], i_body); //로테이션 메트릭스 구하기, inv 로테이션 메트릭스 구하기
                            //cout << '\n';
                        }


                        //global 로드리게스 벡터 만들기
                        for (int j = 0; j <= 27; j++) {
                            if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
                            rod_g[28 * i_body + j] = th_set[28 * i_body + j] * nv_set.row(28 * i_body + j);
                        }
                        frames[0] = Mat::eye(3, 3, CV_32F);
                        frames[28] = Mat::eye(3, 3, CV_32F);
                        //local 로드리게스 벡터 만들기]
                        for (int i = 0; i <= 27; i++) {
                            if (i == 9 || i == 16 || i == 17 || i == 10 || i == 0) { continue; }
                            rod_l[28 * i_body + i] = (frames[28 * i_body + joints_hierarchy[i]].inv() * rod_g[28 * i_body + i].t()).t();
                            frames[28 * i_body + i] = rot_set[28 * i_body + i] * frames[28 * i_body + joints_hierarchy[i]];
                        }

                        //for joint visulization 10_12
                        Drawing_marker(skeleton2, 0, fusion_pset, sensor_calibration2, xy_color2, p_set_2D2, body_frame2, colorMat2, color_image2, color_buffer2, colors0);


                       // for calculation 신뢰도대비 오차 joint 8 왼손 10_12 joint 15 오른손
                         //10_19 confidence level저장 , point 3d 저장, 유클리디안 거리 차이 저장.
                       // int j = test_joint;
       //                 for (int j = 0; j < 28; j++) {
							//if (j == 9 || j == 16 || j == 17 || j == 10) { continue; }
       //                     cout << "joint: " << jointArr[j] << " kinect 1 confidecne level : " << confidence_level[j] << 
							//"point info 1->2 ," << new_pset2.row(j) <<'\n';
       //                     ofs_c << "joint: ," << jointArr[j] << ",kinect 1 confidecne level : ," << confidence_level[j] <<
							//	", " << "point info 1->2 ," << new_pset2.row(j) << " , ";
       //                     cout << "joint: " << jointArr[j] << " kinect 2 confidecne level : " << confidence_level2[j] <<
       //                     "point info 2 " << ", " << p_set2.row(j)<< '\n';
       //                     ofs_c << "joint: ," << jointArr[j] << ", kinect 2 confidecne level : ," << confidence_level2[j] << ',' <<
       //                       "point info 2 " << ", " << p_set2.row(j) << " , " ;

       //                     cout << "유클리디안 거리 : " << sqrt(pow(new_pset2.at<float>(j, 0) - p_set2.at<float>(j, 0), 2) +
       //                         pow(new_pset2.at<float>(j, 1) - p_set2.at<float>(j, 1), 2) + pow(new_pset2.at<float>(j, 2) - p_set2.at<float>(j, 2), 2)) << '\n' << '\n';
       //                     ofs_c << "유클리디안 거리 : ," << sqrt(pow(new_pset2.at<float>(j, 0) - p_set2.at<float>(j, 0), 2) +
       //                         pow(new_pset2.at<float>(j, 1) - p_set2.at<float>(j, 1), 2) + pow(new_pset2.at<float>(j, 2) - p_set2.at<float>(j, 2), 2)) << ',' << '\n';

       //                     c1 = fmt::format("{:.2f}", confidence_level[j]);
       //                     c2 = fmt::format("{:.2f}", confidence_level2[j]);
       //                     c3 = fmt::format("{:.2f}", sqrt(pow(new_pset2.at<float>(j, 0) - p_set2.at<float>(15, 0), 2) + pow(new_pset2.at<float>(15, 1) -
       //                         p_set2.at<float>(15, 1), 2) + pow(new_pset2.at<float>(15, 2) - p_set2.at<float>(15, 2), 2)));
       //                 }

                        cvtColor(colorMat, colorMat, COLOR_RGBA2RGB);
                        cvtColor(colorMat2, colorMat2, COLOR_RGBA2RGB);
                        if (mode) {
                            //for score visualization 10_12
                            putText(colorMat2, score2 + score, Point(40, 100), 3, 2, Scalar(0, 0, 0), 3, 1);
                        }

                    }
                    //818 frame단위 rod vector 저장.
                    /*cout << frame_count << '\n';*/

                    //10_12
                    if (motion_clk < motion_frames_limit[motion_index]) {
                        for (int k = 0; k <= 27; k++) {
                            if (k == 9 || k == 16 || k == 17 || k == 10 || k == 0) { continue; }
                            /*cout << "frame_count " << motion_clk << " " << " k " << k << '\n';*/
                            frame_rod[motion_index][motion_clk][k][0] = rod_l[k].at<float>(0);
                            frame_rod[motion_index][motion_clk][k][1] = rod_l[k].at<float>(1);
                            frame_rod[motion_index][motion_clk][k][2] = rod_l[k].at<float>(2);
                        }
                    }
                    if (motion_clk < motion_frames_limit[motion_index]) {
                        for (int k = 0; k <= 27; k++) {
                            if (k == 9 || k == 16 || k == 17 || k == 10 || k == 0) { continue; }
                            /*cout << "frame_count " << motion_clk << " " << " k " << k << '\n';*/
                            frame_vec[motion_index][motion_clk][k][0] = skeleton[0].joints[k].position.xyz.x;
                            frame_vec[motion_index][motion_clk][k][1] = skeleton[0].joints[k].position.xyz.y;
                            frame_vec[motion_index][motion_clk][k][2] = skeleton[0].joints[k].position.xyz.z;
                        }
                    }
                    if ((motion_clk + 1) % motion_frames_limit[motion_index] == 0) {
                        for (int i = 0; i < motion_frames_limit[motion_index]; i++) {
                            for (int joint_index = 0; joint_index <= 27; joint_index++) {
                                if (joint_index == 9 || joint_index == 16 || joint_index == 17 || joint_index == 10 || joint_index == 0) {
                                    // ofs2 << "," << 0 << "," << 0 << "," << 0 << endl;
                                    // ofs3 << "," << 0 << "," << 0 << "," << 0 << endl;
                                }
                                else {
                                    // ofs2 << "," << frame_rod[motion_index][i][joint_index][0] << "," << frame_rod[motion_index][i][joint_index][1] << "," << frame_rod[motion_index][i][joint_index][2] << endl;
                                    // ofs3 << "," << frame_vec[motion_index][i][joint_index][0] << "," << frame_vec[motion_index][i][joint_index][1] << "," << frame_vec[motion_index][i][joint_index][2] << endl;
                                }
                            }

                        }
                    }

                    if (frame_count == (max_frame_count - 1)) {
                        // ofs2.close();
                        // ofs3.close();

                    }
                    //10_12




                    //저장모드 , 비교모드 10_21
                    if ((motion_clk + 1) % motion_frames_limit[motion_index] == 0 && frame_count) {
                        if (mode == 0) {
                            string file_name = instruction2[motion_index];

                            file_name += ".csv";
                            std::ofstream ofs(file_name);
                            /*ofs << instruction[int((frame_count+1) / motion_frame_limit) - 1] << '\n';*/
                            for (int i = 0; i < motion_frames_limit[motion_index]; i++) {
                                for (int joint_index = 0; joint_index <= 27; joint_index++) {
                                    if (joint_index == 9 || joint_index == 16 || joint_index == 17 || joint_index == 10 || joint_index == 0) {
                                        ofs << "," << 0 << "," << 0 << "," << 0 << endl;
                                    }
                                    else {
                                        ofs << "," << frame_rod[motion_index][i][joint_index][0] << "," << frame_rod[motion_index][i][joint_index][1] << "," << frame_rod[motion_index][i][joint_index][2] << endl;
                                    }
                                }
                            }
                            ofs.close();

                        }
                        else if (mode) {
                            //816 testing csv joint info reading
                            if ((motion_clk + 1) % motion_frames_limit[motion_index] == 0 && frame_count) {
                                cout << "testing open ans frame" << '\n';
                                // frame * joint * vector (answer)
                                vector < vector < vector<double>>> ans_vector;
                                ans_vector.assign(motion_frames_limit_ans[motion_index], vector < vector<double>>(28, vector<double>(3, 0))); //10_19 fbx 비교 모드
								//ans_vector.assign(motion_frames_limit[motion_index], vector < vector<double>>(28, vector<double>(3, 0))); //10_19 자체 제작 비교 모드
                                fstream fs;
                                string buf;
                                string file_name2 = instruction2[motion_index];
                                file_name2 += ".csv";
                                fs.open(file_name2, ios::in);
                                int cnt = 0;
                                float x = 0, y = 0, z = 0;
                                getline(fs, buf, ',');
								
                                //정답 가져오기
                                while (!fs.eof()) {
									//cin.ignore();
                                    getline(fs, buf, ',');


                                    //(cnt / 3 / 28) % 30 --> frame
                                    //(cnt / 3) % 28 --> joint 
                                    //cnt % 3 --> x = 0, y = 1, z = 2 


                                    if (cnt % 3 == 0) {
                                        x = stod(buf);
                                        //ans_vector[(cnt / 3 / 28) % motion_frames_limit[motion_index]][(cnt / 3) % 28][0] = x;// 10_19 자체 비교 모드
                                        
                                        ans_vector[(cnt / 3 / 28) % motion_frames_limit_ans[motion_index]][(cnt / 3) % 28][0] = x; //10_19 fbx 비교모드
                                        // for test dtw
                                        //frame_rod[(cnt) / motion_frame_limit / 3][(cnt / 3) % motion_frame_limit][0] = x;
                                        //cout << "x : " << x << " ";
                                    }
                                    else if (cnt % 3 == 1) {
                                        y = stod(buf);

										//ans_vector[(cnt / 3 / 28) % motion_frames_limit[motion_index]][(cnt / 3) % 28][1] = y; // 10_19 자체 비교 모드
                                        ans_vector[(cnt / 3 / 28) % motion_frames_limit_ans[motion_index]][(cnt / 3) % 28][1] = y; //10_19 fbx 비교모드

                                        // for test dtw
                                        //frame_rod[(cnt) / motion_frame_limit / 3][(cnt / 3) % motion_frame_limit][1] = y;
                                        // cout << "y : " << y << " ";
                                    }
                                    else if (cnt % 3 == 2) {
                                        z = stod(buf);
										//ans_vector[(cnt / 3 / 28) % motion_frames_limit[motion_index]][(cnt / 3) % 28][2] = z; // 10_19 자체 비교 모드
                                        ans_vector[(cnt / 3 / 28) % motion_frames_limit_ans[motion_index]][(cnt / 3) % 28][2] = z;   //10_19 fbx 비교모드                                     // for test dtw
                                        //frame_rod[(cnt) / motion_frame_limit / 3][(cnt / 3) % motion_frame_limit][2] = z;
                                    }
                                    //cout << cnt << endl;
                                    cnt++;


                                }
                                ofs.close();

                                //test cout 정답프레임 10_21                            
         //                       for (int clk = 0; clk < motion_frames_limit_ans[motion_index]; clk++) {
									//for (int joint_index = 0; joint_index <= 27; joint_index++) {
         //                               cout << "joint index :" << joint_index << " clk : " << clk << " x : " << ans_vector[clk][joint_index][0]
         //                                   << " y : " << ans_vector[clk][joint_index][1] << " z : " << ans_vector[clk][joint_index][2] << '\n';
         //                           }
         //                       }

                                //dtw 적용
                                vector<vector<double>> path;

                                //path.assign(motion_frames_limit[motion_index], vector <double>(motion_frames_limit[motion_index], 0));//10_19 자체 비교 모드
								path.assign(motion_frames_limit_ans[motion_index], vector <double>(motion_frames_limit[motion_index], 0)); //10_19 fbx 비교모드

                                //for 구문 joint 0~28 적용한다.
                                //motion_frames_limit_ans[motion_index] <-- frames 수
                                // frame_rod[motion_index][motion_clk][joint_index][xyz],  여기서 motion_clk -> 0~motion_index에 해당하는 frame, joint_index 0 ~28 각각 


                                //several joints dtw calculating 11_10
                                float max_value = 0;
                                int max_joint = -1;
                                int max_index = 0;
                                vector<double> joints_dtw;
                                joints_dtw.assign(28, 0);
								string file_name_several_dtw = instruction2[motion_index];
								file_name_several_dtw += "_several_dtw.csv";
								std::ofstream ofs_several_dtw(file_name_several_dtw);
								vector<pair<float,float>> score_joint; // joint 오류점수 

                                for (int k = 0; k <= 27; k++) {
                                    if (k == 9 || k == 16 || k == 17 || k == 10 || k == 0) { continue; }
                                    // if (k == 11 || k == 4 || k == 12 || k == 5|| k == 27 || k == 26 || k == 3 
                                    //     ||  k == 2 || k == 1 || k == 22 || k == 18 ) { continue; } // 어깨, 쇄골, 목, 얼굴, 척추, 엉덩이, 발 제외
                                    
                                    float current_value = DTW::dtw_distance_only2_several_joints(ans_vector, frame_rod[motion_index],k);
                                    joints_dtw[k] = current_value;
                                    current_value /= motion_frames_limit_ans[motion_index]; // 프레임 수로 나눠줌
                                    
                                    ofs_several_dtw << ", joint index : ," << k << ", joint name : ," << jointArr[k] << ", joint dtw : ," << current_value  << '\n';
                                    if(current_value > 0.5){
										score_joint.push_back ({ k,current_value} );
                                    }
                                    if(current_value > max_value){
                                        max_value = current_value;
                                        max_index = k;
                                    }
                                }
                                if (score_joint.size()) {cout << instruction2[motion_index] << " 자세를 수정하세요. "  << '\n';  }
                                for(int i = 0; i< score_joint.size(); i++){
                                    int joint_index = score_joint[i].first;
                                    
                                    cout<<"joint index : "<< score_joint[i].first << " joint name : "<< jointArr[joint_index] << " score : "<<
                                    score_joint[i].second<<'\n';
                                }

                                ofs_several_dtw << "most wrong joint : ," << max_index << ", joint dtw : ," << max_value;
                                ofs_several_dtw.close();
                                
        //                        //print  each joints dtw 11_10
                                
								//for (int k = 0; k <= 27; k++) {
								//	if (k == 9 || k == 16 || k == 17 || k == 10 || k == 0) { continue; }
        //                            cout << " index : " << k << " joint dtw : " << joints_dtw[k] << '\n';                              
        //                        }
        //                        cout << "most wrong joint : " << max_index << "\n";
        //                        cout << "value : " << max_value << '\n';
                                
                                //DTW 시작 및 저장 10_21
                                vector<vector<int>> my_dtw_path;

                                float pre_score = DTW::dtw_distance_only2(ans_vector, frame_rod[motion_index], path);

								//dtw matched points
                                
								vector<int> aligned_path; //frames 
								aligned_path.assign(frame_rod[motion_index].size(), 0);
                                my_dtw_path = DTW::dtw_path(path, aligned_path);
                                for(int i = 0; i<frame_rod[motion_index].size(); i++){									
                                    all_aligned_path[motion_index] = aligned_path; //store aligned index frame for all motions
                                    //cout << "our frame : " << i << "matched frame : " << aligned_path[i] << '\n';
                                }


                                //dtw debug
                                /*
                                cout << "dtw path size : " << path.size() << '\n';
                                cout << "testing one element :" << path[0][0] << '\n';
                                cout << "dtw path index pairs " << '\n';
                                for(int i=0; i<my_dtw_path.size(); ++i){
                                    
                                    for(int j=0; j<my_dtw_path[0].size(); ++j){
                                        cout << my_dtw_path[i][j] << ' ';
                                    }
                                    cout << '\n';
                                }
                                cout << '\n';
                                */

                                string temp_motion_index = instruction2[motion_index];
                                string file_dtw = " dtw_distance_info.csv";
                                file_dtw = temp_motion_index + file_dtw;
                                std::ofstream dtwfs(file_dtw);
                                for (int i = 0; i < motion_frames_limit_ans[motion_index]; i++) {
                                    for (int j = 0; j < motion_frames_limit[motion_index]; j++) {
                                        dtwfs << path[i][j] << ", ";
                                    }
                                    dtwfs << endl;
                                }
                                
								dtwfs.close();
                                
                                

                                float final_score;
                                
                                //최종 점수 계산 10_26
                                //final_score = 100 * (1 - pre_score / 21 / motion_frames_limit[motion_index]); //10_21 자체 비교모드
								final_score = 50+50 * (1 - pre_score / 21 / motion_frames_limit_ans[motion_index]); //10_21 fbx 비교모드
                                score = fmt::format("{:.2f}", final_score);
                                cout << "frame : " << motion_clk << " , DTW distance_cossim: " << pre_score << '\n';
                               // cout << " path frame to frame " << '\n';
                                //for(int i =0; i<motion_frame_limit; i++ ){
                                //    for(int j = 0; j<motion_frame_limit; j++){
                                //        cout << path[i][j] << " ";
                                //    }
                                //    cout << '\n';
                                //}
								



                            }

                        }
                    }

                    //정답 비디오 저장.
                    if (mode == 0) {
                        if (frame_count == 0) {
                            string file_name = instruction2[motion_index];

                            file_name += ".avi";
                            writer[motion_index].open(file_name, fourcc, 15, Size(1280, 720), 1);

                        }
                        if ((motion_clk + 1) % motion_frames_limit[motion_index] == 0) {
                            cout << "another_videio_started" << '\n';
                            writer[motion_index].release();
                            if (motion_index + 1 < sizeof(motion_frames_limit) / 4 /*추후 전체 프레임 상수 만들거임 8.23 */) {
                                string file_name = instruction2[++motion_index]; //instrucion[] -> legacy 동작 이름 10_26 , instruction2[] --> 지금 사용중인 동작 이름.
                                file_name += ".avi";
                                writer[motion_index].open(file_name, fourcc, 15, Size(1280, 720), 1);
                            }
                            cout << "another_videio_ended" << '\n';


                        }
                        writer[motion_index] << colorMat2;

                    }

                    //test 비디오 저장. mode 1
                    if (mode == 1) {
                        int w = motion_text[motion_index].cols;
						int h = motion_text[motion_index].rows;
                        
                        // 특정 영역 선택 (예: 좌상단 좌표 (100, 100)에서 가로 50, 세로 50 픽셀의 영역)
                        cv::Rect text_area( 0, 720 - h, w, h);
						Mat imageROI = colorMat2(cv::Rect(0, 720 - h, w, h));
						// 영상 ROI 정의
						// Rect는 사각형 영역 지정
						// 410, 270은 각각 logo의 x좌표, y좌표 시작지점
						// logo.cols, logo.rows는 로고의 끝지점
						cv::addWeighted(imageROI, 0, motion_text[motion_index], 1.0,0., imageROI);
						// 영상에 로고 붙이기
						// imageROI = 1.0*imageROI + 0.3*logo + 0
                        
                        total_image[frame_count] = colorMat2;// 모든 이미지 저장 2023.11.16
                        
                        test_writer << colorMat2;
                        if ((motion_clk + 1) % motion_frames_limit[motion_index] == 0 && frame_count) { motion_index++; }
                    }

                    //2023 11 16 for feedback video --> 피시험자 + 정답 
                    


                    /*putText(colorMat, "10 points", Point(640, 50), FONT_ITALIC, 1, (255, 0, 0), 5);*/
                    //830 실시간 점수 확인

                    resize(colorMat, colorMat, Size(1280, 720), 0, 0, INTER_LANCZOS4);
                    resize(colorMat2, colorMat2, Size(1280, 720), 0, 0, INTER_LANCZOS4);
                    Mat resMat;
                    hconcat(colorMat2, colorMat, resMat);
                    resize(resMat, resMat, Size(1280, 720), 0, 0, INTER_LANCZOS4);
                    string c4 = ".png";
                    string c5 = ",";
                    //cv::imwrite(c1+c5+c2+c5+c3+c5+c4, resMat);

                    

                    cv::imshow("image", colorMat2);
                    //int key_input = waitKey();
                    //if (key_input == 27) {
                    //    break;
                    //}

                    if (waitKey(1) > 0) {
                        break;
                    }
                    //10_12 for realtime capture

                    frame_count++;
                }


                k4a_image_release(depth);
                k4a_image_release(color_image);// Remember to release the body frame once you finish using it
                k4a_image_release(depth2);
                k4a_image_release(color_image2);

            }
            k4abt_frame_release(body_frame2);
            k4abt_frame_release(body_frame);   // Remember to release the body frame once you finish using it

        }
    } while (frame_count < max_frame_count);
    // max_frame_count -> 전체 fbx 비교모드 // motion_frame_limit 1동작 자체 비교,저장모드 10_19 //임의의 정수 : 원하는 frame 만큼 돌리고 싶을때 . 

    cout << "i'm out " << '\n';
    //2023 11 16 전체 이미지 저장 + ref 저장
    VideoWriter total_writer;
    total_writer.open("feedback_video.avi", fourcc, 15, Size(1280, 720), 1);
    
    int start_frame = 0;
    for(int i = 1; i<= MOTION_NUM; i++){
        
        for (int j = 0; j < motion_frames_limit[i]; j++) {
            Mat resMat;
			//cv::imshow("image", total_image[start_frame + j]);
            if (motion_frames_divider[i] < 1) { motion_frames_divider[i] = 1; } //2023_11_19 정답 인덱스 초과 막기.
            int motion_sync = floor(all_aligned_path[i][j] / motion_frames_divider[i]);
            cout << "ref frame : " << motion_sync + motion_frames_ref_start[i]  << '\n';
            resize(ref[motion_sync +  motion_frames_ref_start[i]], ref[motion_sync + motion_frames_ref_start[i]], Size(1280, 720), 0, 0, INTER_LANCZOS4); // 1280 844 -> 1280 720 resize

            //cout << " ref shape : " << ref[all_aligned_path[i][j]].size() << " image shape : " << total_image[start_frame + j].size() << '\n';
            //waitKey(100);

            hconcat(total_image[start_frame+j], ref[motion_sync+motion_frames_ref_start[i]], resMat);
            resize(resMat, resMat, Size(1280, 720), 0, 0, INTER_LANCZOS4);

            total_writer << resMat;
        }
        start_frame += motion_frames_limit[i];
    }
    total_writer.release();




    ofs.close(); // joint 3d 저장 kinect 1->2, kinect 2
    ofs_c.close(); //confidence level 저장 10_19
    if (mode == 1) { test_writer.release(); }
    //writer[0].release(); //7.27
    //writer[1].release();
    destroyAllWindows(); //7.27
    printf("Finished body tracking processing!\n");

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);

    k4abt_tracker_shutdown(tracker2);
    k4abt_tracker_destroy(tracker2);

    k4a_device_stop_cameras(device2);
    k4a_device_close(device2);

    return 0;
}



void  static Hierarchy_set() {
    joints_hierarchy.assign(28, 0);
    joints_hierarchy[0] = -1;
    joints_hierarchy[1] = 0, joints_hierarchy[18] = 0; joints_hierarchy[22] = 0; joints_hierarchy[2] = 1;//spine
    joints_hierarchy[3] = 2; joints_hierarchy[26] = 3; joints_hierarchy[27] = 26; //neck
    joints_hierarchy[19] = 18; joints_hierarchy[20] = 19; joints_hierarchy[21] = 20;  //left leg
    joints_hierarchy[23] = 22; joints_hierarchy[24] = 23; joints_hierarchy[25] = 24;  //right leg
    joints_hierarchy[4] = 2; joints_hierarchy[5] = 4; joints_hierarchy[6] = 5; joints_hierarchy[7] = 6;  joints_hierarchy[8] = 7;// left arm
    joints_hierarchy[11] = 2; joints_hierarchy[12] = 11; joints_hierarchy[13] = 12; joints_hierarchy[14] = 13; joints_hierarchy[15] = 14; // right arm
}
void Print2p(float x, float y) {
    cout << " Print out 2 points, x : " << x << " y : " << y << '\n';
    return;
}
void Print3p(float x, float y, float z) {
    cout << "Print out 3 points , x : " << x << " y : " << y << " z " << z << '\n';
    return;
}
void vector_print(vector<vector<vector<float>>>& v) {
    //0,2,8,15, 27, 21,25 --> end of joints connection or joint has over 3 brances 
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10 || i == 0 || i == 2 || i == 8 || i == 15 || i == 27 || i == 21 || i == 25) { continue; }
        cout << "Parent vector , index " << i << " x : " << v[i][0][0] << " y : " << v[i][0][1] << " z : " << v[i][0][2] << '\n';
        cout << "Child vector , index " << i << " x : " << v[i][1][0] << " y : " << v[i][1][1] << " z : " << v[i][1][2] << '\n';
    }
}
void CheckIntrinsicParam(k4a_calibration_t sensor_calibration) {
    vector<float> intrinsics;
    Mat cameraMatrix;
    intrinsics.assign(15, 0);
    for (int i = 0; i < 15; i++) {
        intrinsics[i] = sensor_calibration.depth_camera_calibration.intrinsics.parameters.v[i];
        cout << intrinsicArr[i] << " value : " << intrinsics[i] << '\n';
    }
    cout << '\n';
    /*cout << "cameraMatrix = " << endl << " " << cameraMatrix << endl << '\n';*/
}
void Make_vector(int index, Mat& vectors, Mat& p_set, int i_body) {
    /*cout << "index : " << index << " parnet index : " << joints_hierarchy[index] << '\n';*/
    float x_dif = p_set.at<float>(28 * i_body + index, 0) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 0);
    float y_dif = p_set.at<float>(28 * i_body + index, 1) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 1);
    float z_dif = p_set.at<float>(28 * i_body + index, 2) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 2);
    /*float div = sqrt(pow(x_dif, 2) + pow(y_dif, 2) + pow(z_dif, 2));*/

    ///*cout << "joint number : " << index << " " << " 3D vector info: " << x_dif << ", " << y_dif << ", " << z_dif << '\n';*/
    vectors.at<float>(28 * i_body + index, 0) = x_dif /*/ div*/;
    vectors.at<float>(28 * i_body + index, 1) = y_dif /*/ div*/;
    vectors.at<float>(28 * i_body + index, 2) = z_dif /*/ div*/;
}

void Make_vector2D(int index, Mat& p_set, int i_body) {
    /*cout << "index : " << index << " parnet index : " << joints_hierarchy[index] << '\n';*/
    float x_dif = p_set.at<float>(28 * i_body + index, 0) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 0);
    float y_dif = p_set.at<float>(28 * i_body + index, 1) - p_set.at<float>(28 * i_body + joints_hierarchy[index], 1);
    float div = sqrt(pow(x_dif, 2) + pow(y_dif, 2));
    /*cout << "joint number : " << index << " " << " 2D vector info : " << x_dif << ", " << y_dif << " " << '\n';*/

}

void Get_theta(int index, Mat& vectors, float theta[], int i_body) {
    
    Mat c1 = vectors.row(28 * i_body + joints_hierarchy[index]);
    Mat c2 = vectors.row(28 * i_body + index);

    float x1 = c1.at<float>(0);
    float y1 = c1.at<float>(1);
    float z1 = c1.at<float>(2);
    float div = sqrt(pow(x1, 2) + pow(y1, 2) + pow(z1, 2));

    float x2 = c2.at<float>(0);
    float y2 = c2.at<float>(1);
    float z2 = c2.at<float>(2);
    float div2 = sqrt(pow(x2, 2) + pow(y2, 2) + pow(z2, 2));

    float cos = c1.dot(c2) / div / div2;
    float rad = acos(cos);
    /*if (rad < 10e-9) {
        rad = 0;
    }
    else if (rad > Pi-10e-9) {
            
    }*/
    // cout << "index : " << index << " theta  : " << res << '\n';

    theta[28 * i_body + index] = rad;
}
void Get_n_vector(int index, Mat& vectors, Mat& nv_set, int i_body) {
    Mat c1 = vectors.row(28 * i_body + joints_hierarchy[index]);
    Mat c2 = vectors.row(28 * i_body + index);


    //Mat n_vector = c1.cross(c2); n 전체 정규화 legacy 11_01
    // float div = sqrt(pow(n_vector.at<float>(0), 2) + pow(n_vector.at<float>(1), 2) + pow(n_vector.at<float>(2), 2)); n 전체 정규화 legacy 11_01
    
    //각각 c1,c2 정규화
	float div1 = sqrt(pow(c1.at<float>(0), 2) + pow(c1.at<float>(1), 2) + pow(c1.at<float>(2), 2));
    float div2 = sqrt(pow(c2.at<float>(0), 2) + pow(c2.at<float>(1), 2) + pow(c2.at<float>(2), 2));
    
    
	c1.at<float>(0) = c1.at<float>(0) / div1;
	c1.at<float>(1) = c1.at<float>(1) / div1;
	c1.at<float>(2) = c1.at<float>(2) / div1; //11_01 정규화 o

    c2.at<float>(0) = c2.at<float>(0) / div2;
    c2.at<float> (1) = c2.at<float>(1) / div2;
   c2.at<float> (2) = c2.at<float>(2) / div2; //11_01 정규화 o

    Mat n_vector =c1.cross(c2);
    //cout << "index : " << index << " cross : " << n_vector << '\n';
    //nv_set.at<float>(28 * i_body + index, 0) = n_vector.at<float>(0) / div;
    //nv_set.at<float>(28 * i_body + index, 1) = n_vector.at<float>(1) / div;
    //nv_set.at<float>(28 * i_body + index, 2) = n_vector.at<float>(2) / div; //11_01 정규화 o
    
    nv_set.at<float>(index, 0) = n_vector.at<float>(0);
    nv_set.at<float>(index, 1) = n_vector.at<float>(1);
    nv_set.at<float>(index, 2) = n_vector.at<float>(2);
}
void Get_rotation_matrix(int index, Mat& nv_set, float th_set[], Mat& rot_set, Mat& inv_rot_set, int i_body) {
    Rodrigues(nv_set.row(28 * i_body + index) * th_set[28 * i_body + index], rot_set);
    inv_rot_set = rot_set.inv();
    //cout << "in function rotation mat :" << rot_set << '\n';
}

void Print_joint_error(Mat pset1, Mat pset2, int i_body, string c) {
    string file_name;
    if (c == "12") {
        file_name = "joint_error12.csv";
        cout << "Print joint error kinect1->kinect2" << '\n';
    }
    else if (c == "21") {
        file_name = "joint_error21.csv";
        cout << "Print joint error kinect2->kinect1" << '\n';
    }

    std::ofstream ofs(file_name);
    double error_tot = 0;
    double x_tot = 0;
    double y_tot = 0;
    double z_tot = 0;
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
        double x_joint_error = abs(pset1.at<float>(28 * i_body + i, 0) - pset2.at<float>(28 * i_body + i, 0));
        double y_joint_error = abs(pset1.at<float>(28 * i_body + i, 1) - pset2.at<float>(28 * i_body + i, 1));
        double z_joint_error = abs(pset1.at<float>(28 * i_body + i, 2) - pset2.at<float>(28 * i_body + i, 2));
        double joint_error = sqrt(pow(x_joint_error, 2) + pow(y_joint_error, 2) + pow(z_joint_error, 2));
        cout << "index : " << i << " " << " joint error :" << joint_error << " x error :" << x_joint_error << " y error : " << y_joint_error << " z error : " << z_joint_error << '\n';
        ofs << "index," << i << "," << " joint error ," << joint_error << ", x error ," << x_joint_error << ", y error , " << y_joint_error << " z error , " << z_joint_error << '\n';
        error_tot += joint_error;
        x_tot += x_joint_error; y_tot += y_joint_error; z_tot += z_joint_error;
    }
    cout << "mean error : " << error_tot / 24 << " x mean error : " << x_tot / 24 << " y mean error : " << y_tot / 24 << " z mean error :  " << z_tot / 24 << '\n';
    ofs << "mean error : ," << error_tot / 24 << ", x mean error : ," << x_tot / 24 << ", y mean error : ," << y_tot / 24 << ", z mean error :  ," << z_tot / 24 << '\n';

}


void Multiply_extrinsic_matrix(Mat& pset1, Mat& pset2, Mat transform, int i_body) {
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
        Mat newpset1 = Mat::ones(4, 1, CV_32FC1);
        Mat newpset2 = Mat::ones(4, 1, CV_32FC1);
        newpset1.at<float>(0, 0) = pset1.at<float>(28 * i_body + i, 0);
        newpset1.at<float>(1, 0) = pset1.at<float>(28 * i_body + i, 1);
        newpset1.at<float>(2, 0) = pset1.at<float>(28 * i_body + i, 2);
        newpset2 = transform * newpset1;
        pset2.at<float>(28 * i_body + i, 0) = newpset2.at<float>(0, 0);
        pset2.at<float>(28 * i_body + i, 1) = newpset2.at<float>(1, 0);
        pset2.at<float>(28 * i_body + i, 2) = newpset2.at<float>(2, 0);
        //cout << "test extrinsic result: " << newpset2.at<float>(0,0) << " " << newpset2.at<float>(1,0) << " " << newpset2.at<float>(2,0) << '\n';
    }
}

void color() {
    int i = 0;
    Scalar blue{ 255,0,0 }; Scalar orange{ 0,128,255 }; Scalar yellow{ 0,255,255 }; Scalar green{ 0,255,128 }; Scalar red{ 0,0,255 }; Scalar gray{ 160,160,160 }; Scalar sky_blue{ 76,153,0 }; Scalar bright_gray{ 160,160,160 };

    colors[0] = bright_gray; // 중심
    colors[1] = gray; // 척추1
    colors[2] = gray; // 척추2
    colors[11] = red; colors[4] = red; //쇄골    
    colors[12] = orange; colors[5] = orange; //어깨
    colors[13] = sky_blue; colors[6] = sky_blue; // 팔꿈치
    colors[14] = Scalar(128, 255, 0); colors[7] = Scalar(128, 255, 0); //손목
    colors[15] = Scalar(0, 153, 76); colors[8] = Scalar(0, 153, 76); //손
    colors[3] = Scalar(102, 0, 204); //목
    colors[26] = Scalar(255, 0, 255); //얼굴
    colors[27] = Scalar(255, 153, 255); //코
    colors[22] = red; colors[23] = orange; colors[24] = yellow; colors[25] = green;//우측하체
    colors[18] = red; colors[19] = orange; colors[20] = yellow; colors[21] = green;//좌측하체
}

void color0() {
    int i;
    for (i = 0; i < 28; i++) {
        colors0[i] = Scalar(255, 255, 255);
    }

}

static void Get_3dPoints(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set) {
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
        int num_j = i;
        float x = skeleton[i_body].joints[num_j].position.xyz.x;
        float y = skeleton[i_body].joints[num_j].position.xyz.y;
        float z = skeleton[i_body].joints[num_j].position.xyz.z; // 24*3 
        p_set.at<float>(28 * i_body + i, 0) = x;
        p_set.at<float>(28 * i_body + i, 1) = y;
        p_set.at<float>(28 * i_body + i, 2) = z;
    }
}

void Setting_confidence_level(k4abt_skeleton_t& skeleton, double  confi_level[]) {
    for (int i = 0; i < 28; i++) {
        if (i == 9 || i == 10 || i == 16 || i == 17) continue;
        if (skeleton.joints[i].confidence_level == K4ABT_JOINT_CONFIDENCE_LOW) {
            confi_level[i] = 0.1;
        }
        else if (skeleton.joints[i].confidence_level == K4ABT_JOINT_CONFIDENCE_MEDIUM) {
            confi_level[i] = 1;

        }
        else if (skeleton.joints[i].confidence_level == K4ABT_JOINT_CONFIDENCE_NONE) {
            confi_level[i] = 0.01;
        }
    }
}


//low 0.1
//medium 1
// none 0.01
void Fusion_points(Mat& p_set2, Mat& newpset2, Mat& fusion_pset, double confidence_level[], double confidence_level2[]) {
    for (int i = 0; i < 28; i++) {
        if (i == 9 || i == 10 || i == 16 || i == 17) continue;
        fusion_pset.at<float>(i, 0) = (p_set2.at<float>(i, 0) * confidence_level2[i] + newpset2.at<float>(i, 0) * confidence_level[i]) / (confidence_level[i] + confidence_level2[i]);
        fusion_pset.at<float>(i, 1) = (p_set2.at<float>(i, 1) * confidence_level2[i] + newpset2.at<float>(i, 1) * confidence_level[i]) / (confidence_level[i] + confidence_level2[i]);
        fusion_pset.at<float>(i, 2) = (p_set2.at<float>(i, 2) * confidence_level2[i] + newpset2.at<float>(i, 2) * confidence_level[i]) / (confidence_level[i] + confidence_level2[i]);
    }
}



static void  Drawing_circle(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set, k4a_calibration_t sensor_calibration, k4a_float2_t xy_color[], Mat& p_set_2D, k4abt_frame_t body_frame, Mat& colorMat,
    k4a_image_t color_image, uint8_t* color_buffer, Scalar colors[]) {
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
        int num_j = i;
        int re = 0;
        k4a_float3_t points3d;
        points3d.xyz.x = p_set.at<float>(28 * i_body + i, 0);
        points3d.xyz.y = p_set.at<float>(28 * i_body + i, 1);
        points3d.xyz.z = p_set.at<float>(28 * i_body + i, 2);

        //cout << p_set.row(i) << '\n';
        k4a_calibration_3d_to_2d(&sensor_calibration, &points3d, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &xy_color[28 * i_body + i], &re);
        p_set_2D.at<float>(28 * i_body + i, 0) = xy_color[28 * i_body + i].xy.x;
        p_set_2D.at<float>(28 * i_body + i, 1) = xy_color[28 * i_body + i].xy.y;
        uint32_t id = k4abt_frame_get_body_id(body_frame, i);
        if (color_image != NULL) {
            if (color_buffer != NULL) {
                cv::circle(colorMat, Point(xy_color[28 * i_body + i].xy.x, xy_color[28 * i_body + i].xy.y), 7, colors[i], 4, 1, 0);
            }
        }
    }
}

static void Drawing_marker(vector<k4abt_skeleton_t> skeleton, int i_body, Mat& p_set, k4a_calibration_t sensor_calibration, k4a_float2_t xy_color[], Mat& p_set_2D, k4abt_frame_t body_frame, Mat& colorMat,
    k4a_image_t color_image, uint8_t* color_buffer, Scalar colors[]) {
    for (int i = 0; i <= 27; i++) {
        if (i == 9 || i == 16 || i == 17 || i == 10) { continue; }
        int num_j = i;
        int re = 0;
        k4a_float3_t points3d;
        points3d.xyz.x = p_set.at<float>(28 * i_body + i, 0);
        points3d.xyz.y = p_set.at<float>(28 * i_body + i, 1);
        points3d.xyz.z = p_set.at<float>(28 * i_body + i, 2);

        //cout << p_set.row(i) << '\n';
        k4a_calibration_3d_to_2d(&sensor_calibration, &points3d, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &xy_color[28 * i_body + i], &re);
        uint32_t id = k4abt_frame_get_body_id(body_frame, i);
        if (color_image != NULL) {
            if (color_buffer != NULL) {
                cv::drawMarker(colorMat, Point(xy_color[28 * i_body + i].xy.x, xy_color[28 * i_body + i].xy.y), colors[i], 0, 10, 5, 0);
            }
        }
    }
}


void static Comparison_Rod(int num_bodies, Mat rod_l[]) {
    if (num_bodies != 2) {
        cout << "compare error : numbody \n";
        return;
    }
    float norm1_sum;
    Mat R1, R2;
    float diff_cossim = 0;
    float diff_rotation = 0;
    for (int j = 0; j <= 27; j++) {
        if (j == 9 || j == 16 || j == 17 || j == 10 || j == 0) { continue; }
        Rodrigues(rod_l[j], R1);
        Rodrigues(rod_l[28 * (num_bodies - 1) + j], R2);

        diff_cossim += abs((rod_l[j].dot(rod_l[28 * (num_bodies - 1) + j])) / (norm(rod_l[j], cv::NORM_L2) * norm(rod_l[28 * (num_bodies - 1) + j], cv::NORM_L2))); // cosine similarity
        /*cout << " joint " << j << " cossim diff : " << abs((rod_l[j].dot(rod_l[28 * (num_bodies - 1) + j])) / (norm(rod_l[j], cv::NORM_L2) * norm(rod_l[28 * (num_bodies - 1) + j], cv::NORM_L2))) << '\n';*/

        diff_rotation += norm(R1 - R2); // rotation similarity
        /* cout << " joint " << j << " rotation diff : "  << norm(R1 - R2) << "\n";   */

    }
    cout << "diff by cossim " << diff_cossim << "\n";
    cout << "diff by rotation: " << diff_rotation << "\n";
}
void static create_xy_table(const k4a_calibration_t* calibration, k4a_image_t xy_table)
{
    k4a_float2_t* table_data = (k4a_float2_t*)(void*)k4a_image_get_buffer(xy_table);

    int width = calibration->depth_camera_calibration.resolution_width;
    int height = calibration->depth_camera_calibration.resolution_height;

    k4a_float2_t p;
    k4a_float3_t ray;
    int valid;

    for (int y = 0, idx = 0; y < height; y++)
    {
        p.xy.y = (float)y;
        for (int x = 0; x < width; x++, idx++)
        {
            p.xy.x = (float)x;

            k4a_calibration_2d_to_3d(
                calibration, &p, 1.f, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &ray, &valid);

            if (valid)
            {
                table_data[idx].xy.x = ray.xyz.x;
                table_data[idx].xy.y = ray.xyz.y;
            }
            else
            {
                table_data[idx].xy.x = nanf("");
                table_data[idx].xy.y = nanf("");
            }
        }
    }
}

void static generate_point_cloud(const k4a_image_t depth_image,
    const k4a_image_t xy_table,
    k4a_image_t point_cloud,
    int* point_count)
{
    int width = k4a_image_get_width_pixels(depth_image);
    int height = k4a_image_get_height_pixels(depth_image);

    uint16_t* depth_data = (uint16_t*)(void*)k4a_image_get_buffer(depth_image);
    k4a_float2_t* xy_table_data = (k4a_float2_t*)(void*)k4a_image_get_buffer(xy_table);
    k4a_float3_t* point_cloud_data = (k4a_float3_t*)(void*)k4a_image_get_buffer(point_cloud);

    *point_count = 0;
    for (int i = 0; i < width * height; i++)
    {
        if (depth_data[i] != 0 && !isnan(xy_table_data[i].xy.x) && !isnan(xy_table_data[i].xy.y) && depth_data[i] < 1900 && depth_data[i] > 1000)
        {
            point_cloud_data[i].xyz.x = xy_table_data[i].xy.x * (float)depth_data[i];
            point_cloud_data[i].xyz.y = xy_table_data[i].xy.y * (float)depth_data[i];
            point_cloud_data[i].xyz.z = (float)depth_data[i];
            (*point_count)++;
        }
        else
        {
            point_cloud_data[i].xyz.x = nanf("");
            point_cloud_data[i].xyz.y = nanf("");
            point_cloud_data[i].xyz.z = nanf("");
        }
    }
}

void static write_point_cloud(const char* file_name, const k4a_image_t point_cloud, int point_count)
{
    int width = k4a_image_get_width_pixels(point_cloud);
    int height = k4a_image_get_height_pixels(point_cloud);

    k4a_float3_t* point_cloud_data = (k4a_float3_t*)(void*)k4a_image_get_buffer(point_cloud);

    // save to the ply file
    std::ofstream ofs(file_name); // text mode first
    ofs << "ply" << std::endl;
    ofs << "format ascii 1.0" << std::endl;
    ofs << "element vertex"
        << " " << point_count << std::endl;
    ofs << "property float x" << std::endl;
    ofs << "property float y" << std::endl;
    ofs << "property float z" << std::endl;
    ofs << "end_header" << std::endl;
    ofs.close();

    std::stringstream ss;
    for (int i = 0; i < width * height; i++)
    {
        if (isnan(point_cloud_data[i].xyz.x) || isnan(point_cloud_data[i].xyz.y) || isnan(point_cloud_data[i].xyz.z))
        {
            continue;
        }

        ss << (float)point_cloud_data[i].xyz.x << " " << (float)point_cloud_data[i].xyz.y << " "
            << (float)point_cloud_data[i].xyz.z << std::endl;
    }

    std::ofstream ofs_text(file_name, std::ios::out | std::ios::app);
    ofs_text.write(ss.str().c_str(), (std::streamsize)ss.str().length());



}

void MakeExtrinsicMatrix(string filename1, string filename2, Mat& transform) {

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    // Prepare input
    std::shared_ptr<geometry::PointCloud> source =
        open3d::io::CreatePointCloudFromFile(filename1);
    source->PaintUniformColor({ 1, 0.706, 0 });
    std::shared_ptr<geometry::PointCloud> target =
        open3d::io::CreatePointCloudFromFile(filename2);
    target->PaintUniformColor({ 0, 0.651, 0.929 });
    if (source == nullptr || target == nullptr) {
        utility::LogWarning("Unable to load source or target file.");
        return;

    }

    float vox = 0.1;
    double nn = 50;
    int iter = 100;
    double cor = 200;
    std::vector<double> maxcor = { cor, cor / 2 , cor / 3, cor / 4 };
    /*std::vector<double> maxcor = { cor, cor  , cor, cor };*/
    float elapse = 1e-9;
    std::vector<float> voxel_sizes = { vox, vox / 2 , vox / 4 , vox / 8 , vox / 16 };
    Eigen::Matrix4d trans;
    trans << 1, 0, 0, -1000,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    for (int k = 0; k < maxcor.size(); k++) {
        //float voxel_size = voxel_sizes[i];

        //auto source_down = source->VoxelDownSample(voxel_size);
        //source_down->EkstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
        //	voxel_size * 2.0, nn));

        //auto target_down = target->VoxelDownSample(voxel_size);
        //target_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
        //	voxel_size * 2.0, nn));

        auto result = pipelines::registration::RegistrationICP(
            *source, *target, maxcor[k], trans,
            pipelines::registration::
            TransformationEstimationPointToPoint(),
            pipelines::registration::ICPConvergenceCriteria(elapse, elapse,
                iter));
        trans = result.transformation_;

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                transform.at<float>(i, j) = trans(i, j);
            }
        }
        if (k > 2) {
            VisualizeRegistration(*source, *target, trans);
        }

    }

    std::stringstream ss;
    ss << trans;
    utility::LogInfo("Final transformation = \n{}", ss.str());

    return;
}

void Icp_go() {
    k4a_device_t device = NULL;
    k4a_device_t device2 = NULL;
    VERIFY(k4a_device_open(0, &device), "Open K4A Device 1 failed!");
    VERIFY(k4a_device_open(1, &device2), "Open K4A Device 2 failed!");

    const int32_t TIMEOUT_IN_MS = 1000000;
    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    deviceConfig.camera_fps = K4A_FRAMES_PER_SECOND_30;
    deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_720P;

    k4a_device_configuration_t deviceConfig2 = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig2.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    deviceConfig2.camera_fps = K4A_FRAMES_PER_SECOND_30;
    deviceConfig2.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    deviceConfig2.color_resolution = K4A_COLOR_RESOLUTION_720P;
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras 1 failed!");
    VERIFY(k4a_device_start_cameras(device2, &deviceConfig2), "Start K4A cameras 2 failed!");

    // sensor calibration
    k4a_calibration_t sensor_calibration;
    k4a_calibration_t sensor_calibration2;
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &sensor_calibration),
        "Get depth camera 1 calibration failed!");
    VERIFY(k4a_device_get_calibration(device2, deviceConfig2.depth_mode, deviceConfig2.color_resolution, &sensor_calibration2),
        "Get depth camera 2 calibration failed!");

    //Check intrinsics info
    cout << "Kinect 1 intrinsic: " << '\n';
    CheckIntrinsicParam(sensor_calibration);
    cout << "Kinect 2 intrinsic: " << '\n';
    CheckIntrinsicParam(sensor_calibration2);

    //Body traker config
    k4abt_tracker_t tracker = NULL;
    k4abt_tracker_t tracker2 = NULL;
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
    VERIFY(k4abt_tracker_create(&sensor_calibration, tracker_config, &tracker), "Body tracker 1 initialization failed!");
    VERIFY(k4abt_tracker_create(&sensor_calibration2, tracker_config, &tracker2), "Body tracker2 initialization failed!");

    //Making  point clouds
    k4a_capture_t capture = NULL;
    k4a_capture_t capture2 = NULL;
    std::string file_name = "point_cloud_image.ply";
    std::string file_name2 = "point_cloud_image2.ply";
    k4a_image_t depth_image = NULL;
    k4a_image_t depth_image2 = NULL;
    k4a_image_t xy_table = NULL;
    k4a_image_t xy_table2 = NULL;
    k4a_image_t point_cloud = NULL;
    k4a_image_t point_cloud2 = NULL;
    int point_count = 0;
    int point_count2 = 0;

    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
        sensor_calibration.depth_camera_calibration.resolution_width,
        sensor_calibration.depth_camera_calibration.resolution_height,
        sensor_calibration.depth_camera_calibration.resolution_width * (int)sizeof(k4a_float2_t),
        &xy_table);

    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
        sensor_calibration2.depth_camera_calibration.resolution_width,
        sensor_calibration2.depth_camera_calibration.resolution_height,
        sensor_calibration2.depth_camera_calibration.resolution_width * (int)sizeof(k4a_float2_t),
        &xy_table2);

    create_xy_table(&sensor_calibration, xy_table);
    create_xy_table(&sensor_calibration2, xy_table2);

    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
        sensor_calibration.depth_camera_calibration.resolution_width,
        sensor_calibration.depth_camera_calibration.resolution_height,
        sensor_calibration.depth_camera_calibration.resolution_width * (int)sizeof(k4a_float3_t),
        &point_cloud);
    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
        sensor_calibration2.depth_camera_calibration.resolution_width,
        sensor_calibration2.depth_camera_calibration.resolution_height,
        sensor_calibration2.depth_camera_calibration.resolution_width * (int)sizeof(k4a_float3_t),
        &point_cloud2);
    //get capture for point cloud
     // Get a capture
    switch (k4a_device_get_capture(device, &capture, TIMEOUT_IN_MS))
    {
    case K4A_WAIT_RESULT_SUCCEEDED:
        break;
    case K4A_WAIT_RESULT_TIMEOUT:
        printf("Timed out waiting for a capture\n");
        k4a_device_close(device);
    case K4A_WAIT_RESULT_FAILED:
        printf("Failed to read a capture\n");
        k4a_device_close(device);
    }
    switch (k4a_device_get_capture(device2, &capture2, TIMEOUT_IN_MS))
    {
    case K4A_WAIT_RESULT_SUCCEEDED:
        break;
    case K4A_WAIT_RESULT_TIMEOUT:
        printf("Timed out waiting for a capture\n");
        k4a_device_close(device);
    case K4A_WAIT_RESULT_FAILED:
        printf("Failed to read a capture\n");
        k4a_device_close(device);
    }
    // Get a depth image
    depth_image = k4a_capture_get_depth_image(capture);
    if (depth_image == 0)
    {
        printf("Failed to get depth image from capture\n");
        k4a_device_close(device);
    }
    depth_image2 = k4a_capture_get_depth_image(capture2);
    if (depth_image2 == 0)
    {
        printf("Failed to get depth image from capture\n");
        k4a_device_close(device);
    }
    generate_point_cloud(depth_image, xy_table, point_cloud, &point_count);
    write_point_cloud(file_name.c_str(), point_cloud, point_count);

    generate_point_cloud(depth_image2, xy_table2, point_cloud2, &point_count2);
    write_point_cloud(file_name2.c_str(), point_cloud2, point_count2);

    //make extrinsic matrix between kinect1, kinect2
    cv::Mat trans(4, 4, CV_32FC1);
    MakeExtrinsicMatrix("point_cloud_image.ply", "point_cloud_image2.ply", trans);
    cout << "extrinsic matrix:" << trans;

    //save extrinsic matirx 4X4
    cv::FileStorage fs_mat("extrinsic_matrix.xml", cv::FileStorage::WRITE);
    fs_mat << "trans" << trans;
    fs_mat.release();


    //shutdown kinect

    k4abt_tracker_shutdown(tracker);
    k4abt_tracker_destroy(tracker);

    k4a_device_stop_cameras(device);
    k4a_device_close(device);

    k4abt_tracker_shutdown(tracker2);
    k4abt_tracker_destroy(tracker2);

    k4a_device_stop_cameras(device2);
    k4a_device_close(device2);
}