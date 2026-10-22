#ifndef AI_DETECT_H
#define AI_DETECT_H

/*头文件*/
//控制台输入输出
#include <iostream>
//字符串 向量 队列
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <atomic>
#include <iomanip>
//日志输出
#include <dji_logger.h>
//用于mkdir
#include <sys/stat.h>
#include <sys/types.h>
//SDK基础类型定义
#include "dji_typedef.h"
//视频流发送接口
#include "dji_payload_camera.h"
//高速通道带宽配置
#include "dji_high_speed_data_channel.h"
//时间戳、线程、时间格式化
#include <chrono>
#include <thread>
#include <ctime>
#include <sstream>
#include <cstdlib>
//文件读写功能
#include <fstream>

/*  yolo 相关内容   */
//标准输入输出
#include <iostream>
//标准字符串
#include <string>
//yolo相关内容
#include "YOLOv11.h"


//定义类别标签
static const char* s_classLables[] = {
    "sangu"
};

//定义日期名
static const char* s_invalidLables[] = {
    "XXX", "WW", "YYYYYYYYYYY", "ZZZZZZZZ"
};

#ifdef OPEN_CV_INSTALLED
//图像处理与DNN推理
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>

//yolo检测封装
#include "liveview/image_processor_yolovfastest.hpp"
#endif
/*ROS2头文件*/
#include "rclcpp/rclcpp.hpp"
#include "uav_detect_result/msg/uav_detect_result.hpp"

/* Private constants ---------------------------------------------------------*/
//标签数量常量
#define YOLO_LABLES_NUM       1
//决定注册Pilot的类别数
#define INVALID_CLASS_NUM     4

/*  YOLO模型相关内容 START    */
// 写视频回调函数

/**
 * @brief Setting up Tensorrt logger
 */
// TensorRT 日志回调
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

YOLOv11 YOLO_("/home/dji/DJI_workshop/LZP_PSDK/customed_yolo_model/best-yolo26.engine",logger);
/*  YOLO模型相关内容 END      */
// 视频保存内容
cv::VideoWriter writer;
// 多线程保存

/*  PSDK视频流相关内容 START    */
//接受回调函数
void DjiLiveview_RcvImageCallback(E_DjiLiveViewCameraPosition position, const uint8_t *buf, uint32_t len, T_DjiLiveviewImageInfo imageInfo);
//编码器回调函数 存储H264视频
void DjiLiveview_EncoderUseCallback(const uint8_t *buf, uint32_t len);
//识别线程
void* DjiLiveview_ObjectDetectionThread(void * arg);
//显示线程
void* DjiLiveview_DisplayThread(void * arg);
//保存线程
void* DjiLiveview_RecordThread(void * arg);
//获取时间戳 用于H264文件
std::string getCurrentTimestamp();

//H264码流输出为文件
void outH264Tofile(const uint8_t *buf, int32_t len);
//YUV码流输出为文件
void outYUVTofile(const uint8_t *buf, int32_t len);
//定义系统级变量
T_DjiOsalHandler *osalHandler;
//定义错误代码
T_DjiReturnCode returnCode;
//无人机位置变量
E_DjiMountPosition mountPosition = DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1;
//定义视频流来源
int mediaSource = 7; //support 0(app liveview)/1(1080p)/7(4k) for H30 camera
//视频源设定
E_DjiLiveViewCameraSource MediaResource;
//相机位置
E_DjiLiveViewCameraPosition CameraPostion;

//H264码流的文件输出流
static std::ofstream outFileH264;
//YUV原始图像数据的文件输出流
static std::ofstream outFileYUV;

#ifdef OPEN_CV_INSTALLED
    inline static ImageProcessorYolovFastest processor{"YOLOvFastest"};
    //需要保存的照片队列
    static std::queue<cv::Mat> output_queue;
    //保存视频队列
    static std::queue<cv::Mat> record_queue;
    //照片队列
    static std::queue<cv::Mat> s_imageQueue;
    //元数据队列
    static std::queue<T_DjiLiveViewStandardMetaData *> s_metaQueue;
    //检测线程的句柄，用来掌控检测线程
    T_DjiTaskHandle my_procThreadHandle;
    //元数据队列互斥锁
    T_DjiMutexHandle my_metaQueueMutexHandle;
    //图像队列互斥锁
    T_DjiMutexHandle my_imageQueueMutexHandle;
    //输出图像队列互斥锁
    T_DjiMutexHandle output_imageQueueMutexHandle;
    //视频保存队列互斥锁
    T_DjiMutexHandle record_imageQueueMutexHandle;
    //写视频线程的句柄，用来掌控写视频线程
    T_DjiTaskHandle output_ThreadHandle;
    //保存视频线程句柄
    T_DjiTaskHandle record_ThreadHandle;
#endif

/*  PSDK视频流相关内容 END    */

//保存视频路径
static std::string g_record_video_path;
//退出标志
static std::atomic_bool g_shutdown{false};
//检测结果发布者
static rclcpp::Publisher<uav_detect_result::msg::UavDetectResult>::SharedPtr g_detect_pub;

//推理频率上限
static constexpr double kMaxInferHz = 60.0;



class ai_detect:public rclcpp::Node{
    public:
    //构造函数
    ai_detect(int argc,char ** argv):rclcpp::Node("ai_detect")
    {
        CameraPostion = DJI_LIVEVIEW_CAMERA_POSITION_NO_1;
        //H30T相机位于位置1
        mountPosition = DJI_MOUNT_POSITION_PAYLOAD_PORT_NO1;
        //设置H30T变焦视频流
        MediaResource = DJI_LIVEVIEW_CAMERA_SOURCE_M30T_ZOOM;

        osalHandler = DjiPlatform_GetOsalHandler();
        // 输出用户选择的码流
        USER_LOG_INFO("Input cammera sourece: ");
        switch(mediaSource)
        {
            case 0:USER_LOG_INFO("app liveview");break;
            case 1:USER_LOG_INFO("1080P for H30T");break;
            case 7:USER_LOG_INFO("4k for H30T");break;
        }

        // 生成带时间戳的输出文件名
        std::string timestamp = getCurrentTimestamp();

        // 创建输出目录，避免目录不存在导致的写文件失败
        mkdir ("data", 0755);
        //识别结果视频保存名
        g_record_video_path = "data/yolo_annotated_" + timestamp + ".mp4";
        // 广角端视频流保存名
        std::string h264FileName = "data/output_" + timestamp + ".h264";
        // 变焦段视频流保存名
        std::string zoomh264FileName = "data/zoomoutput_" + timestamp + ".h264";
        // 打开H264输出文件，保存码流
        outFileH264.open(h264FileName, std::ios::out | std::ios::binary | std::ios::app);
        if (!outFileH264) {
            std::cerr << "cant open " << h264FileName << std::endl;
        }

        //检测结果发布者（无论是否启用OpenCV都创建话题）
        g_detect_pub = this->create_publisher<uav_detect_result::msg::UavDetectResult>(
            "yolo_detect/result", 10);

        #ifdef OPEN_CV_INSTALLED
            // 创建my_metaQueueMutexHandle和my_imageQueueMutexHandle互斥锁
            osalHandler->MutexCreate(&my_metaQueueMutexHandle);
            osalHandler->MutexCreate(&my_imageQueueMutexHandle);
            osalHandler->MutexCreate(&output_imageQueueMutexHandle);
            osalHandler->MutexCreate(&record_imageQueueMutexHandle);
            osalHandler->TaskCreate("objectDetectionTask",DjiLiveview_ObjectDetectionThread,1024*1024,NULL, &my_procThreadHandle);

            osalHandler->TaskCreate("display_yolo",DjiLiveview_DisplayThread,1024*1024,NULL, &output_ThreadHandle);
            osalHandler->TaskCreate("record_yolo",DjiLiveview_RecordThread,1024*1024,NULL, &record_ThreadHandle);
            if (processor.Init() != 0) {
                std::cerr << "Failed to initialize the processor." << std::endl;
                return ;
            }
        #endif 
        //高带宽通道 三类流量带宽占比
        const T_DjiDataChannelBandwidthProportionOfHighspeedChannel bandwidthProportionOfHighspeedChannel =
            {10, 60, 30};
        //配置通道带宽
        returnCode = DjiHighSpeedDataChannel_SetBandwidthProportion(bandwidthProportionOfHighspeedChannel);
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
        {
            USER_LOG_ERROR("Liveview init failed, HighSpeed channel init error: 0x%08llX", returnCode);
            return;
        }

        //初始化
        USER_LOG_INFO("step 1: init liveview");
        returnCode = DjiLiveview_Init();
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
        {
            USER_LOG_ERROR("Liveview init faild, ret: 0x%08llX", returnCode);
        }

        //注册标签
        #ifdef OPEN_CV_INSTALLED
            returnCode = DjiLiveview_RegUserAiTargetLableList(YOLO_LABLES_NUM, s_classLables);
        #else
            returnCode = DjiLiveview_RegUserAiTargetLableList(INVALID_CLASS_NUM, s_invalidLables);
        #endif
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
        {
            USER_LOG_ERROR("Reg ai target lable faild, ret: 0x%08llX", returnCode);
        }

        //启动解码，将视频流解码为图像流订阅，并注册回调
        USER_LOG_INFO("step 2: reg encoder callback");
        returnCode = DjiLiveview_RegEncoderCallback(DjiLiveview_EncoderUseCallback);
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
        {
            USER_LOG_ERROR( "reg Yuv Encoder callback faild, ret: 0x%08llX", returnCode);
        }
        //启动YUV流
        USER_LOG_INFO("step 3:start yuv stream");
        returnCode = DjiLiveview_StartImageStream(CameraPostion, MediaResource,
                                                PIXFMT_RGB_PACKED ,DjiLiveview_RcvImageCallback);
        if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
        {
            USER_LOG_ERROR( "start to subscribe YUV stream failed, ret: 0x%08llX", returnCode);
        }
        
    }
    ~ai_detect()
    {
    }
    private:

};

//将像素坐标转换为 1/10000 的比例
static inline uint16_t ToDjiRatio(int value, int total) {
    if (total <= 0) {
        return 0;
    }
    double ratio = static_cast<double>(value) / static_cast<double>(total);
    if (ratio < 0.0) {
        ratio = 0.0;
    } else if (ratio > 1.0) {
        ratio = 1.0;
    }
    return static_cast<uint16_t>(ratio * 10000.0);
}

static inline std::string BuildDetectResultJson(const std::vector<Detection>& objects,
                                                int img_w,
                                                int img_h) {
    std::ostringstream oss;
    oss << "{";
    oss << "\"img_w\":" << img_w << ",\"img_h\":" << img_h << ",";
    oss << "\"detections\":[";
    const size_t label_count = sizeof(s_classLables) / sizeof(s_classLables[0]);
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& det = objects[i];
        int x = std::max(0, det.bbox.x);
        int y = std::max(0, det.bbox.y);
        int w = std::max(0, det.bbox.width);
        int h = std::max(0, det.bbox.height);
        if (x + w > img_w) {
            w = std::max(0, img_w - x);
        }
        if (y + h > img_h) {
            h = std::max(0, img_h - y);
        }
        const char* label = "unknown";
        if (det.class_id >= 0 && static_cast<size_t>(det.class_id) < label_count) {
            label = s_classLables[det.class_id];
        }
        if (i > 0) {
            oss << ",";
        }
        oss << "{"
            << "\"id\":" << i
            << ",\"class_id\":" << det.class_id
            << ",\"label\":\"" << label << "\""
            << ",\"conf\":" << std::fixed << std::setprecision(3) << det.conf
            << ",\"x\":" << x
            << ",\"y\":" << y
            << ",\"w\":" << w
            << ",\"h\":" << h
            << "}";
    }
    oss << "]}";
    return oss.str();
}

//处理回调函数
void* DjiLiveview_ObjectDetectionThread(void * arg) {
    //创建错误代码
    T_DjiReturnCode DjiStat;
    //创建系统句柄
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();

    const auto infer_period = std::chrono::duration<double>(1.0 / kMaxInferHz);
    auto last_infer_start = std::chrono::steady_clock::now() - infer_period;
    while(1) {
        #ifdef OPEN_CV_INSTALLED
        //启动imag线程锁
        osalHandler->MutexLock(my_imageQueueMutexHandle);
        if (s_imageQueue.empty()) {
            //如果没有图片则取消image线程锁
            osalHandler->MutexUnlock(my_imageQueueMutexHandle);
            continue;
        }
        //丢弃旧帧，只保留最新一帧
        while (s_imageQueue.size() > 1) {
            s_imageQueue.pop();
        }
        cv::Mat bgr_image = s_imageQueue.front();
        s_imageQueue.pop();
        //完成东西获取的任务，取消image线程锁
        osalHandler->MutexUnlock(my_imageQueueMutexHandle);
 
        //限频到 60Hz（超过则睡眠丢帧）
        auto now = std::chrono::steady_clock::now();
        auto elapsed = now - last_infer_start;
        if (elapsed < infer_period) {
            std::this_thread::sleep_for(infer_period - elapsed);
        }
        last_infer_start = std::chrono::steady_clock::now();

        /* 自定义yolo模型进行推理 START*/
        cv::Mat customed_yolo_bgr = bgr_image;
        YOLO_.preprocess(customed_yolo_bgr);
        // 计算推理时间
        auto start = std::chrono::system_clock::now();
        YOLO_.infer();
        auto end = std::chrono::system_clock::now();
        // 创建存储容器
        vector<Detection> objects;
        
        YOLO_.postprocess(objects);
        YOLO_.draw(customed_yolo_bgr, objects);

        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        printf("cost %2.4lf ms\n", tc);

        //将数据推入输出队列中
        osalHandler->MutexLock(output_imageQueueMutexHandle);
        while (output_queue.size() > 30) {
            output_queue.pop();
        }
        output_queue.push(customed_yolo_bgr);
        osalHandler->MutexUnlock(output_imageQueueMutexHandle);

        //将数据推入视频保存队列中
        osalHandler->MutexLock(record_imageQueueMutexHandle);
        while (record_queue.size() > 120) {
            record_queue.pop();
        }
        record_queue.push(customed_yolo_bgr);
        osalHandler->MutexUnlock(record_imageQueueMutexHandle);

        
        /* 自定义yolo模型进行推理 END  */

        /*  PSDK原始识别、编码内容  START*/

        //定义存储BGR的图像
        bgr_image = customed_yolo_bgr;
        //将转化的BGR转变为share_ptr，方便在内部异步引用
        std::shared_ptr<cv::Mat> image_ptr = std::make_shared<cv::Mat>(bgr_image);

        //定义接收检测结果的容器
        std::vector<T_DjiLiveViewBoundingBox> bounding_boxes;
        
        //推理
        //processor.Process(image_ptr, bounding_boxes);

        //使用自定义模型的结果填充 bounding_boxes
        bounding_boxes.reserve(objects.size());
        const int img_w = bgr_image.cols;
        const int img_h = bgr_image.rows;
        for (size_t i = 0; i < objects.size(); ++i) {
            const auto& det = objects[i];
            int x = std::max(0, det.bbox.x);
            int y = std::max(0, det.bbox.y);
            int w = std::max(0, det.bbox.width);
            int h = std::max(0, det.bbox.height);
            if (x + w > img_w) {
                w = std::max(0, img_w - x);
            }
            if (y + h > img_h) {
                h = std::max(0, img_h - y);
            }
            T_DjiLiveViewBoundingBox box{};
            box.id = static_cast<uint16_t>(i);
            box.type = static_cast<uint8_t>(det.class_id);
            box.state = DJI_LIVEVIEW_OBJ_STATE_TRACKED;
            box.box.cx = ToDjiRatio(x + w / 2, img_w);
            box.box.cy = ToDjiRatio(y + h / 2, img_h);
            box.box.w = ToDjiRatio(w, img_w);
            box.box.h = ToDjiRatio(h, img_h);
            box.box.distance = 0;
            bounding_boxes.push_back(box);
        }

        
        //记录存在多少个检测框
        T_DjiLiveViewStandardMetaData *metaData = (T_DjiLiveViewStandardMetaData *)malloc(
            sizeof(T_DjiLiveViewStandardMetaData) + bounding_boxes.size() * sizeof(T_DjiLiveViewBoundingBox));
        if (metaData) {
            metaData->boxCount = static_cast<uint8_t>(bounding_boxes.size());
            for (size_t i = 0; i < bounding_boxes.size(); i++) {
                metaData->boxData[i] = bounding_boxes[i];
            }
        }

        if (metaData) {
            DjiLiveview_SendAiMetaToPilot(metaData);
        }

        //开启元数据队列锁
        osalHandler->MutexLock(my_metaQueueMutexHandle);
        //传入元数据
        if (metaData) {
            s_metaQueue.push(metaData);
        }
        //取消元数据队列锁
        osalHandler->MutexUnlock(my_metaQueueMutexHandle);

        //发布ROS2识别结果
        if (g_detect_pub) {
            uav_detect_result::msg::UavDetectResult msg;
            msg.num = static_cast<int64_t>(objects.size());
            msg.content = BuildDetectResultJson(objects, img_w, img_h);
            g_detect_pub->publish(msg);
        }
        /*  PSDK原始识别、编码内容  END*/
        #else
            break;
        #endif
    }
    return NULL;
}

//保存线程：写入识别后的视频
void* DjiLiveview_RecordThread(void * arg) {
#ifdef OPEN_CV_INSTALLED
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();
    const double record_fps = kMaxInferHz;
    cv::VideoWriter local_writer;

    while (!g_shutdown.load()) {
        cv::Mat frame;
        osalHandler->MutexLock(record_imageQueueMutexHandle);
        if (!record_queue.empty()) {
            frame = record_queue.front();
            record_queue.pop();
        }
        osalHandler->MutexUnlock(record_imageQueueMutexHandle);

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        if (!local_writer.isOpened()) {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            if (!local_writer.open(g_record_video_path, fourcc, record_fps, frame.size(), true)) {
                std::cerr << "Failed to open video writer: " << g_record_video_path << std::endl;
                return NULL;
            }
        }

        local_writer.write(frame);
    }

    if (local_writer.isOpened()) {
        local_writer.release();
    }
#endif
    return NULL;
}

//显示线程：实时展示检测结果
void* DjiLiveview_DisplayThread(void * arg) {
#ifdef OPEN_CV_INSTALLED
    if (std::getenv("DISPLAY") == nullptr) {
        std::cerr << "DISPLAY not set, skip OpenCV window display." << std::endl;
        return NULL;
    }

    cv::namedWindow("YOLO Detect", cv::WINDOW_NORMAL);
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();

    while (1) {
        cv::Mat frame;
        osalHandler->MutexLock(output_imageQueueMutexHandle);
        if (!output_queue.empty()) {
            while (output_queue.size() > 1) {
                output_queue.pop();
            }
            frame = output_queue.front();
            output_queue.pop();
        }
        osalHandler->MutexUnlock(output_imageQueueMutexHandle);

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        cv::imshow("YOLO Detect", frame);
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }

    cv::destroyWindow("YOLO Detect");
#endif
    return NULL;
}

//编码回调保存函数
void DjiLiveview_EncoderUseCallback(const uint8_t *buf, uint32_t len){
    //定义错误代码
    T_DjiReturnCode returnCode;
    //将H264文件存储到本地
    outH264Tofile(buf, len);

    //发送编码后的视频流
    returnCode = DjiPayloadCamera_SendVideoStream(buf, len);
    if (returnCode != DJI_ERROR_SYSTEM_MODULE_CODE_SUCCESS)
    {
        USER_LOG_ERROR("failed to send video to pilot, ret: 0x%08llX", returnCode);
    }
}

//图像帧处理回调函数
void DjiLiveview_RcvImageCallback(E_DjiLiveViewCameraPosition position, const uint8_t *buf, uint32_t len, T_DjiLiveviewImageInfo imageInfo){

    //定义错误代码
    T_DjiReturnCode DjiStat;
    //输出长度变量
    uint32_t OutPutLen;
    //目标框容器
    std::vector<T_DjiLiveViewBoundingBox> bounding_boxes;
    //AI元数据指针
    T_DjiLiveViewStandardMetaData * metaData = nullptr;
    //获取系统级指针
    T_DjiOsalHandler *osalHandler = DjiPlatform_GetOsalHandler();

#ifdef OPEN_CV_INSTALLED
    //将图像缓冲区封装为一个opencv的cv::Mat (输入为 RGB)
    cv::Mat rgb_image(imageInfo.height, imageInfo.width, CV_8UC3, const_cast<uint8_t*>(buf));
    cv::Mat bgr_image;
    cv::cvtColor(rgb_image, bgr_image, cv::COLOR_RGB2BGR);
    //克隆一份备用
    cv::Mat bgr_image_copy = bgr_image.clone();
    //启用image队列锁
    osalHandler->MutexLock(my_imageQueueMutexHandle);
    //消费者无法消化，丢弃一部分帧
    while (s_imageQueue.size() > 30) {
        USER_LOG_WARN("The image queue is full. Drop this strike.");
        s_imageQueue.pop();
    }
    //推入新的帧
    s_imageQueue.push(bgr_image_copy);
    //取消image队列锁
    osalHandler->MutexUnlock(my_imageQueueMutexHandle);

    //启用元数据队列锁
    osalHandler->MutexLock(my_metaQueueMutexHandle);
    //如果元数据不为空，则取出一个元数据
    if (!s_metaQueue.empty()) {
        metaData = s_metaQueue.front();
        s_metaQueue.pop();
    }
    //取消元数据队列锁
    osalHandler->MutexUnlock(my_metaQueueMutexHandle);

    //将RGB帧编码为H264并附带元数据
    DjiLiveview_EncodeAFrameToH264(buf, len, imageInfo, metaData);
    //用完后释放元数据内存
    if(metaData != nullptr) free(metaData);

#else
//原本无Opencv的处理部分
#endif
}

//获取时间戳
std::string getCurrentTimestamp() {
    std::time_t now = std::time(nullptr);
    std::tm* now_tm = std::localtime(&now);

    std::ostringstream oss;
    oss << (now_tm->tm_year + 1900)
        << (now_tm->tm_mon + 1)
        << now_tm->tm_mday
        << now_tm->tm_hour
        << now_tm->tm_min;

    return oss.str();
}

//H264码流输出为文件
void outH264Tofile(const uint8_t *buf, int32_t len) {
    if (!outFileH264) {
        USER_LOG_ERROR( "output.h264 is not open");
        return;
    }
    outFileH264.write(reinterpret_cast<const char *>(buf), len);
}

//YUV码流输出为文件
void outYUVTofile(const uint8_t *buf, int32_t len) {
    if (!outFileYUV) {
        USER_LOG_ERROR( "outyuv.h264 is not open");
        return;
    }
    outFileYUV.write(reinterpret_cast<const char *>(buf), len);
}

#endif
 