#include "camera_manager.h"
//应用程序的核心框架
#include "application.hpp"
//PSDK的日志模块
#include "dji_logger.h"
//获取无人机的基本信息
#include "dji_aircraft_info.h"
//C标准头文件 捕获信号后实现平稳退出
#include "signal.h"
//PSDK 头文件
#include "dji_platform.h"
//云台管理头文件
#include "gimbal_manager.h"
//相机控制器头文件

#include "rclcpp/rclcpp.hpp"

#include "ai_detect.h"

#include "std_msgs/msg/string.hpp"
#include <atomic>



int main(int argc, char **argv)
{
    // 初始化ros2
    rclcpp::init(argc,argv);

    //创建signal，以便安全退出
    signal(SIGINT, [](int signalNum) -> void {
        (void)signalNum;
        g_shutdown.store(true);
        rclcpp::shutdown();
    });
    //进行系统初始化
    Application application(argc, argv);

    auto ai_detect_node = std::make_shared<ai_detect>(argc,argv);
    // 创建实例并开始运行
    rclcpp::spin(ai_detect_node);

    rclcpp::shutdown();

    return 0;
}
