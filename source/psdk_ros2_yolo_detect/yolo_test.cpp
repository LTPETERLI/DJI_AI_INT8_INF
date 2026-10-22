#include "YOLOv11.h"
#include "logging.h"
#include <opencv2/opencv.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
const char* kDefaultEnginePath =
    "/home/dji/DJI_workshop/LZP_PSDK/customed_yolo_model/yolo11x_sangu_best.engine";
const char* kDefaultOutputPath =
    "/media/peter/My Passport/电网训练工程/data/4-defect/images/train/Broken_strands#0010.jpg";
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n"
                  << "  yolo_detect_test <image_path>\n"
                  << "  yolo_detect_test <engine_path> <image_path> [output_path]\n";
        return 1;
    }

    std::string engine_path = kDefaultEnginePath;
    std::string image_path;
    std::string output_path = kDefaultOutputPath;

    if (argc == 2) {
        image_path = argv[1];
    } else {
        engine_path = argv[1];
        image_path = argv[2];
        if (argc >= 4) {
            output_path = argv[3];
        }
    }

    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return 1;
    }
    engine_file.close();

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return 1;
    }

    Logger logger;
    YOLOv11 yolo(engine_path, logger);

    yolo.preprocess(image);
    auto start = std::chrono::high_resolution_clock::now();
    yolo.infer();
    auto end = std::chrono::high_resolution_clock::now();

    std::vector<Detection> objects;
    yolo.postprocess(objects);
    yolo.draw(image, objects);

    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    std::cout << "Detections: " << objects.size() << "\n";
    std::cout << "Infer time: " << ms << " ms\n";

    if (!cv::imwrite(output_path, image)) {
        std::cerr << "Failed to write output: " << output_path << std::endl;
        return 1;
    }

    std::cout << "Output saved to: " << output_path << "\n";
    return 0;
}
