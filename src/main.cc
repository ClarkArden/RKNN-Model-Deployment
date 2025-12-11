#include <iostream>
#include <memory>
#include <sys/time.h>
#include "yolo11.hpp"
#include "yolov5.hpp"

// 获取微秒级时间戳
static int64_t __get_us(struct timeval t) {
    return (t.tv_sec * 1000000 + t.tv_usec);
}

void test_yolo11(const std::string& img_path) {
    LOG("========== Testing YOLO11 ==========");
    std::string model_path = "./model/yolo11.rknn";
    detector::DetectParam detect_param = {0.25, 0.45, 114, 80};

    std::unique_ptr<rknn::Model> yolo11 = std::make_unique<detector::YOLO11>(model_path, logger::Level::DEBUG, detect_param);

    // 延迟测试
    int test_count = 20;
    struct timeval start_time, stop_time;

    cv::Mat img = cv::imread(img_path);
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < test_count; ++i) {
        auto res = yolo11->inference(img);
        LOGD("YOLO11: the number of object = %d", std::get<object_detect_result_list>(res).count);
    }
    gettimeofday(&stop_time, NULL);

    LOG("YOLO11: loop count = %d , average run %f ms\n", test_count,
           (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

    // 最后一次推理用于绘制结果
    detector::out_path = "./out_yolo11.jpg";
    yolo11->inference(img);
    yolo11->draw(img);
}

void test_yolov5(const std::string& img_path) {
    LOG("========== Testing YOLOv5 ==========");
    std::string model_path = "./model/yolov5.rknn";
    detector::DetectParam detect_param = {0.25, 0.45, 114, 80};

    std::unique_ptr<rknn::Model> yolov5 = std::make_unique<detector::YOLO5>(model_path, logger::Level::DEBUG, detect_param);

    // 延迟测试
    int test_count = 20;
    struct timeval start_time, stop_time;

    cv::Mat img = cv::imread(img_path);
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < test_count; ++i) {
        auto res = yolov5->inference(img);
        LOGD("YOLOv5: the number of object = %d", std::get<object_detect_result_list>(res).count);
    }
    gettimeofday(&stop_time, NULL);

    LOG("YOLOv5: loop count = %d , average run %f ms\n", test_count,
           (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

    // 最后一次推理用于绘制结果
    detector::out_path = "./out_yolov5.jpg";
    yolov5->inference(img);
    yolov5->draw(img);
}

int main(int argc, char* argv[]){
    std::string img_path = "./model/car.jpg";
    std::string model_type = "all";  // 默认测试全部

    // 解析命令行参数
    if (argc >= 2) {
        model_type = argv[1];
    }
    if (argc >= 3) {
        img_path = argv[2];
    }

    LOG("Image path: %s", img_path.c_str());
    LOG("Model type: %s", model_type.c_str());

    if (model_type == "yolo11" || model_type == "all") {
        test_yolo11(img_path);
    }

    if (model_type == "yolov5" || model_type == "all") {
        test_yolov5(img_path);
    }

    if (model_type != "yolo11" && model_type != "yolov5" && model_type != "all") {
        LOGE("Unknown model type: %s", model_type.c_str());
        LOG("Usage: %s [model_type] [image_path]", argv[0]);
        LOG("  model_type: yolo11, yolov5, all (default: all)");
        LOG("  image_path: path to test image (default: ./model/car.jpg)");
        return -1;
    }

    return 0;
}