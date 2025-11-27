#include <iostream>
#include <memory>
#include <sys/time.h>
#include "yolo11.hpp"

// 获取微秒级时间戳
static int64_t __get_us(struct timeval t) {
    return (t.tv_sec * 1000000 + t.tv_usec);
}

int main(){
    std::string model_path = "./model/yolo11.rknn";
    std::string  img_path =  "./model/car.jpg";

    detector::DetectParam detect_param = {0.25, 0.45, 114, 80};

    std::unique_ptr<rknn::Model> yolo11 = std::make_unique<detector::YOLO11>(model_path, logger::Level::DEBUG, detect_param);

    // 延迟测试
    int test_count = 20;
    struct timeval start_time, stop_time;

    cv::Mat img = cv::imread(img_path);
    gettimeofday(&start_time, NULL);
    for (int i = 0; i < test_count; ++i) {
        auto res = yolo11->inference(img);
        LOGD("the number of object = %d", std::get<object_detect_result_list>(res).count);
    }
    gettimeofday(&stop_time, NULL);

    LOGD("loop count = %d , average run %f ms\n", test_count,
           (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / test_count);

    // 最后一次推理用于绘制结果
    yolo11->inference(img);
    yolo11->draw(img);

    return 0;
}