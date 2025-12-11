#include <iostream>
#include <memory>
#include <vector>
#include <thread>
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

// 测试多核绑定：创建多个独立的模型实例，每个绑定到不同的NPU核心
void test_multi_core(const std::string& img_path) {
    LOG("========== Testing Multi-Core Binding ==========");
    std::string model_path = "./model/yolo11.rknn";
    detector::DetectParam detect_param = {0.25, 0.45, 114, 80};

    // 创建3个独立的YOLO11实例，每个会自动绑定到不同的NPU核心 (0, 1, 2)
    LOG("Creating 3 independent YOLO11 instances (each binds to different NPU core)...");
    std::vector<std::unique_ptr<detector::YOLO11>> models;
    for (int i = 0; i < 3; i++) {
        LOG("Creating model instance %d...", i);
        models.push_back(std::make_unique<detector::YOLO11>(model_path, logger::Level::INFO, detect_param));
    }

    cv::Mat img = cv::imread(img_path);

    // 在多个线程中并行运行推理
    LOG("Running inference on 3 models in parallel threads...");
    std::vector<std::thread> threads;
    struct timeval start_time, stop_time;

    gettimeofday(&start_time, NULL);
    for (int i = 0; i < 3; i++) {
        threads.emplace_back([&models, &img, i]() {
            for (int j = 0; j < 10; j++) {
                auto res = models[i]->inference(img);
                LOG("Thread %d, Inference %d: detected %d objects",
                    i, j, std::get<object_detect_result_list>(res).count);
            }
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    gettimeofday(&stop_time, NULL);

    LOG("Multi-core test completed: 3 threads x 10 inferences = 30 total inferences");
    LOG("Total time: %f ms, Average per inference: %f ms\n",
        (__get_us(stop_time) - __get_us(start_time)) / 1000.0,
        (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / 30);
}

// 测试模型参数复用：多个模型实例共享同一份权重
void test_share_weight(const std::string& img_path) {
    LOG("========== Testing Weight Sharing (rknn_dup_context) ==========");
    std::string model_path = "./model/yolo11.rknn";
    detector::DetectParam detect_param = {0.25, 0.45, 114, 80};

    // 创建第一个模型实例（加载完整的模型权重）
    LOG("Creating primary model instance (loads full weights)...");
    auto primary_model = std::make_unique<detector::YOLO11>(model_path, logger::Level::INFO, detect_param);

    // 创建多个共享权重的模型实例（使用rknn_dup_context）
    LOG("Creating 2 secondary model instances (sharing weights via rknn_dup_context)...");
    std::vector<std::unique_ptr<detector::YOLO11>> shared_models;
    for (int i = 0; i < 2; i++) {
        LOG("Creating shared model instance %d...", i);
        shared_models.push_back(std::make_unique<detector::YOLO11>(
            model_path, logger::Level::INFO, primary_model->get_context(), detect_param));
    }

    cv::Mat img = cv::imread(img_path);

    // 在多个线程中并行运行推理
    LOG("Running inference on 3 models (1 primary + 2 shared) in parallel threads...");
    std::vector<std::thread> threads;
    struct timeval start_time, stop_time;

    gettimeofday(&start_time, NULL);

    // Primary model thread
    threads.emplace_back([&primary_model, &img]() {
        for (int j = 0; j < 10; j++) {
            auto res = primary_model->inference(img);
            LOG("Primary model, Inference %d: detected %d objects",
                j, std::get<object_detect_result_list>(res).count);
        }
    });

    // Shared model threads
    for (int i = 0; i < 2; i++) {
        threads.emplace_back([&shared_models, &img, i]() {
            for (int j = 0; j < 10; j++) {
                auto res = shared_models[i]->inference(img);
                LOG("Shared model %d, Inference %d: detected %d objects",
                    i, j, std::get<object_detect_result_list>(res).count);
            }
        });
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    gettimeofday(&stop_time, NULL);

    LOG("Weight sharing test completed: 3 threads x 10 inferences = 30 total inferences");
    LOG("Total time: %f ms, Average per inference: %f ms\n",
        (__get_us(stop_time) - __get_us(start_time)) / 1000.0,
        (__get_us(stop_time) - __get_us(start_time)) / 1000.0 / 30);
}

void print_usage(const char* program_name) {
    LOG("Usage: %s [test_type] [image_path]", program_name);
    LOG("  test_type:");
    LOG("    yolo11     - Test YOLO11 single model");
    LOG("    yolov5     - Test YOLOv5 single model");
    LOG("    all        - Test both YOLO11 and YOLOv5");
    LOG("    multicore  - Test multi-core binding (3 independent models)");
    LOG("    share      - Test weight sharing (rknn_dup_context)");
    LOG("  image_path: path to test image (default: ./model/car.jpg)");
}

int main(int argc, char* argv[]){
    std::string img_path = "./model/car.jpg";
    std::string test_type = "all";  // 默认测试全部

    // 解析命令行参数
    if (argc >= 2) {
        test_type = argv[1];
    }
    if (argc >= 3) {
        img_path = argv[2];
    }

    LOG("Image path: %s", img_path.c_str());
    LOG("Test type: %s", test_type.c_str());

    if (test_type == "yolo11") {
        test_yolo11(img_path);
    } else if (test_type == "yolov5") {
        test_yolov5(img_path);
    } else if (test_type == "all") {
        test_yolo11(img_path);
        test_yolov5(img_path);
    } else if (test_type == "multicore") {
        test_multi_core(img_path);
    } else if (test_type == "share") {
        test_share_weight(img_path);
    } else {
        LOGE("Unknown test type: %s", test_type.c_str());
        print_usage(argv[0]);
        return -1;
    }

    return 0;
}
