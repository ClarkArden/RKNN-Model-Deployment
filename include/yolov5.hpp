#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rknn_model.hpp"
#include "yolo11.hpp"  // For DetectParam

#define LABEL_NALE_TXT_PATH_V5 "./model/coco_80_labels_list.txt"

namespace detector
{
    extern std::string out_path;

    class YOLO5 : public rknn::Model {
    public:
        // Standard constructor - creates new rknn context
        YOLO5(std::string model_path, logger::Level level, DetectParam detect_param);
        // Constructor with context sharing - reuses weights from existing context
        YOLO5(std::string model_path, logger::Level level, rknn_context* ctx_in, DetectParam detect_param);
        ~YOLO5();

        // infer method for thread pool (returns object_detect_result_list directly)
        object_detect_result_list infer(cv::Mat img);

        virtual bool preprocess() override;
        virtual bool postprocess() override;
        virtual void draw(cv::Mat img) override;

        // YOLOv5 anchor-based processing functions
        int process_i8(int8_t *input, int *anchor, int grid_h, int grid_w,
                       int stride, int32_t zp, float scale,
                       std::vector<float> &boxes,
                       std::vector<float> &objProbs,
                       std::vector<int> &classId,
                       float threshold);

        int process_u8(uint8_t *input, int *anchor, int grid_h, int grid_w,
                       int stride, int32_t zp, float scale,
                       std::vector<float> &boxes,
                       std::vector<float> &objProbs,
                       std::vector<int> &classId,
                       float threshold);

        int process_fp32(float *input, int *anchor, int grid_h, int grid_w,
                         int stride,
                         std::vector<float> &boxes,
                         std::vector<float> &objProbs,
                         std::vector<int> &classId,
                         float threshold);

        int init_post_process();

    private:
        DetectParam m_detectParam;
        image_rect_t m_pads;
        float m_scale;
        cv::Mat m_resized_img;

        std::unique_ptr<object_detect_result_list> m_odReseultsPtr;
        std::vector<float> m_filterBoxes;
        std::vector<float> m_objProbs;
        std::vector<int> m_classId;
    };

}; // namespace detector
