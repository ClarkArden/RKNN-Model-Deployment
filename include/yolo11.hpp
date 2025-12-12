#pragma once

#include <memory>
#include <string>
#include <vector>

#include "rknn_model.hpp"

#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"


namespace detector
{
    extern std::string out_path ;

    struct DetectParam{
        float confidence;
        float nms_threshold;
        int bf_color = 114;
        int class_num = 80;
    };
    class YOLO11 : public rknn::Model{
    public:
        // Standard constructor - creates new rknn context
        YOLO11(std::string model_path, logger::Level level, DetectParam detect_param);
        // Constructor with context sharing - reuses weights from existing context
        YOLO11(std::string model_path, logger::Level level, rknn_context* ctx_in, DetectParam detect_param);

        ~YOLO11();

        // infer method for thread pool (returns object_detect_result_list directly)
        object_detect_result_list infer(cv::Mat img);

        virtual bool preprocess() override;
        virtual bool postprocess() override;
        int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
            int8_t *score_tensor, int32_t score_zp, float score_scale,
            int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
            int grid_h, int grid_w, int stride, int dfl_len,
            std::vector<float> &boxes, 
            std::vector<float> &objProbs, 
            std::vector<int> &classId, 
            float threshold);

        int process_u8(uint8_t *box_tensor, int32_t box_zp, float box_scale,
            uint8_t *score_tensor, int32_t score_zp, float score_scale,
            uint8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
            int grid_h, int grid_w, int stride, int dfl_len,
            std::vector<float> &boxes,
            std::vector<float> &objProbs,
            std::vector<int> &classId,
            float threshold);

        int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor, 
            int grid_h, int grid_w, int stride, int dfl_len,
            std::vector<float> &boxes, 
            std::vector<float> &objProbs, 
            std::vector<int> &classId, 
            float threshold);
        
            int init_post_process();
            void draw(cv::Mat img) override;
        
    private:
        DetectParam m_detectParam;
        image_rect_t m_pads;
        float m_scale;
        cv::Mat m_resized_img;

        std::unique_ptr<object_detect_result_list> m_odReseultsPtr; 
        std::vector<float> m_filterBoxes;
        std::vector<float> m_objProbs;
        std::vector<int> m_classId;
 

        
    /* data */
    };
    
    
    
}; // namespace detector
