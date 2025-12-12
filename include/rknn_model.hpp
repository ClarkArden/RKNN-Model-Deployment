#pragma once
#include <memory>
#include <variant>
#include <string>
#include <mutex>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "rknn_api.h"
#include "logger.hpp"
#include "type.hpp"

namespace rknn{

    // RK3588 has 3 NPU cores
    constexpr int RK3588_NPU_CORE_NUM = 3;

    // Get NPU core number for round-robin assignment
    inline int get_core_num() {
        static int core_num = 0;
        static std::mutex mtx;

        std::lock_guard<std::mutex> lock(mtx);
        int temp = core_num % RK3588_NPU_CORE_NUM;
        core_num++;
        return temp;
    }

    enum task_type{
        DETECTION,
        CLASSFICATION,
        SEGMENTATION
    };

    //每一个模型都有一个image_info
    struct image_info{
        int model_height;
        int model_width;
        int model_channels;

        image_info(int height, int width, int channels) : model_height(height), model_width(width), model_channels(channels) {}
    };

    //对Params设定一些默认参数，这些Param参数是model所需要的
    struct  Params{
        image_info image_attrs = {224, 224, 3};
        task_type task = DETECTION;
        bool is_quant;
    };

    using ModelResult = std::variant<object_detect_result_list>;

    class Model
    {

    public:
        // Standard constructor - creates new rknn context
        Model(std::string model_path, logger::Level level);
        // Constructor with context sharing - reuses weights from existing context
        Model(std::string model_path, logger::Level level, rknn_context* ctx_in);
        virtual ~Model();
        int init_model(rknn_context* ctx_in = nullptr);
        ModelResult inference(cv::Mat img);
        virtual void draw(cv::Mat img) = 0;

        // Get pointer to rknn context for sharing with other instances
        rknn_context* get_context() { return &m_rknnCtx; }

    protected:
         virtual bool preprocess() = 0;
         virtual bool postprocess() = 0;

    private:
        void dump_tensor_attr(rknn_tensor_attr *attr);

    protected:
        std::unique_ptr<Params> m_params;

        std::string m_rknnPath;

        rknn_context m_rknnCtx;
        rknn_input_output_num m_ioNum;
        rknn_tensor_attr* m_inputAttrs;
        rknn_tensor_attr* m_outputAttrs;

        std::shared_ptr<logger::Logger>         m_logger;
        std::unique_ptr<rknn_input[]>           m_rknnInputPtr;
        std::unique_ptr<rknn_output[]>          m_rknnOutputPtr;

        //source image
        cv::Mat     m_img;

        ModelResult m_result;

        // Mutex for thread-safe inference
        std::mutex m_inferenceMtx;



    /* data */

    };
  

}