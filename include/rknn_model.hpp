#pragma once
#include <memory>
#include <variant>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "rknn_api.h"
#include "logger.hpp"
#include "type.hpp"

namespace rknn{
    
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
        Model(std::string model_path, logger::Level level);
        virtual ~Model();
        int init_model();
        ModelResult inference(cv::Mat img);
        virtual void draw(cv::Mat img) = 0;

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
        

    
    /* data */

    };
  

}