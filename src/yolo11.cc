#include "yolo11.hpp"
#include "utils.hpp"

#include <set>

std::string detector::out_path = "./out.jpg";

detector::YOLO11::YOLO11(std::string model_path, logger::Level level,  DetectParam detect_param):rknn::Model(model_path, level) {
    m_detectParam = detect_param;
    init_post_process();
    m_odReseultsPtr = std::make_unique<object_detect_result_list>();

}

detector::YOLO11::YOLO11(std::string model_path, logger::Level level, rknn_context* ctx_in, DetectParam detect_param)
    :rknn::Model(model_path, level, ctx_in) {
    m_detectParam = detect_param;
    init_post_process();
    m_odReseultsPtr = std::make_unique<object_detect_result_list>();
}

detector::YOLO11::~YOLO11() {}

bool detector::YOLO11::preprocess() {
  // 将原始图像处理成模型所需的大小
  memset(&m_pads, 0, sizeof(image_rect_t));
  cv::Size target_size(m_params->image_attrs.model_height,
                       m_params->image_attrs.model_width);
  m_resized_img = cv::Mat(target_size.height, target_size.width, CV_8UC3);

  // compute scale
  float scale_h = (float)target_size.height / m_img.rows;
  float scale_w = (float)target_size.width / m_img.cols;
  m_scale = std::min(scale_h, scale_w);

  letterbox(m_img, m_resized_img, m_pads, m_scale, target_size,
            cv::Scalar(128, 128, 128));
  m_rknnInputPtr[0].index = 0;
  m_rknnInputPtr[0].type = RKNN_TENSOR_UINT8;
  m_rknnInputPtr[0].size = m_params->image_attrs.model_height *
                           m_params->image_attrs.model_width *
                           m_params->image_attrs.model_channels;
  m_rknnInputPtr[0].fmt = RKNN_TENSOR_NHWC;
  m_rknnInputPtr[0].pass_through = 0;
  m_rknnInputPtr[0].buf = m_resized_img.data;

  return true;
}

bool detector::YOLO11::postprocess() {
    m_filterBoxes.clear();
    m_objProbs.clear();
    m_classId.clear();

    rknn_output *_outputs = (rknn_output *)m_rknnOutputPtr.get();

    int validCount = 0;
    int grid_h = 0;
    int grid_w = 0;
    int stride = 0;
    int model_in_h = m_params->image_attrs.model_height;
    int model_in_w = m_params->image_attrs.model_width;

    memset(m_odReseultsPtr.get(), 0, sizeof(object_detect_result_list));

    int dfl_len = m_outputAttrs[0].dims[1] / 4;
    int output_per_branch = m_ioNum.n_output / 3;

    for(int i = 0; i < 3; i++){
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if(output_per_branch == 3){
            score_sum = _outputs[i*output_per_branch + 2].buf;
            score_sum_zp = m_outputAttrs[i*output_per_branch + 2].zp;
            score_sum_scale = m_outputAttrs[i*output_per_branch +2].scale;
        }
        int box_idx = i*output_per_branch;
        int score_idx = i*output_per_branch + 1;

        grid_h = m_outputAttrs[box_idx].dims[2];
        grid_w = m_outputAttrs[box_idx].dims[3];

        stride = model_in_h / grid_h;

        if(m_params->is_quant){
            validCount += process_i8((int8_t *)_outputs[box_idx].buf, m_outputAttrs[box_idx].zp, m_outputAttrs[box_idx].scale,
                                     (int8_t *)_outputs[score_idx].buf, m_outputAttrs[score_idx].zp, m_outputAttrs[score_idx].scale,
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len, 
                                     m_filterBoxes, m_objProbs, m_classId, m_detectParam.confidence);
        }else{
            validCount += process_fp32((float *)_outputs[box_idx].buf, (float *)_outputs[score_idx].buf, (float *)score_sum,
                                       grid_h, grid_w, stride, dfl_len, 
                                       m_filterBoxes, m_objProbs, m_classId, m_detectParam.confidence);

        }
    }

    if(validCount <= 0){
        return false;
    }

    std::vector<int> indexArray;
    for(int i = 0; i < validCount; ++i){
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(m_objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(m_classId), std::end(m_classId));

    for(auto c : class_set){
        nms(validCount, m_filterBoxes, m_classId, indexArray, c, m_detectParam.nms_threshold);
    }
    int last_count = 0;
    m_odReseultsPtr->count = 0;

    for(int i = 0; i < validCount; i++){
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = indexArray[i];

        float x1 = m_filterBoxes[n * 4 + 0] - m_pads.left;
        float y1 = m_filterBoxes[n * 4 + 1] - m_pads.top;
        float x2 = x1 + m_filterBoxes[n * 4 + 2];
        float y2 = y1 + m_filterBoxes[n * 4 + 3];
        int id = m_classId[n];
        float obj_conf = m_objProbs[i];

        m_odReseultsPtr->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / m_scale);
        m_odReseultsPtr->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / m_scale);
        m_odReseultsPtr->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / m_scale);
        m_odReseultsPtr->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / m_scale);
        m_odReseultsPtr->results[last_count].prop = obj_conf;
        m_odReseultsPtr->results[last_count].cls_id = id;
        last_count++;
        
    }
    m_odReseultsPtr->count = last_count;
     
    m_result = *m_odReseultsPtr;

    
    return true; }

int detector::YOLO11::process_i8(
    int8_t* box_tensor, int32_t box_zp, float box_scale, int8_t* score_tensor,
    int32_t score_zp, float score_scale, int8_t* score_sum_tensor,
    int32_t score_sum_zp, float score_sum_scale, int grid_h, int grid_w,
    int stride, int dfl_len, std::vector<float>& boxes,
    std::vector<float>& objProbs, std::vector<int>& classId, float threshold) {
        int validCount = 0;
        int grid_len = grid_h * grid_w;
        int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
        int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);
    
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int offset = i* grid_w + j;
                int max_class_id = -1;
    
                // 通过 score sum 起到快速过滤的作用
                if (score_sum_tensor != nullptr){
                    if (score_sum_tensor[offset] < score_sum_thres_i8){
                        continue;
                    }
                }
    
                int8_t max_score = -score_zp;
                for (int c= 0; c< m_detectParam.class_num; c++){
                    if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                    {
                        max_score = score_tensor[offset];
                        max_class_id = c;
                    }
                    offset += grid_len;
                }
    
                // compute box
                if (max_score> score_thres_i8){
                    offset = i* grid_w + j;
                    float box[4];
                    float before_dfl[dfl_len*4];
                    for (int k=0; k< dfl_len*4; k++){
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);
    
                    float x1,y1,x2,y2,w,h;
                    x1 = (-box[0] + j + 0.5)*stride;
                    y1 = (-box[1] + i + 0.5)*stride;
                    x2 = (box[2] + j + 0.5)*stride;
                    y2 = (box[3] + i + 0.5)*stride;
                    w = x2 - x1;
                    h = y2 - y1;
                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);
    
                    objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                    classId.push_back(max_class_id);
                    validCount ++;
                }
            }
        }
        return validCount;
    }

int detector::YOLO11::process_u8(
    uint8_t* box_tensor, int32_t box_zp, float box_scale, uint8_t* score_tensor,
    int32_t score_zp, float score_scale, uint8_t* score_sum_tensor,
    int32_t score_sum_zp, float score_sum_scale, int grid_h, int grid_w,
    int stride, int dfl_len, std::vector<float>& boxes,
    std::vector<float>& objProbs, std::vector<int>& classId, float threshold) {
        int validCount = 0;
        int grid_len = grid_h * grid_w;
        uint8_t score_thres_u8 = qnt_f32_to_affine_u8(threshold, score_zp, score_scale);
        uint8_t score_sum_thres_u8 = qnt_f32_to_affine_u8(threshold, score_sum_zp, score_sum_scale);
    
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int offset = i * grid_w + j;
                int max_class_id = -1;
    
                // Use score sum to quickly filter
                if (score_sum_tensor != nullptr)
                {
                    if (score_sum_tensor[offset] < score_sum_thres_u8)
                    {
                        continue;
                    }
                }
    
                uint8_t max_score = -score_zp;
                for (int c = 0; c < m_detectParam.class_num; c++)
                {
                    if ((score_tensor[offset] > score_thres_u8) && (score_tensor[offset] > max_score))
                    {
                        max_score = score_tensor[offset];
                        max_class_id = c;
                    }
                    offset += grid_len;
                }
    
                // compute box
                if (max_score > score_thres_u8)
                {
                    offset = i * grid_w + j;
                    float box[4];
                    float before_dfl[dfl_len * 4];
                    for (int k = 0; k < dfl_len * 4; k++)
                    {
                        before_dfl[k] = deqnt_affine_u8_to_f32(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);
    
                    float x1, y1, x2, y2, w, h;
                    x1 = (-box[0] + j + 0.5) * stride;
                    y1 = (-box[1] + i + 0.5) * stride;
                    x2 = (box[2] + j + 0.5) * stride;
                    y2 = (box[3] + i + 0.5) * stride;
                    w = x2 - x1;
                    h = y2 - y1;
                    boxes.push_back(x1);
                    boxes.push_back(y1);
                    boxes.push_back(w);
                    boxes.push_back(h);
    
                    objProbs.push_back(deqnt_affine_u8_to_f32(max_score, score_zp, score_scale));
                    classId.push_back(max_class_id);
                    validCount++;
                }
            }
        }
        return validCount;
}

int detector::YOLO11::process_fp32(float* box_tensor, float* score_tensor,
                                   float* score_sum_tensor, int grid_h,
                                   int grid_w, int stride, int dfl_len,
                                   std::vector<float>& boxes,
                                   std::vector<float>& objProbs,
                                   std::vector<int>& classId, float threshold) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < threshold){
                    continue;
                }
            }

            float max_score = 0;
            for (int c= 0; c< m_detectParam.class_num; c++){
                if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> threshold){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

int detector::YOLO11::init_post_process() { 
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0)
    {
        LOGE("Load %s failed!\n", LABEL_NALE_TXT_PATH);
        return -1;
    }
    return 0;
}

void detector::YOLO11::draw(cv::Mat img) {
    char text[256];
    for (int i = 0; i < m_odReseultsPtr->count; i++)
    {
        object_detect_result *det_result = &(m_odReseultsPtr->results[i]);
        LOGV("%s  @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        sprintf(text, "%s %.1f%% #", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(256, 0, 0, 256), 1);

        cv::putText(img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }
    LOGD("save detect result to %s\n", out_path.c_str());
    cv::imwrite(out_path, img);

}
