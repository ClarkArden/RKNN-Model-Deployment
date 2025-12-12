#include "yolov5.hpp"
#include "utils.hpp"

#include <set>

// YOLOv5 anchors for 3 output layers
static int anchor0[6] = {10, 13, 16, 30, 33, 23};      // stride 8
static int anchor1[6] = {30, 61, 62, 45, 59, 119};     // stride 16
static int anchor2[6] = {116, 90, 156, 198, 373, 326}; // stride 32

detector::YOLO5::YOLO5(std::string model_path, logger::Level level, DetectParam detect_param)
    : rknn::Model(model_path, level) {
    m_detectParam = detect_param;
    init_post_process();
    m_odReseultsPtr = std::make_unique<object_detect_result_list>();
}

detector::YOLO5::YOLO5(std::string model_path, logger::Level level, rknn_context* ctx_in, DetectParam detect_param)
    : rknn::Model(model_path, level, ctx_in) {
    m_detectParam = detect_param;
    init_post_process();
    m_odReseultsPtr = std::make_unique<object_detect_result_list>();
}

detector::YOLO5::~YOLO5() {}

object_detect_result_list detector::YOLO5::infer(cv::Mat img) {
    auto result = inference(img);
    return std::get<object_detect_result_list>(result);
}

bool detector::YOLO5::preprocess() {
    // Preprocess image to model input size with letterbox
    memset(&m_pads, 0, sizeof(image_rect_t));
    cv::Size target_size(m_params->image_attrs.model_width,
                         m_params->image_attrs.model_height);
    m_resized_img = cv::Mat(target_size.height, target_size.width, CV_8UC3);

    // Compute scale factor
    float scale_h = (float)target_size.height / m_img.rows;
    float scale_w = (float)target_size.width / m_img.cols;
    m_scale = std::min(scale_h, scale_w);

    letterbox(m_img, m_resized_img, m_pads, m_scale, target_size,
              cv::Scalar(128, 128, 128));

    // Setup RKNN input
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

bool detector::YOLO5::postprocess() {
    m_filterBoxes.clear();
    m_objProbs.clear();
    m_classId.clear();

    rknn_output *_outputs = (rknn_output *)m_rknnOutputPtr.get();

    int validCount = 0;
    int model_in_h = m_params->image_attrs.model_height;
    int model_in_w = m_params->image_attrs.model_width;

    memset(m_odReseultsPtr.get(), 0, sizeof(object_detect_result_list));

    // YOLOv5 has 3 output branches with strides 8, 16, 32
    int *anchors[3] = {anchor0, anchor1, anchor2};
    int strides[3] = {8, 16, 32};

    for (int i = 0; i < 3; i++) {
        int grid_h = m_outputAttrs[i].dims[2];
        int grid_w = m_outputAttrs[i].dims[3];
        int stride = strides[i];

        if (m_params->is_quant) {
            validCount += process_i8((int8_t *)_outputs[i].buf,
                                     anchors[i], grid_h, grid_w, stride,
                                     m_outputAttrs[i].zp, m_outputAttrs[i].scale,
                                     m_filterBoxes, m_objProbs, m_classId,
                                     m_detectParam.confidence);
        } else {
            validCount += process_fp32((float *)_outputs[i].buf,
                                       anchors[i], grid_h, grid_w, stride,
                                       m_filterBoxes, m_objProbs, m_classId,
                                       m_detectParam.confidence);
        }
    }

    if (validCount <= 0) {
        return false;
    }

    // Sort by confidence (descending)
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(m_objProbs, 0, validCount - 1, indexArray);

    // Apply NMS for each class
    std::set<int> class_set(std::begin(m_classId), std::end(m_classId));
    for (auto c : class_set) {
        nms(validCount, m_filterBoxes, m_classId, indexArray, c, m_detectParam.nms_threshold);
    }

    // Collect final results
    int last_count = 0;
    m_odReseultsPtr->count = 0;

    for (int i = 0; i < validCount; i++) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
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

    return true;
}

int detector::YOLO5::process_i8(int8_t *input, int *anchor, int grid_h, int grid_w,
                                int stride, int32_t zp, float scale,
                                std::vector<float> &boxes,
                                std::vector<float> &objProbs,
                                std::vector<int> &classId,
                                float threshold) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int anchor_num = 3;
    int prop_box_size = 5 + m_detectParam.class_num;  // 4 bbox + 1 obj_conf + class_num

    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);

    for (int a = 0; a < anchor_num; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t *in_ptr = input + a * grid_len * prop_box_size + i * grid_w + j;

                // Get objectness confidence
                int8_t box_confidence = in_ptr[4 * grid_len];
                if (box_confidence < thres_i8) {
                    continue;
                }

                // Find max class score
                int max_class_id = -1;
                int8_t max_score = -128;
                for (int c = 0; c < m_detectParam.class_num; c++) {
                    int8_t class_score = in_ptr[(5 + c) * grid_len];
                    if (class_score > max_score) {
                        max_score = class_score;
                        max_class_id = c;
                    }
                }

                // Compute final confidence: obj_conf * class_conf
                float obj_conf_f32 = deqnt_affine_to_f32(box_confidence, zp, scale);
                float class_conf_f32 = deqnt_affine_to_f32(max_score, zp, scale);
                float final_conf = obj_conf_f32 * class_conf_f32;

                if (final_conf < threshold) {
                    continue;
                }

                // Decode box coordinates (YOLOv5 anchor-based decoding)
                float box_x = deqnt_affine_to_f32(in_ptr[0], zp, scale) * 2.0 - 0.5;
                float box_y = deqnt_affine_to_f32(in_ptr[grid_len], zp, scale) * 2.0 - 0.5;
                float box_w = deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale) * 2.0;
                float box_h = deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale) * 2.0;

                // Apply grid and anchor transformations
                box_x = (box_x + j) * stride;
                box_y = (box_y + i) * stride;
                box_w = box_w * box_w * anchor[a * 2];
                box_h = box_h * box_h * anchor[a * 2 + 1];

                // Convert from center to corner format (x, y, w, h)
                float x1 = box_x - box_w / 2.0;
                float y1 = box_y - box_h / 2.0;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(box_w);
                boxes.push_back(box_h);

                objProbs.push_back(final_conf);
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

int detector::YOLO5::process_u8(uint8_t *input, int *anchor, int grid_h, int grid_w,
                                int stride, int32_t zp, float scale,
                                std::vector<float> &boxes,
                                std::vector<float> &objProbs,
                                std::vector<int> &classId,
                                float threshold) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int anchor_num = 3;
    int prop_box_size = 5 + m_detectParam.class_num;

    uint8_t thres_u8 = qnt_f32_to_affine_u8(threshold, zp, scale);

    for (int a = 0; a < anchor_num; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                uint8_t *in_ptr = input + a * grid_len * prop_box_size + i * grid_w + j;

                // Get objectness confidence
                uint8_t box_confidence = in_ptr[4 * grid_len];
                if (box_confidence < thres_u8) {
                    continue;
                }

                // Find max class score
                int max_class_id = -1;
                uint8_t max_score = 0;
                for (int c = 0; c < m_detectParam.class_num; c++) {
                    uint8_t class_score = in_ptr[(5 + c) * grid_len];
                    if (class_score > max_score) {
                        max_score = class_score;
                        max_class_id = c;
                    }
                }

                // Compute final confidence
                float obj_conf_f32 = deqnt_affine_u8_to_f32(box_confidence, zp, scale);
                float class_conf_f32 = deqnt_affine_u8_to_f32(max_score, zp, scale);
                float final_conf = obj_conf_f32 * class_conf_f32;

                if (final_conf < threshold) {
                    continue;
                }

                // Decode box coordinates
                float box_x = deqnt_affine_u8_to_f32(in_ptr[0], zp, scale) * 2.0 - 0.5;
                float box_y = deqnt_affine_u8_to_f32(in_ptr[grid_len], zp, scale) * 2.0 - 0.5;
                float box_w = deqnt_affine_u8_to_f32(in_ptr[2 * grid_len], zp, scale) * 2.0;
                float box_h = deqnt_affine_u8_to_f32(in_ptr[3 * grid_len], zp, scale) * 2.0;

                box_x = (box_x + j) * stride;
                box_y = (box_y + i) * stride;
                box_w = box_w * box_w * anchor[a * 2];
                box_h = box_h * box_h * anchor[a * 2 + 1];

                float x1 = box_x - box_w / 2.0;
                float y1 = box_y - box_h / 2.0;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(box_w);
                boxes.push_back(box_h);

                objProbs.push_back(final_conf);
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

int detector::YOLO5::process_fp32(float *input, int *anchor, int grid_h, int grid_w,
                                  int stride,
                                  std::vector<float> &boxes,
                                  std::vector<float> &objProbs,
                                  std::vector<int> &classId,
                                  float threshold) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int anchor_num = 3;
    int prop_box_size = 5 + m_detectParam.class_num;

    for (int a = 0; a < anchor_num; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                float *in_ptr = input + a * grid_len * prop_box_size + i * grid_w + j;

                // Get objectness confidence (already sigmoid applied in model)
                float box_confidence = in_ptr[4 * grid_len];
                if (box_confidence < threshold) {
                    continue;
                }

                // Find max class score
                int max_class_id = -1;
                float max_score = 0;
                for (int c = 0; c < m_detectParam.class_num; c++) {
                    float class_score = in_ptr[(5 + c) * grid_len];
                    if (class_score > max_score) {
                        max_score = class_score;
                        max_class_id = c;
                    }
                }

                // Compute final confidence
                float final_conf = box_confidence * max_score;
                if (final_conf < threshold) {
                    continue;
                }

                // Decode box coordinates
                float box_x = in_ptr[0] * 2.0 - 0.5;
                float box_y = in_ptr[grid_len] * 2.0 - 0.5;
                float box_w = in_ptr[2 * grid_len] * 2.0;
                float box_h = in_ptr[3 * grid_len] * 2.0;

                box_x = (box_x + j) * stride;
                box_y = (box_y + i) * stride;
                box_w = box_w * box_w * anchor[a * 2];
                box_h = box_h * box_h * anchor[a * 2 + 1];

                float x1 = box_x - box_w / 2.0;
                float y1 = box_y - box_h / 2.0;

                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(box_w);
                boxes.push_back(box_h);

                objProbs.push_back(final_conf);
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

int detector::YOLO5::init_post_process() {
    int ret = 0;
    ret = loadLabelName(LABEL_NALE_TXT_PATH_V5, labels);
    if (ret < 0) {
        LOGE("Load %s failed!\n", LABEL_NALE_TXT_PATH_V5);
        return -1;
    }
    return 0;
}

void detector::YOLO5::draw(cv::Mat img) {
    char text[256];
    for (int i = 0; i < m_odReseultsPtr->count; i++) {
        object_detect_result *det_result = &(m_odReseultsPtr->results[i]);
        LOGV("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
             det_result->box.left, det_result->box.top,
             det_result->box.right, det_result->box.bottom,
             det_result->prop);

        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0, 255), 2);
        cv::putText(img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
    }
    LOGD("save detect result to %s\n", out_path.c_str());
    cv::imwrite(out_path, img);
}
