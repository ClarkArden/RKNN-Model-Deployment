#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "type.hpp"

unsigned char* load_model(const char* filename, int* model_size);

void letterbox(const cv::Mat &image,  cv::Mat &padded_img, image_rect_t& pads, const float scale, const cv::Size &traget_size, const cv::Scalar &pad_color = cv::Scalar(128, 128, 128));

inline  int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

inline  int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; };

int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);

uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale);

float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);

float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale);

void compute_dfl(float* tensor, int dfl_len, float* box);

int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices);

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1);

int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order, int filterId, float threshold);

char *readLine(FILE *fp, char *buffer, int *len);

int readLines(const char *fileName, char *lines[], int max_line);

int loadLabelName(const char *locationFilename, char *label[]);

const char *coco_cls_to_name(int cls_id);


// void draw()