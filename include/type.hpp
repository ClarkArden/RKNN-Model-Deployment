#pragma once

#include <stdint.h>
#include <vector>

#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80



/**
 * @brief Image pixel format
 * 
 */
typedef enum {
    IMAGE_FORMAT_GRAY8,
    IMAGE_FORMAT_RGB888,
    IMAGE_FORMAT_RGBA8888,
    IMAGE_FORMAT_YUV420SP_NV21,
    IMAGE_FORMAT_YUV420SP_NV12,
} image_format_t;

/**
 * @brief Image buffer
 * 
 */
typedef struct {
    int width;
    int height;
    int width_stride;
    int height_stride;
    image_format_t format;
    unsigned char* virt_addr;
    int size;
    int fd;
} image_buffer_t;

/**
 * @brief Image rectangle
 * 
 */
typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

/**
 * @brief Image obb rectangle
 * 
 */
typedef struct {
    int x;
    int y;
    int w;
    int h;
    float angle;
} image_obb_box_t;



typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;


extern char *labels[OBJ_CLASS_NUM];