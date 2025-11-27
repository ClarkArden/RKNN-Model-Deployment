#include "utils.hpp"
#include "logger.hpp"

// 定义 labels 全局变量
char *labels[OBJ_CLASS_NUM];

unsigned char* load_model(const char* filename, int* model_size) {
    FILE* fp = fopen(filename, "rb");
    if(fp == NULL){
        LOGE("fopen %s fail!", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char* model = (unsigned char*)malloc(model_len + 1);
    
    if(model == NULL){
        LOGE("malloc fail! size=%d", model_len);
        fclose(fp);
        return NULL;
    }

    model[model_len] = 0;
    fseek(fp, 0, SEEK_SET); 
    if(model_len != fread(model, 1, model_len, fp)){
        LOGE("fread %s fail!", filename);
        free(model);
        fclose(fp);
        return NULL;
    }
    *model_size = model_len;
    fclose(fp);
    return model;
}

void letterbox(const cv::Mat& image,  cv::Mat& padded_img,
               image_rect_t& pads, const float scale,
               const cv::Size& traget_size, const cv::Scalar& pad_color) {
    
                //adjust image size;
                cv::Mat resized_img;
                cv::resize(image, resized_img, cv::Size(), scale, scale);

                //compute pad size
                int pad_width = traget_size.width - resized_img.cols;
                int pad_height = traget_size.height - resized_img.rows;

                pads.left = pad_width / 2;
                pads.right = pad_width - pads.left;
                pads.top = pad_height / 2;
                pads.bottom = pad_height - pads.top;
                
                //add pad surrunding image
                cv::copyMakeBorder(resized_img, padded_img, pads.top, pads.bottom, pads.left, pads.right, cv::BORDER_CONSTANT, pad_color);

    
               }

int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

void compute_dfl(float* tensor, int dfl_len, float* box) {
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

int quick_sort_indice_inverse(std::vector<float>& input, int left, int right,
                              std::vector<int>& indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
                       float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

int nms(int validCount, std::vector<float>& outputLocations,
        std::vector<int> classIds, std::vector<int>& order, int filterId,
        float threshold) {
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

char* readLine(FILE* fp, char* buffer, int* len) {
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL)
        {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp)))
    {
        free(buffer);
        return NULL;
    }
    return buffer;


}

int readLines(const char* fileName, char* lines[], int max_line) {
    FILE *file = fopen(fileName, "r");
    char *s = NULL;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

int loadLabelName(const char* locationFilename, char* label[]) { 
    LOGD("load lable %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);
    return 0;
}

const char* coco_cls_to_name(int cls_id) {
    if (cls_id >= OBJ_CLASS_NUM)
    {
        return "null";
    }

    if (labels[cls_id])
    {
        return labels[cls_id];
    }

    return "null";
 }
