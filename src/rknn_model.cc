#include "rknn_model.hpp"
#include "utils.hpp"

rknn::Model::Model(std::string model_path, logger::Level level) {
    m_rknnPath = model_path;
    m_logger = std::make_shared<logger::Logger>(level);
    m_params = std::make_unique<Params>();
    init_model(nullptr);
}

rknn::Model::Model(std::string model_path, logger::Level level, rknn_context* ctx_in) {
    m_rknnPath = model_path;
    m_logger = std::make_shared<logger::Logger>(level);
    m_params = std::make_unique<Params>();
    init_model(ctx_in);
}

rknn::Model::~Model() {
    if(m_inputAttrs != NULL){
        free(m_inputAttrs);
        m_inputAttrs = NULL;
    }
    if(m_outputAttrs != NULL){
        free(m_outputAttrs);
        m_outputAttrs = NULL;
    }
    if(m_rknnCtx != 0){
        rknn_destroy(m_rknnCtx);
        m_rknnCtx = 0;
    }
}

int rknn::Model::init_model(rknn_context* ctx_in) {
    int ret;
    int model_len = 0;
    unsigned char* model;

    //load RKNN Model
    model  = load_model(m_rknnPath.c_str(), &model_len);
    if(model == NULL){
        LOGE("load model fail!");
        return -1;
    }
    m_rknnCtx = 0;

    // Model parameter reuse: use rknn_dup_context if ctx_in is provided
    if (ctx_in != nullptr) {
        ret = rknn_dup_context(ctx_in, &m_rknnCtx);
        LOG("Using shared context (rknn_dup_context)");
    } else {
        ret = rknn_init(&m_rknnCtx, model, model_len, 0, NULL);
        LOG("Creating new context (rknn_init)");
    }
    free(model);
    if(ret < 0){
        LOGE("rknn_init/rknn_dup_context fail! ret = %d", ret);
        return -1;
    }

    // Set model to bind to specific NPU core (round-robin assignment)
    int core_id = get_core_num();
    rknn_core_mask core_mask;
    switch (core_id) {
        case 0:
            core_mask = RKNN_NPU_CORE_0;
            break;
        case 1:
            core_mask = RKNN_NPU_CORE_1;
            break;
        case 2:
            core_mask = RKNN_NPU_CORE_2;
            break;
        default:
            core_mask = RKNN_NPU_CORE_AUTO;
            break;
    }
    ret = rknn_set_core_mask(m_rknnCtx, core_mask);
    if (ret < 0) {
        LOGE("rknn_set_core_mask fail! ret = %d, core_id = %d", ret, core_id);
        return -1;
    }
    LOG("Model bindied to NPU core %d", core_id);

    //get model input info Output NUmber
    rknn_input_output_num io_num;
    ret = rknn_query(m_rknnCtx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if(ret != RKNN_SUCC){
        LOGF("rknn_querry fail! ret = %d", ret);
        return -1;
    }

    LOG("model input num: %d, output num: %d", io_num.n_input, io_num.n_output);

    //Get Model Input Info
    LOGV("input tensors:");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for(int i = 0; i < io_num.n_input; i++){
        input_attrs[i].index = i;
        ret = rknn_query(m_rknnCtx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if(ret != RKNN_SUCC){
            LOGF("rknn_query fail! ret = %d", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    //Get Model Output Info
    LOGV("Output tensors:");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for(int i = 0; i < io_num.n_output; i++){
        output_attrs[i].index = i;
        ret = rknn_query(m_rknnCtx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if(ret != RKNN_SUCC){
            LOGE("rknn_querry fail! ret=%d", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    } 
    
    if(output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type == RKNN_TENSOR_INT8){
        m_params->is_quant = true;
    }else{
        m_params->is_quant = false;
    }
    
    m_ioNum = io_num;
    m_inputAttrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(m_inputAttrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    m_outputAttrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(m_outputAttrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));
    
    if(input_attrs[0].fmt == RKNN_TENSOR_NCHW){
        LOG("model is NCHW input fmt.");
        m_params->image_attrs.model_channels = input_attrs[0].dims[1]; 
        m_params->image_attrs.model_height = input_attrs[0].dims[2];
        m_params->image_attrs.model_width = input_attrs[0].dims[3];
    }else{
        LOG("model is NHWC input fmt.");
        m_params->image_attrs.model_height = input_attrs[0].dims[1];
        m_params->image_attrs.model_width = input_attrs[0].dims[2];
        m_params->image_attrs.model_channels = input_attrs[0].dims[3]; 

    }
    
    LOG("model input height=%d, width=%d, channel=%d",m_params->image_attrs.model_height, m_params->image_attrs.model_width, m_params->image_attrs.model_channels);

    return 0; 
}

rknn::ModelResult rknn::Model::inference(cv::Mat img) {
    int ret;
    m_img = img.clone();

    m_rknnInputPtr = std::make_unique<rknn_input[]>(m_ioNum.n_input);
    memset(m_rknnInputPtr.get(), 0, m_ioNum.n_input * sizeof(rknn_input));

    // pre process
    preprocess();
    //set  rknn input
    ret  = rknn_inputs_set(m_rknnCtx, m_ioNum.n_input, m_rknnInputPtr.get());
    if(ret < 0){
        LOGE("rknn_input_set fail! ret=%d", ret);
        return rknn::ModelResult();
    }

    //Run
    LOGD("rknn_run!");
    ret  = rknn_run(m_rknnCtx, nullptr);
    if(ret < 0){
        LOGE("rknn_run fail! ret=%d", ret);
        return rknn::ModelResult();
    }

    // Get output
    m_rknnOutputPtr = std::make_unique<rknn_output[]>(m_ioNum.n_output);
    memset(m_rknnOutputPtr.get(), 0, m_ioNum.n_output * sizeof(rknn_output));
    for(int i=0; i < m_ioNum.n_output; i++){
        m_rknnOutputPtr[i].index = i;
        m_rknnOutputPtr[i].want_float = (!m_params->is_quant);
    }
    ret  = rknn_outputs_get(m_rknnCtx, m_ioNum.n_output, m_rknnOutputPtr.get(), NULL);
    if(ret < 0){
        LOGE("rknn_output_get fail! ret =%d",  ret);
        return rknn::ModelResult();
    }

    //post process
    postprocess();
    
    //Remeber to release rknn output
    ret = rknn_outputs_release(m_rknnCtx, m_ioNum.n_output, m_rknnOutputPtr.get());
    
     return m_result; }


void rknn::Model::dump_tensor_attr(rknn_tensor_attr* attr) {
    LOGV("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f",
        attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
        attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
        get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}
