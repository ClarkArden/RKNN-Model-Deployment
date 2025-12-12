#ifndef RKNNPOOL_H
#define RKNNPOOL_H

#include "ThreadPool.hpp"
#include <vector>
#include <iostream>
#include <mutex>
#include <queue>
#include <memory>

#include "rknn_model.hpp"

namespace rknn {

// rknnModel: 模型类型 (如 detector::YOLO11, detector::YOLO5)
// inputType: 输入类型 (如 cv::Mat)
// outputType: 输出类型 (如 object_detect_result_list)
template <typename rknnModel, typename inputType, typename outputType>
class RknnPool
{
private:
    int m_threadNum;
    std::string m_modelPath;
    logger::Level m_logLevel;

    long long m_id;
    std::mutex m_idMtx, m_queueMtx;
    std::unique_ptr<dpool::ThreadPool> m_pool;
    std::queue<std::future<outputType>> m_futures;
    std::vector<std::shared_ptr<rknnModel>> m_models;

protected:
    int getModelId();

public:
    // modelPath: 模型路径
    // threadNum: 线程数 (建议设置为NPU核心数，RK3588为3)
    // level: 日志级别
    template <typename... Args>
    RknnPool(const std::string& modelPath, int threadNum, logger::Level level, Args&&... args);

    ~RknnPool();

    // 初始化模型池
    // 第一个模型使用rknn_init加载完整权重
    // 后续模型使用rknn_dup_context复用权重，绑定不同核心
    template <typename... Args>
    int init(Args&&... args);

    // 提交推理任务
    int put(inputType inputData);

    // 获取推理结果 (阻塞等待)
    int get(outputType& outputData);

    // 获取队列中待处理的任务数
    size_t getPendingCount();
};

// 构造函数实现
template <typename rknnModel, typename inputType, typename outputType>
template <typename... Args>
RknnPool<rknnModel, inputType, outputType>::RknnPool(
    const std::string& modelPath, int threadNum, logger::Level level, Args&&... args)
    : m_modelPath(modelPath), m_threadNum(threadNum), m_logLevel(level), m_id(0)
{
}

// 初始化实现
template <typename rknnModel, typename inputType, typename outputType>
template <typename... Args>
int RknnPool<rknnModel, inputType, outputType>::init(Args&&... args)
{
    try
    {
        // 创建线程池
        m_pool = std::make_unique<dpool::ThreadPool>(m_threadNum);

        // 创建第一个模型实例 (加载完整权重)
        std::cout << "[RknnPool] Creating primary model instance (loads full weights)..." << std::endl;
        m_models.push_back(std::make_shared<rknnModel>(
            m_modelPath, m_logLevel, std::forward<Args>(args)...));

        // 创建后续模型实例 (使用rknn_dup_context复用权重)
        for (int i = 1; i < m_threadNum; i++)
        {
            std::cout << "[RknnPool] Creating shared model instance " << i
                      << " (sharing weights via rknn_dup_context)..." << std::endl;
            m_models.push_back(std::make_shared<rknnModel>(
                m_modelPath, m_logLevel, m_models[0]->get_context(),
                std::forward<Args>(args)...));
        }

        std::cout << "[RknnPool] Initialized " << m_threadNum << " models successfully" << std::endl;
    }
    catch (const std::bad_alloc& e)
    {
        std::cerr << "[RknnPool] Out of memory: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[RknnPool] Initialization failed: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

// 获取模型ID (轮询分配)
template <typename rknnModel, typename inputType, typename outputType>
int RknnPool<rknnModel, inputType, outputType>::getModelId()
{
    std::lock_guard<std::mutex> lock(m_idMtx);
    int modelId = m_id % m_threadNum;
    m_id++;
    return modelId;
}

// 提交推理任务
template <typename rknnModel, typename inputType, typename outputType>
int RknnPool<rknnModel, inputType, outputType>::put(inputType inputData)
{
    int modelId = getModelId();
    std::lock_guard<std::mutex> lock(m_queueMtx);
    // 提交到线程池，调用模型的infer方法
    m_futures.push(m_pool->submit(&rknnModel::infer, m_models[modelId], inputData));
    return 0;
}

// 获取推理结果
template <typename rknnModel, typename inputType, typename outputType>
int RknnPool<rknnModel, inputType, outputType>::get(outputType& outputData)
{
    std::unique_lock<std::mutex> lock(m_queueMtx);
    if (m_futures.empty())
        return 1; // 队列为空

    auto fut = std::move(m_futures.front());
    m_futures.pop();
    lock.unlock();  // 在等待future时释放锁

    outputData = fut.get();
    return 0;
}

// 获取待处理任务数
template <typename rknnModel, typename inputType, typename outputType>
size_t RknnPool<rknnModel, inputType, outputType>::getPendingCount()
{
    std::lock_guard<std::mutex> lock(m_queueMtx);
    return m_futures.size();
}

// 析构函数
template <typename rknnModel, typename inputType, typename outputType>
RknnPool<rknnModel, inputType, outputType>::~RknnPool()
{
    // 等待所有任务完成
    while (true)
    {
        std::unique_lock<std::mutex> lock(m_queueMtx);
        if (m_futures.empty())
            break;
        auto fut = std::move(m_futures.front());
        m_futures.pop();
        lock.unlock();
        fut.get();  // 等待任务完成
    }
    std::cout << "[RknnPool] Pool destroyed" << std::endl;
}

} // namespace rknn

#endif // RKNNPOOL_H
