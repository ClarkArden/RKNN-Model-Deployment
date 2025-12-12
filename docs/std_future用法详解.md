# std::future 在线程池中的用法详解

## 1. 什么是 std::future

`std::future` 是 C++11 引入的异步编程工具，用于**获取异步操作的结果**。它代表一个"未来"的值——这个值现在可能还没准备好，但将来某个时刻会准备好。

### 核心概念

```
┌─────────────────┐                      ┌─────────────────┐
│   主线程        │                      │   工作线程       │
│                 │    std::future       │                 │
│   提交任务 ─────┼──────────────────────┼──→ 执行任务     │
│                 │                      │       │         │
│   继续其他工作  │                      │       ▼         │
│       │         │                      │   计算结果      │
│       ▼         │                      │       │         │
│   需要结果时    │                      │       ▼         │
│   future.get() ←┼──────────────────────┼── 返回结果      │
│       │         │                      │                 │
│       ▼         │                      │                 │
│   使用结果      │                      │                 │
└─────────────────┘                      └─────────────────┘
```

## 2. 基本用法示例

### 2.1 配合 std::async 使用

```cpp
#include <future>
#include <iostream>

// 耗时的计算函数
int heavy_computation(int x) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return x * x;
}

int main() {
    // 异步启动任务，立即返回 future
    std::future<int> fut = std::async(std::launch::async, heavy_computation, 10);

    std::cout << "任务已提交，继续做其他事情..." << std::endl;

    // 做其他工作...

    // 需要结果时调用 get()，会阻塞直到结果准备好
    int result = fut.get();
    std::cout << "结果: " << result << std::endl;  // 输出: 100

    return 0;
}
```

### 2.2 配合 std::packaged_task 使用

```cpp
#include <future>
#include <thread>
#include <iostream>

int main() {
    // 将函数包装成 packaged_task
    std::packaged_task<int(int)> task([](int x) {
        return x * 2;
    });

    // 获取与 task 关联的 future
    std::future<int> fut = task.get_future();

    // 在新线程中执行 task
    std::thread t(std::move(task), 21);

    // 获取结果
    int result = fut.get();  // 阻塞等待
    std::cout << "结果: " << result << std::endl;  // 输出: 42

    t.join();
    return 0;
}
```

## 3. 线程池中的 future 实现

### 3.1 ThreadPool::submit 函数分析

```cpp
// ThreadPool.hpp 中的 submit 函数
template <typename Func, typename... Ts>
auto submit(Func &&func, Ts &&...params)
    -> std::future<typename std::result_of<Func(Ts...)>::type>
{
    // 1. 绑定函数和参数
    auto execute = std::bind(std::forward<Func>(func), std::forward<Ts>(params)...);

    // 2. 推导返回类型
    using ReturnType = typename std::result_of<Func(Ts...)>::type;
    using PackagedTask = std::packaged_task<ReturnType()>;

    // 3. 创建 packaged_task（用 shared_ptr 包装以便复制）
    auto task = std::make_shared<PackagedTask>(std::move(execute));

    // 4. 获取 future（这是返回给调用者的）
    auto result = task->get_future();

    // 5. 将任务包装成 void() 类型放入队列
    MutexGuard guard(mutex_);
    tasks_.emplace([task]() { (*task)(); });

    // 6. 通知或创建工作线程
    if (idleThreads_ > 0) {
        cv_.notify_one();
    } else if (currentThreads_ < maxThreads_) {
        Thread t(&ThreadPool::worker, this);
        threads_[t.get_id()] = std::move(t);
        ++currentThreads_;
    }

    // 7. 返回 future
    return result;
}
```

### 3.2 执行流程图解

```
调用 submit(func, args...)
         │
         ▼
┌─────────────────────────────────────┐
│ 1. bind(func, args...) → execute    │  绑定函数和参数
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 2. packaged_task<ReturnType()>      │  包装成可调用对象
│    task(execute)                    │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 3. future = task.get_future()       │  获取关联的 future
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 4. tasks_.push([task]{ (*task)(); })│  放入任务队列
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ 5. return future                    │  返回 future 给调用者
└─────────────────────────────────────┘

         ···后台线程···

┌─────────────────────────────────────┐
│ 工作线程从队列取出任务并执行         │
│ (*task)() → 执行 execute()          │
│          → 结果存入 future          │
└─────────────────────────────────────┘
```

## 4. RknnPool 中的 future 使用

### 4.1 任务提交 (put)

```cpp
template <typename rknnModel, typename inputType, typename outputType>
int RknnPool<rknnModel, inputType, outputType>::put(inputType inputData)
{
    int modelId = getModelId();
    std::lock_guard<std::mutex> lock(m_queueMtx);

    // submit 返回 std::future<outputType>
    // 将其存入队列，稍后通过 get() 获取结果
    m_futures.push(
        m_pool->submit(
            &rknnModel::infer,      // 成员函数指针
            m_models[modelId],       // this 指针 (shared_ptr)
            inputData                // 参数
        )
    );

    return 0;
}
```

### 4.2 结果获取 (get)

```cpp
template <typename rknnModel, typename inputType, typename outputType>
int RknnPool<rknnModel, inputType, outputType>::get(outputType& outputData)
{
    std::unique_lock<std::mutex> lock(m_queueMtx);
    if (m_futures.empty())
        return 1;

    // 从队列中取出 future
    auto fut = std::move(m_futures.front());
    m_futures.pop();

    // 释放锁（重要！避免阻塞其他操作）
    lock.unlock();

    // 调用 get() 等待结果
    // 如果任务未完成，会阻塞直到完成
    outputData = fut.get();

    return 0;
}
```

### 4.3 完整流程示意

```
main线程                    线程池                      模型实例
    │                         │                           │
    │  pool.put(img1)         │                           │
    │ ────────────────────────>                           │
    │  返回（立即）            │  submit → future1        │
    │                         │ ──────────────────────────>
    │  pool.put(img2)         │         infer(img1)       │
    │ ────────────────────────>                           │
    │  返回（立即）            │  submit → future2        │
    │                         │ ──────────────────────────>
    │                         │         infer(img2)       │
    │  pool.get(result)       │                           │
    │ ────────────────────────>                           │
    │  future1.get()          │                           │
    │  [等待中...]            │                           │
    │                         │ <─────────── 完成 ────────│
    │ <─────────── result1    │                           │
    │                         │                           │
    │  pool.get(result)       │                           │
    │ ────────────────────────>                           │
    │  future2.get()          │                           │
    │ <─────────── result2    │                           │
    │                         │                           │
```

## 5. std::future 的关键方法

### 5.1 get() - 获取结果

```cpp
std::future<int> fut = std::async([]{ return 42; });

// get() 会阻塞直到结果准备好
int result = fut.get();

// 注意：get() 只能调用一次！再次调用会抛出异常
// int result2 = fut.get();  // 错误！
```

### 5.2 wait() - 等待完成

```cpp
std::future<int> fut = std::async([]{
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return 42;
});

// wait() 只等待，不获取结果
fut.wait();  // 阻塞直到任务完成

// 之后调用 get() 立即返回
int result = fut.get();
```

### 5.3 wait_for() - 带超时等待

```cpp
std::future<int> fut = std::async([]{
    std::this_thread::sleep_for(std::chrono::seconds(10));
    return 42;
});

// 等待最多 1 秒
auto status = fut.wait_for(std::chrono::seconds(1));

if (status == std::future_status::ready) {
    std::cout << "结果已就绪: " << fut.get() << std::endl;
} else if (status == std::future_status::timeout) {
    std::cout << "超时，任务仍在执行" << std::endl;
} else if (status == std::future_status::deferred) {
    std::cout << "任务延迟执行（惰性求值）" << std::endl;
}
```

### 5.4 valid() - 检查有效性

```cpp
std::future<int> fut = std::async([]{ return 42; });

std::cout << fut.valid() << std::endl;  // 输出: 1 (true)

int result = fut.get();

std::cout << fut.valid() << std::endl;  // 输出: 0 (false)
// get() 后 future 变为无效
```

## 6. std::future vs std::shared_future

### 6.1 区别

| 特性 | std::future | std::shared_future |
|------|------------|-------------------|
| get() 调用次数 | 只能调用一次 | 可以调用多次 |
| 可复制性 | 只能移动 | 可以复制 |
| 多线程共享 | 不支持 | 支持 |

### 6.2 shared_future 示例

```cpp
std::promise<int> prom;
std::shared_future<int> sfut = prom.get_future().share();

// 多个线程可以共享同一个 shared_future
std::thread t1([sfut]{
    std::cout << "线程1: " << sfut.get() << std::endl;
});
std::thread t2([sfut]{
    std::cout << "线程2: " << sfut.get() << std::endl;
});

prom.set_value(42);

t1.join();
t2.join();
```

## 7. 异常处理

```cpp
std::future<int> fut = std::async([]{
    throw std::runtime_error("计算出错！");
    return 42;
});

try {
    int result = fut.get();  // 异常会在这里重新抛出
} catch (const std::exception& e) {
    std::cout << "捕获异常: " << e.what() << std::endl;
}
```

## 8. 实际应用：生产者-消费者模式

```cpp
// 本项目中的视频流处理示例
void test_thread_pool_video(const std::string& img_path) {
    rknn::RknnPool<detector::YOLO11, cv::Mat, object_detect_result_list> pool(...);

    std::atomic<bool> producer_done{false};

    // 生产者线程 - 提交任务
    std::thread producer([&]() {
        for (int i = 0; i < 100; i++) {
            pool.put(img);  // 提交任务，返回 future 存入队列
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        producer_done = true;
    });

    // 消费者线程 - 获取结果
    std::thread consumer([&]() {
        object_detect_result_list result;
        while (!producer_done || pool.getPendingCount() > 0) {
            if (pool.get(result) == 0) {  // 内部调用 future.get()
                // 处理结果...
            }
        }
    });

    producer.join();
    consumer.join();
}
```

## 9. 总结

### future 的核心作用

1. **异步获取结果**：提交任务后可以继续其他工作，需要时再获取结果
2. **线程间通信**：安全地将计算结果从工作线程传递到主线程
3. **异常传播**：工作线程中的异常可以传递到调用 get() 的线程

### 线程池中的 future 流程

```
submit() ──→ packaged_task ──→ future ──→ 任务队列
                                  │
                                  ▼
                              返回给调用者
                                  │
                                  ▼
                     调用 get() 时阻塞等待结果
```

### 最佳实践

1. **get() 只能调用一次**：调用后 future 变为无效
2. **使用 std::move**：future 不可复制，只能移动
3. **注意阻塞**：get() 会阻塞，不要在持有锁时调用
4. **检查 valid()**：调用 get() 前确保 future 有效
