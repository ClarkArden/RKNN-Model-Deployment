#include <iostream>
#include "logger.hpp"

int main(int, char**){
    std::cout << "Hello, from rknn_mode!\n";

    // 测试logger的各个级别
    std::cout << "\n=== Testing Logger ===" << std::endl;

    // 创建logger实例，设置为DEBUG级别以显示所有日志
    auto logger = logger::create_logger(logger::Level::DEBUG);

    // 测试各个日志级别
    LOGD("This is a DEBUG message");
    LOGV("This is a VERBOSE message");
    LOG("This is an INFO message");
    LOGW("This is a WARNING message");

    // 测试带参数的日志
    int value = 42;
    const char* name = "test";
    LOG("Testing with parameters: value=%d, name=%s", value, name);

    std::cout << "\n=== Logger Test Completed ===" << std::endl;

    // 注意：LOGE和LOGF会导致程序退出，所以放在最后注释掉
    // LOGE("This is an ERROR message - will exit");
    // LOGF("This is a FATAL message - will exit");

    return 0;
}
