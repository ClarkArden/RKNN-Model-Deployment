#include "logger.hpp"
#include <cstdlib>

using namespace std;

namespace logger {

Level Logger::m_level = Level::INFO;

Logger::Logger(Level level) {
    m_level = level;
}

void Logger::__log_info(Level level, const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);
    int n = 0;

    switch (level) {
        case Level::DEBUG: n += snprintf(msg + n, sizeof(msg) - n, DGREEN "[debug]" CLEAR); break;
        case Level::VERB:  n += snprintf(msg + n, sizeof(msg) - n, PURPLE "[verb]" CLEAR); break;
        case Level::INFO:  n += snprintf(msg + n, sizeof(msg) - n, YELLOW "[info]" CLEAR); break;
        case Level::WARN:  n += snprintf(msg + n, sizeof(msg) - n, BLUE "[warn]" CLEAR); break;
        case Level::ERROR: n += snprintf(msg + n, sizeof(msg) - n, RED "[error]" CLEAR); break;
        default:           n += snprintf(msg + n, sizeof(msg) - n, RED "[fatal]" CLEAR); break;
    }

    n += vsnprintf(msg + n, sizeof(msg) - n, format, args);

    va_end(args);

    if (level <= m_level)
        fprintf(stdout, "%s\n", msg);

    if (level <= Level::ERROR) {
        fflush(stdout);
        exit(0);
    }
}

shared_ptr<Logger> create_logger(Level level) {
    return make_shared<Logger>(level);
}

} // namespace logger
