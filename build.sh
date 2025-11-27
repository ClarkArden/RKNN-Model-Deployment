#!/bin/bash

# RK3588交叉编译脚本
# 设置交叉编译环境并编译项目

set -e  # 遇到错误立即退出

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== RK3588 Cross Compilation Script ===${NC}"

# 加载RK3588交叉编译环境
ENV_SETUP="/home/clark/rk3588/rk3588-buildroot-2021.11-sysroot-v1.0/buildroot/output/rockchip_rk3588/host/environment-setup"

if [ ! -f "$ENV_SETUP" ]; then
    echo -e "${RED}Error: Environment setup file not found at $ENV_SETUP${NC}"
    exit 1
fi

echo -e "${YELLOW}Loading cross-compilation environment...${NC}"
source "$ENV_SETUP"

# 显示编译器信息
echo -e "${YELLOW}Using compiler:${NC}"
echo "CC  = $CC"
echo "CXX = $CXX"
$CC --version | head -n 1
echo ""

# 创建并进入build目录
BUILD_DIR="build"
if [ "$1" == "clean" ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 运行CMake配置
echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake -DCMAKE_C_COMPILER="$CC" \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
      ..

# 编译
echo -e "${YELLOW}Building project...${NC}"
make -j$(nproc)

# 检查编译结果
if [ -f "rknn_model" ]; then
    echo -e "${GREEN}=== Build Successful ===${NC}"
    echo -e "${GREEN}Executable: build/rknn_model${NC}"
    file rknn_model
else
    echo -e "${RED}=== Build Failed ===${NC}"
    exit 1
fi
