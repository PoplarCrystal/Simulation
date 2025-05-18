#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR="$(dirname "$SCRIPT_DIR")"  # 获取上一级目录

cd "${DIR}/lcm_types/java" || exit 1  # 如果 cd 失败则退出
export CLASSPATH="${DIR}/lcm_types/java/my_types.jar"
pwd
lcm-spy