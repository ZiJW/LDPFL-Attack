#!/bin/bash

# 遍历./params下的所有文件
for file in ./params/*; do
    # 将文件复制为param.py
    cp $file ./param.py

    echo \"Copied $file to param.py\"

    sleep 10

    # 循环执行launcher.py直到成功
    until python launcher.py; do
        echo \"launcher.py执行失败, 10秒后重试...\"
        sleep 10
    done

    # 等待10秒
    sleep 10
done
