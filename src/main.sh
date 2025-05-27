#!/bin/bash

for file in ./params/*; do
    # 将文件复制为param.py
    cp $file ./param.py

    echo \"Copied $file to param.py\"

    sleep 10

    until python launcher.py; do
        echo \"launcher.py执行失败, 10秒后重试...\"
        sleep 10
    done

    sleep 10
done
