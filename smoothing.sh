#!/bin/bash

for number in {0..80}
do
    # 書き換え
    echo $((number+1))
    #echo idx_t: $number
    if grep -q "idx_t: ${number}" examples/aneurysm3D/configs/config.yaml; then
        sed -i "s/idx_t: ${number}/idx_t: $((number+1))/" examples/aneurysm3D/configs/config.yaml
    else
        echo "Error: 'idx_t: ${number}' not found in config.yaml. Exiting."
        exit 1
    fi
    # Pythonスクリプト実行
    python examples/aneurysm3D/train.py
    
    # バイナリ実行ファイル実行
    ./a.out
    
    # ファイル名変更
    mv test.vtu "test_${number}.vtu"
done