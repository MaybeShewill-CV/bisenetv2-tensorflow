#!/usr/bin/env bash
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
# Author: MaybeShewill-CV
# Date: 2020/04/15
#
# convert tensorflow frozen model into onnx

# ------------ 设置常量 ----------------
parameters=2
input_para_nums=$#
para0=$0
para1=$1
para2=$2

# ------------ 帮助函数 ----------------
function usage() {
  echo "usage: $para0 [frozen_pb_file_path] [output_onnx_model_file_path]"
  echo "       frozen_pb_file_path   tensorflow frozen model file path"
  echo "       output_onnx_model_file_path  converted onnx model file save path"
  echo "examples: "
  echo "       $para0 ./checkpoint/bisenet.pb ./checkpoint/bisenet.onnx"
  exit 1
}

# ------------ 主函数 ------------------
function main() {
if [ ${input_para_nums} != ${parameters} ];
then
  usage
else
  python -m tf2onnx.convert --input ${para1} --output ${para2} --inputs input_tensor:0 --outputs final_output:0 --opset 13
fi
}

main
