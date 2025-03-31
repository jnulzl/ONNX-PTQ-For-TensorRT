input=$1
model_prefix=${input%.*}

echo python -m modelopt.onnx.quantization \
    --onnx_path=$1 \
    --quantize_mode=int8 \
    --calibration_data=$2 \
    --calibration_method=entropy \
    --output_path=${model_prefix}_int8.onnx \
    --calibration_shapes=$3

python -m modelopt.onnx.quantization \
    --onnx_path=$1 \
    --quantize_mode=int8 \
    --calibration_data=$2 \
    --calibration_method=entropy \
    --output_path=${model_prefix}_int8.onnx \
    --calibration_shapes=$3

# ./change_onnx_batch_size.sh ${model_prefix}_int8.onnx $2
# 
# echo trtexec --onnx=${model_prefix}_int8_bs$2.onnx --saveEngine=${model_prefix}_int8_bs$2.trt --int8 --dumpProfile
# 
# trtexec --onnx=${model_prefix}_int8_bs$2.onnx --saveEngine=${model_prefix}_int8_bs$2.trt --int8 --dumpProfile

