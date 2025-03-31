input=$1
model_prefix=${input%.*}

echo python -m modelopt.onnx.quantization \
    --onnx_path=$1 \
    --quantize_mode=\<fp8\|int8\|int4\>\
    --calibration_data=calib.npy \
    --calibration_method=\<max\|entropy\|awq_clip\|rtn_dq\> \
#    --keep_intermediate_files \
    --output_path=${model_prefix}_$2.onnx

python -m modelopt.onnx.quantization \
    --onnx_path=$1 \
    --quantize_mode=$2 \
    --calibration_data=calib.npy \
    --calibration_method=$3 \
#    --keep_intermediate_files \
    --output_path=${model_prefix}_$2.onnx

echo trtexec --onnx=${model_prefix}_$2.onnx --saveEngine=${model_prefix}_$2.trt --$2 --dumpProfile

trtexec --onnx=${model_prefix}_$2.onnx --saveEngine=${model_prefix}_$2.trt --$2 --dumpProfile


