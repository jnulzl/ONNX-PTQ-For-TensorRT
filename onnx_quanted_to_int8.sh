input=$1
model_prefix=${input%.*}

./change_onnx_batch_size.sh ${model_prefix}.onnx $2

echo trtexec --onnx=${model_prefix}_bs$2.onnx --saveEngine=${model_prefix}_bs$2.trt --int8 --dumpProfile

trtexec --onnx=${model_prefix}_bs$2.onnx --saveEngine=${model_prefix}_bs$2.trt --int8 --dumpProfile

