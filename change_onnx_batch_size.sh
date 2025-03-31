input=$1
model_prefix=${input%.*}

python -m onnxruntime.tools.make_dynamic_shape_fixed --input_name "images" --input_shape $2,3,640,640 $1 ${model_prefix}_bs$2.onnx
