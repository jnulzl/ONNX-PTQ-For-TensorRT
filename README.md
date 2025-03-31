## ONNX PTQ For TensorRT

### Create env

```shell
conda create -n onnx_quant python==3.10
```

### Install modelopt

```shell
pip install nvidia-modelopt\[onnx\] -U --extra-index-url https://pypi.nvidia.com opencv-python
```

### Demo for yolov5

-	Download [`yolov5 onnx`](https://objects.githubusercontent.com/github-production-release-asset-2e65be/264818686/dedf8e44-e7fd-450a-a0e8-c27424f86ce2?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20250331%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250331T064306Z&X-Amz-Expires=300&X-Amz-Signature=5d3e60d88954dc8ac6fd6a8a6b2113f07aa568a3f3b09ab2587e8f94ccff4185&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dyolov5n.onnx&response-content-type=application%2Foctet-stream) model


-	Prepare calibration data

```shell
find coco_samples -name '*jpg' > coco_list.txt
python image_prep.py --mean 0.0 0.0 0.0 --scale 0.0039215 0.0039215 0.0039215 --calibration_data_list_file_path coco_list.txt --shape 640 640 --is_bgr2rgb --is_hwc2chw --is_fixed_scale --output_path coco_samples.npy --fp16
```

- `PTQ` for yolov5 

```shell
./onnx_ptq_int8.sh yolov5n.onnx  coco_samples.npy 'images:1x3x640x640'
```

- Change `onnx` batch size(Optional)

```shell
./change_onnx_batch_size.sh xx.onnx batch_size
```

- Quanted `onnx` to trt engine files(Optional, TensorRT needed)

```shell
trtexec --onnx=xxx_quanted.onnx --saveEngine=xxx_quanted.trt --int8 --dumpProfile
```

### tips

- trt_guided_options(In `xxx/site-packages/modelopt/onnx/quantization/int8.py`)
```
{
	'QuantizeBias': False, 
	'ActivationSymmetric': True, 
	'OpTypesToExcludeOutputQuantization': ['Pow', 'Sigmoid', 'Resize', 'Transpose', 'Mul', 'Conv', 'Constant', 'Add', 'MaxPool', 'Concat', 'Reshape', 'Split'], 
	'AddQDQPairToWeight': True,
	'QDQOpTypePerChannelSupportToAxis': {'Conv': 0, 'ConvTranspose': 1}, 
	'DedicatedQDQPair': True, 
	'ForceQuantizeNoInputCheck': True, 
	'TrtExtraPluginLibraryPaths': None, 
	'ExecutionProviders': ['CPUExecutionProvider']
}
```

ref `quantize_static` int `site-packages/onnxruntime/quantization/quantize.py`
