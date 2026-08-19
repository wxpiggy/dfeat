# XFeat + LighterGlue TensorRT

该 Demo 的运行时链路完全使用 OpenCV、CUDA 和 TensorRT：

1. CUDA 完成灰度化、归一化和模型尺寸缩放；
2. TensorRT 执行 XFeat；
3. CUDA 完成 softmax、depth-to-space、NMS、Top-K、双三次描述子插值和 L2 归一化；
4. TensorRT 按两幅图各自的实际点数执行 LighterGlue。

LibTorch/PyTorch 只在重新导出 ONNX 时需要，`match_test` 及两个运行时动态库均不链接
`libtorch`、`torch_cpu` 或 `c10`。

## 依赖

- CUDA 11.x+
- TensorRT 8.6+
- OpenCV 4.x
- yaml-cpp
- PyTorch、ONNX、onnxsim（仅 ONNX 导出脚本需要）

## 编译

```bash
cmake -S . -B build \
  -DTENSORRT_ROOT=/path/to/TensorRT \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc
cmake --build build -j
```

如目标 GPU 架构不在默认的 `61;75;86` 中，可增加
`-DCMAKE_CUDA_ARCHITECTURES=<compute capability>`。

## 构建引擎

XFeat 引擎的空间尺寸可以固定；Demo 会在 CUDA 中把原图缩放至引擎输入尺寸，并将输出点
重新映射到各自的原图坐标系。

```bash
trtexec \
  --onnx=weights/xfeat_1_800_800.onnx \
  --saveEngine=weights/xfeat_1_800_800.engine
```

LighterGlue 必须使用动态点数 ONNX 和动态 TensorRT profile。当前导出脚本会始终把
`N0/N1` 标为动态轴；仓库里已有的旧静态 ONNX 需要重新生成：

```bash
python3 scripts/export.py \
  --xfeat_only_lighterglue \
  --height 800 --width 800 --top_k 512 \
  --export_path weights/lightglue_L6_dynamic.onnx

trtexec \
  --onnx=weights/lightglue_L6_dynamic.onnx \
  --minShapes=mkpts0:1x1x2,feats0:1x1x64,mkpts1:1x1x2,feats1:1x1x64 \
  --optShapes=mkpts0:1x256x2,feats0:1x256x64,mkpts1:1x256x2,feats1:1x256x64 \
  --maxShapes=mkpts0:1x512x2,feats0:1x512x64,mkpts1:1x512x2,feats1:1x512x64 \
  --saveEngine=weights/lightglue_L6_dynamic.engine
```

`maxShapes` 的点数应不小于 `config/xfeat_lightglue.yaml` 中的 `max_keypoints`。

## 运行 Demo

```bash
./build/match_test \
  /path/to/image0.png \
  /path/to/image1.png \
  weights/xfeat_1_800_800.engine \
  weights/lightglue_L6_dynamic.engine \
  config/xfeat_lightglue.yaml \
  matching_result.png
```

关键点始终采用 `(x, y)` 顺序。传给 LighterGlue 的 `image0_size`、`image1_size` 分别是两幅
原始图像自己的 `(width, height)`；两幅图不要求同尺寸。每次匹配都会按实际 `N0/N1` 设置
`mkpts0/feats0/mkpts1/feats1` 的动态 binding，不再固定为 512 点，也不会复用上一帧匹配结果。
