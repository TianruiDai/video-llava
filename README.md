# video-llava

当前阶段直接调用 Hugging Face 上的 Video-LLaVA，不重写模型内部。后续会把 Video-LLaVA 拆开，单独看它的核心组件和模型。

## 当前阶段

通过 [`src/models/builder.py`](src/models/builder.py) 加载 `LanguageBind/Video-LLaVA-7B-hf`：

- `VideoLlavaForConditionalGeneration` 负责前向、生成和损失
- `AutoProcessor` 把文本和视频转成张量

权重约 14.7 GB，缓存在 `~/.cache/huggingface/hub`。`builder.py` 在导入 `transformers` 之前设置了 `HF_HUB_DISABLE_XET=1`，大文件走普通 HTTP，避免 Xet/CAS 在文件重组阶段失败。

## 后续方向

接下来不再把 `generate()` 当作黑盒，而是拆开 Video-LLaVA，看各组件和它们之间的数据流：

- 视频编码器 `video_tower`（`E_V`）。特征抽取见 [`src/models/feature_extractor.py`](src/models/feature_extractor.py)
- 图像编码器 `image_tower`（`E_I`）
- 投影层 `multi_modal_projector`（`P`）
- 语言模型 `language_model`（`E_L`）

[`src/models/freezing.py`](src/models/freezing.py) 已按这四块预留了冻结和只解冻投影层的接口。

## 目录

- `scripts/inspect_model.py`：加载模型，打印 `config` 和 `model.model` 的子模块
- `scripts/infer_video.py`：对单个视频做问答，参数为 `--video` 和 `--question`
- `src/data/video_reader.py`：均匀采样 `num_frames` 帧，输出形状为 `(T, H, W, 3)` 的数组
- `src/data/dataset.py`：`VideoQADataset` 读取 JSONL，每行包含 `video`、`question`、`answer`

JSONL 示例：

```json
{"video": "/path/to/video.mp4", "question": "视频里的人在做什么？", "answer": "一个人在走路。"}
```

## 环境与运行

在 `video_llava` 环境（Python 3.11）中安装依赖。PyTorch 使用已安装的 CUDA 版本（当前为 `2.14.1+cu130`），不要换成 CPU 轮子。

脚本通过 `from models` 和 `from data` 导入，需要把 `src` 放进 `PYTHONPATH`：

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/inspect_model.py
PYTHONPATH=src python scripts/infer_video.py --video /path/to/video.mp4 --question "Describe the video."
```

推理时均匀采样 8 帧。提示词格式为：

```text
USER: <video>
{question} ASSISTANT:
```

## 已知限制

[`src/data/__init__.py`](src/data/__init__.py) 使用 `from video_reader import ...` 这种绝对导入。只设置 `PYTHONPATH=src` 时，`from data import sample_video_uniform` 会找不到模块，因此上面的推理命令目前还不能直接跑通。需要改成包内相对导入后再运行 `scripts/infer_video.py`。
