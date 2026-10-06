import os

# 大文件默认走 Xet/CAS（us.aws.cdn.hf.co）。当前网络能下到接近完成，
# 但重组分片时请求失败。必须在导入 huggingface_hub 之前关闭。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from pyexpat import model
import torch

from transformers import (
    AutoProcessor,
    VideoLlavaForConditionalGeneration,
)

# 该脚本把Hugging Face的加载模块统一封装起来
# processor 负责把raw text, video, image 转为 tensor
# model 负责用该tensor算hidden state, logits, loss

DEFAULT_MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"

def build_video_llava(
    model_id: str = DEFAULT_MODEL_ID,
    dtype: torch.dtype = torch.bfloat16,
): 
    processor = AutoProcessor.from_pretrained(
        model_id
    )

    processor.tokenizer.padding_side = 'left'

    model = (
        VideoLlavaForConditionalGeneration
        .from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
    )

    return model, processor
