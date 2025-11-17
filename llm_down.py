from modelscope import snapshot_download

# 下载Qwen3-4B-Thinking模型
model_dir = snapshot_download(
    "Qwen/Qwen3-4B-Thinking-2507",
    revision="master"
)
print(f"模型已下载到: {model_dir}")