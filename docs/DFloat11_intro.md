# DFloat11（DF11）原理速记

## 1. 它在干什么？

**目标：**  
在**不改变数值、不改推理代码**的前提下，把 BF16 权重从 16 bit 压成约 **11 bit/权重** 的可逆格式，并在 GPU 上实时还原回 **原始的 BF16 位串**。

> 推理结果（logits、token 输出、生成图像等）与原始 BF16 完全 bit-exact 一致。

---

## 2. 核心观察：只动 exponent 就够了

BF16 结构（共 16 bit）：

- sign：1 bit  
- exponent：8 bit  
- mantissa：7 bit  

对大模型权重做统计后发现：

- sign、mantissa 接近满熵 → **几乎压不动**
- exponent 熵大约只有 ~2.6 bit（远小于 8 bit）  
  → **有很大压缩空间**

因此 DF11 选择：

- **sign + mantissa：原样存成 1 个字节（无损、无编码）**
- **exponent：单独抽出来，用 Huffman 编码压缩**

平均下来：1（sign）+7（mantissa）+2.6（编码后的 exponent）≈ 10.6 bit  
再加上一点元数据，整体约等于 **11 bit / 权重**。

---

## 3. DF11 的存储格式

对一个 BF16 权重数组 `W`：

1. **拆位：**
   ```text
   raw      = BF16 16 bit
   sign     = raw >> 15
   exponent = (raw >> 7) & 0xFF
   mantissa = raw & 0x7F
````

2. **固定 1 字节存 sign+mantissa：**

   ```text
   packed_sm = (sign << 7) | mantissa  // 高 1 bit 是 sign，低 7 bit 是 mantissa
   PackedSignMantissa[] = packed_sm 的数组，长度 = 权重个数
   ```

3. **只压 exponent：**

   * 收集所有 exponent，统计频率
   * 基于频率构造 Huffman 树，给每个 exponent 一个变长 0/1 串
   * 把所有 exponent 的码字串起来 → bitstream → 打包成字节数组

   得到：

   ```text
   EncodedExponent: 对 exponent 的 Huffman bitstream
   ```

4. **解码辅助结构：**

   为了在 GPU 上高效解码变长码，DF11预计算：

   * **多级 LUT（层次化查找表）**：
     把 Huffman 树按高度 8 分段，每段生成一个 256 项的小查表：

     * 输入：8 bit 前缀
     * 输出：

       * 要么是最终 exponent 值
       * 要么是“指向下一层 LUT 的标签”（用 240–255 等未使用 exponent 值充当）

   * **CodeLengths[exponent]**：存每个 exponent 的码长（bit 数）

   * **Gaps**：每个 GPU 线程应从该字节片段的第几个 bit 开始解码（避免跨线程时落在 code 中间）

   * **BlockOutputPos**：每个 block 解出来的 exponent 在全局输出数组里的起始 index

---

## 4. “解压”过程：本质是查找 + 位拼装

在推理时，解压过程分两步（在 GPU 上完成）：

### 4.1 只解 exponent，数一数每线程有多少个元素

每个线程：

* 从 `Gaps` 给定的 bit 偏移开始
* 循环：

  1. 一次从 `EncodedExponent` 中读取若干字节到寄存器（例如 4 字节）
  2. 用多级 LUT 解出当前 exponent（纯查表，无浮点）
  3. 用 `CodeLengths[exponent]` 决定向前移动多少 bit
  4. `NumElements[thread]++`

然后 block 内做一次 prefix-sum：得到每个线程在全局输出中的起始位置 `ThreadOutputPos[thread]`。

### 4.2 再解 exponent，一边解一边还原 BF16

再跑一遍类似的循环，这次：

对每个解出来的 exponent：

```text
byte     = PackedSignMantissa[ThreadOutputPos[thread]]
sign     = byte >> 7
mantissa = byte & 0x7F

bf16 = (sign << 15) | (exponent << 7) | mantissa
Outputs[ThreadOutputPos[thread]] = bf16
ThreadOutputPos[thread]++
```

**关键点：**

* **没有任何数值计算**（没有乘、加、scale 等）
* “解压”只是：

  * 从 bitstream 解出 exponent（通过查 LUT）
  * 和原样保存的 sign、mantissa 做简单位拼装

> 所以说：**DF11 的“解压”从数值角度其实不解码，只是查找 + 拼装原始 BF16 位串。**

---

## 5. 推理时如何使用 DF11 模型？

整体流程：

1. **离线预处理一次：**

   * 从原 BF16 checkpoint 中提取权重
   * 统计 exponent 分布 → 构建 Huffman 树 → 生成 LUT、CodeLengths
   * 生成 PackedSignMantissa、EncodedExponent、Gaps、BlockOutputPos
   * 保存为 DF11 模型（原 BF16 可以扔）

2. **上线：**

   * 将 DF11 模型完整载入 GPU 显存（比原 BF16 小约 30%）
   * 推理时对每个 transformer block：

     * 调用 DF11 kernel 解出该 block 所需权重的 BF16 数组
     * 用现有 BF16 GEMM/Attention kernel 正常计算
     * 用完后丢弃临时 BF16，只保留 DF11 版本常驻显存

---

## 6. 和传统量化的区别

| 特性   | DFloat11                             | INT8/INT4 等量化                 |
| ---- | ------------------------------------ | ----------------------------- |
| 精度   | 完全 bit-exact，无任何数值误差                 | 有量化误差，可能引起行为变化                |
| 算子改动 | 不需要改 matmul/attention，还是 BF16 kernel | 需要特定的 INT8/INT4 kernel，或反量化操作 |
| 压缩率  | ~11 bit / weight（约 0.69× BF16 体积）    | 更高（8/4/2 bit），但要权衡精度和稳定性      |
| 计算开销 | 多一点“查表+位运算”解压开销                      | 量化/反量化开销 + 特殊 kernel          |
| 适用场景 | 要求行为完全一致（合规/回溯/对比实验）                 | 可以接受近似、追求更高压缩/吞吐的场景           |

---

## 7. 一句话描述给 AI 听

> **DFloat11 是一种专门针对 BF16 权重的无损压缩格式：
> 把 BF16 的 sign+mantissa 原样装进 1 字节，只对 exponent 做熵编码存成 bitstream；
> 推理时通过查表把 exponent 解回 8 bit，再和原 sign+mantissa 拼装成原始 BF16，
> 从而在几乎不改推理代码的情况下，把模型体积从 16 bit 压到约 11 bit。**

---

## 8. SteadyDancer-14B 的 DF11 使用说明

本节记录如何用仓库内脚本，对 SteadyDancer-14B 的 Wan 主干 + T5 + CLIP 做 DF11 压缩，并生成一个自洽的 DF11 权重目录。

### 8.1 目录结构与目标（SteadyDancer）

下载完成后，原始 SteadyDancer-14B 模型目录（示例）：

```text
models/SteadyDancer-14B/
  config.json
  configuration.json
  diffusion_pytorch_model-00001-of-00007.safetensors
  ...
  diffusion_pytorch_model-00007-of-00007.safetensors
  diffusion_pytorch_model.safetensors.index.json
  Wan2.1_VAE.pth
  models_t5_umt5-xxl-enc-bf16.pth
  models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
  xlm-roberta-large/
  assets/
  ...
```

---

## 9. One-to-All-14B（Wan2.1）DF11 推理实践与排错（本仓库）

这部分是**“效果很差/很糊”**时的快速排查清单，重点结论是：在链路正确时，DF11/FP8/BF16 的首帧质量通常非常接近；明显变糊更多来自**推理参数过低**（尤其是 steps/分辨率）。

### 9.1 推荐基线参数（先用它确认链路没错）

- `num_inference_steps=30`
- `ONE_TO_ALL_MAX_SHORT=576`（upstream 默认；更小会省显存但明显损失细节）
- `ONE_TO_ALL_MAIN_CHUNK` 建议先不设，走自动（<=36GB 显存默认 33，否则 81）

### 9.2 DF11 block offload 推荐开关

参考 `apps/worker/env.example`：

- `ONE_TO_ALL_USE_DF11=1`
- `ONE_TO_ALL_DF11_PREFER_BLOCK_OFFLOAD=1`（优先使用 DF11 的块级 offload）
- `ONE_TO_ALL_DF11_CPU_OFFLOAD=1`
- `ONE_TO_ALL_DF11_PIN_MEMORY=1`
- `ONE_TO_ALL_DF11_TEXT_ENCODER=1`（若存在 `text_encoder_df11/`）
- `ONE_TO_ALL_PRECOMPUTE_PROMPT_EMBEDS=1` + `ONE_TO_ALL_DROP_TEXT_ENCODER_AFTER_PROMPT=1`（更省显存，且避免每个 chunk 重复 T5）

### 9.3 典型“质量很差”的根因：steps/分辨率太低

如果你用过类似下面这类组合做 smoke test：

- `num_inference_steps=2~5`
- `ONE_TO_ALL_MAX_SHORT=256`
- 只跑前 1 个 chunk（debug）

那么“糊/纹理水波纹/不稳定”是预期行为，并不能说明 DF11 或 FP8 有问题。

仓库推理 wrapper 已经加了运行时提示：

- 当 `num_inference_steps < 10` 会打印 `[warn] low num_inference_steps...`
- 当 `min_side<=256` 或 `max_short<384` 会打印 `[warn] low resolution...`
- 会打印最终 resize（原图 -> 实际推理分辨率）与 split plan 信息

代码位置：`libs/pycore/src/py_core/one_to_all_animation_infer.py`

### 9.4 快速 A/B 用的 debug 开关（只用于定位，不用于最终结果）

- `ONE_TO_ALL_DEBUG_MAX_FRAMES`：截断 pose 序列的帧数（更快但效果会变差）
- `ONE_TO_ALL_DEBUG_MAX_CHUNKS`：只跑前 N 个 chunk（更快但视频不完整）

### 9.5 复现命令（推荐用 worker 的 venv 直接跑，避免 uv 额外解析依赖）

在 repo root 执行（示例输入来自 upstream examples）：

```bash
ONE_TO_ALL_DEVICE=cuda:0 \
ONE_TO_ALL_USE_DF11=1 \
ONE_TO_ALL_DF11_PREFER_BLOCK_OFFLOAD=1 \
ONE_TO_ALL_DF11_CPU_OFFLOAD=1 \
ONE_TO_ALL_DF11_PIN_MEMORY=1 \
ONE_TO_ALL_MAX_SHORT=576 \
apps/worker/.venv/bin/python scripts/infer_one_to_all_animation.py \
  --model-name 14b_df11 \
  --reference-image third_party/One-to-All-Animation/examples/joker2_resize.png \
  --motion-video third_party/One-to-All-Animation/examples/douyinvid5_v2.mp4 \
  --num-inference-steps 30
```

如果你想用 `uv run`，且机器环境会限制 `~/.cache/uv`，可以设置一个可写缓存目录：

```bash
UV_CACHE_DIR=$PWD/data/uv_cache uv run --project apps/worker ...
```

### 9.6 常见环境坑（尤其是在容器里）

- `torch.cuda.is_available()` 必须是 `True`；否则 DF11（依赖 CUDA/CuPy kernel）无法工作。
- `nvidia-smi` 如果报 “Failed to initialize NVML”，通常意味着容器未正确获得 `utility` 能力或 NVML 初始化受限；需要检查容器启动参数（`NVIDIA_VISIBLE_DEVICES`、`NVIDIA_DRIVER_CAPABILITIES=compute,utility`）以及设备节点权限。
- 网络受限时，`uv run` 可能会尝试从 PyPI 拉取构建依赖（例如 `setuptools`）；这种情况下优先使用已经创建好的 `apps/worker/.venv/bin/python` 来跑推理脚本。

DF11 压缩后的目标结构（仅保留 DF11 权重，不重复原始 BF16 / FP16 权重）：

```text
models/SteadyDancer-14B-df11/
  # Wan diffusion 主干的 DF11 权重（单文件）
  model.safetensors
  config.json              # 仅包含 dfloat11_config

  # VAE 仍使用原始权重（未接入 DF11）
  Wan2.1_VAE.pth

  # T5 encoder 的 DF11 权重
  t5_df11/
    model.safetensors
    config.json          # dfloat11_config（T5）

  # CLIP 模型的 DF11 权重
  clip_df11/
    model.safetensors
    config.json          # dfloat11_config（CLIP）

  # 其他非权重资产（从原目录复制）
  xlm-roberta-large/
  assets/
  configuration.json
  README.md
  LICENSE.txt
  ...
```

注意：

- `SteadyDancer-14B` 目录中的原始权重始终保留，用于对比 / 调试：
  - diffusion BF16 分片：`diffusion_pytorch_model-*.safetensors` + index。
  - VAE：`Wan2.1_VAE.pth`。
  - T5：`models_t5_umt5-xxl-enc-bf16.pth`。
  - CLIP：`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`。
- `SteadyDancer-14B-df11` 目录中：
  - 不会保留任何 `diffusion_pytorch_model-*.safetensors` 或 index 文件。
  - 会只复制 VAE 的原始权重 `Wan2.1_VAE.pth`，保证 DF11 目录自洽；其余 `.pth` / `.safetensors`（T5 / CLIP / 其他）不会复制，只保留 DF11 导出的 `model.safetensors`。
  - 目前仅提供 diffusion 主干 / T5 / CLIP 的 DF11 权重；VAE 未接入 DF11，仍使用 `Wan2.1_VAE.pth`。

### 8.2 压缩脚本与依赖

- 脚本入口：`scripts/compress_steadydancer_dfloat11.py`
- 依赖：
  - 已下载好的 SteadyDancer-14B 权重（见 `scripts/download_models.py`）。
  - `dfloat11` Python 包（建议 CUDA 12 版本）：

    ```bash
    uv run --project apps/worker pip install 'dfloat11[cuda12]'
    ```

  - 上游 SteadyDancer 源码：`third_party/SteadyDancer/`（仓库自带）。

### 8.3 基本用法（压缩 SteadyDancer-14B）

在仓库根目录，使用 worker 的 Python 环境运行：

```bash
uv run --project apps/worker python scripts/compress_steadydancer_dfloat11.py \
  --no-check-correctness
```

行为说明：

- 自动解析 checkpoint 目录：
  - 优先使用 `--ckpt-dir`
  - 否则使用 `$STEADYDANCER_CKPT_DIR`
  - 否则使用 `<MODELS_DIR>/SteadyDancer-14B`
- 输出目录默认是 `<ckpt_dir>-df11`，例如：
  - `models/SteadyDancer-14B-df11`
- 压缩内容：
  - Wan diffusion 主干（`wan.modules.model_dancer.WanModel`）
  - T5 encoder（`wan.modules.t5.umt5_xxl(encoder_only=True)`）
  - CLIP 模型（`wan.modules.clip.clip_xlm_roberta_vit_h_14`）
- 路径约定：
  - T5 / CLIP ckpt 路径来自 `wan.configs.WAN_CONFIGS['i2v-14B']` 中的 `t5_checkpoint` / `clip_checkpoint` 字段。

可选参数：

```bash
uv run --project apps/worker python scripts/compress_steadydancer_dfloat11.py \
  --ckpt-dir models/SteadyDancer-14B \
  --save-dir models/SteadyDancer-14B-df11 \
  --threads-per-block 512 \
  --skip-diffusion \
  --no-check-correctness \
  --skip-t5 \
  --skip-clip
```

- `--threads-per-block`：
  - 覆盖 DF11 内部的 `threads_per_block`，写入 `dfloat11_config`。
  - 默认 `0` 表示使用上游默认值（通常是 512）。
- `--skip-diffusion`：
  - 跳过 Wan diffusion 主干的 DF11 压缩，只运行 T5 / CLIP（如果未显式 `--skip-t5` / `--skip-clip`）。
  - 适合已经完成 diffusion 主干压缩、只想补跑 T5 / CLIP 的场景。
- `--no-check-correctness`：
  - 关闭 DF11 的 GPU bit‑exact 校验，避免 14B 模型在小显存卡上 OOM。
- `--skip-t5` / `--skip-clip`：
  - 跳过 T5 / CLIP 的 DF11 压缩，只压 Wan diffusion 主干。

### 8.4 压缩策略与幂等性

压缩脚本对不同组件采用了统一的“按压缩率选择性压缩”策略（仅在 `--no-check-correctness` 时启用）：

- 对 Wan diffusion 主干 / T5 / CLIP 的每个 `nn.Linear` / `nn.Embedding`：
  - 先按 DFloat11 的流程完成一次编码；
  - 计算压缩率：
    - `compressed_bytes / original_bytes * 100`（原始 BF16 为 2 字节 / 元素）；
  - 若压缩率 `<= 100%`：
    - 将该层真正转换为 DF11，删除原始 `weight`，注册 DF11 所需的 buffer；
    - `dfloat11_config.pattern_dict` 中会记录该模块；
  - 若压缩率 `> 100%`：
    - 跳过该层，仅打印提示，不做 DF11 转换；
    - 该层在最终的 `model.safetensors` 中仍以 BF16 形式保存；
    - 对应模块不会写入 `dfloat11_config.pattern_dict`。

这样可以避免“压缩后反而变大”的极端层，同时保证：

- 绝大多数组件用 DF11 节省显存；
- 少数压不动的层保留为 BF16，不影响正确性。

脚本是幂等的：

- 重复运行会重新生成 DF11 单文件，但目录结构保持不变；
- 若已存在 `t5_df11/model.safetensors` 或 `clip_df11/model.safetensors`，对应子模块会跳过重复压缩。

### 8.5 推理时的接入思路（高层）

当前仓库只提供 **离线压缩脚本**，不强行绑死推理路径；接入方式建议：

- Wan diffusion 主干：
  - 使用 `DFloat11Model.from_pretrained(dfloat11_model_name_or_path=...)` 挂载到现有 `WanModel` 上，类似官方 FLUX/Wan2.1 示例。
- T5 / CLIP：
  - 同样可以通过 DF11 提供的 API 针对各自 DF11 子目录构建 `DFloat11Model`，或在自定义加载逻辑里读取 `t5_df11` / `clip_df11` 的 `dfloat11_config`。

### 8.6 apps/worker 集成（DF11 开关）

当前仓库已经在 `apps/worker` 中接入了一个简单的 DF11 推理路径（单进程）：

- 环境变量开关：
  - `STEADYDANCER_USE_DF11=1` 时，Celery 任务会通过 `wan.WanI2VDancer` + `DFloat11Model.from_pretrained(...)` 在 Worker 进程内直接加载 DF11 权重；
  - 否则仍走原始的 `generate_dancer.py` BF16 CLI 路径。
- 路径约定：
  - `STEADYDANCER_CKPT_DIR`：指向原始 BF16 目录（默认 `<MODELS_DIR>/SteadyDancer-14B`），用于加载 VAE / T5 / CLIP 等组件；
  - `STEADYDANCER_DF11_DIR`：可选，默认 `<MODELS_DIR>/SteadyDancer-14B-df11`，即 `models/SteadyDancer-14B-df11`，与离线压缩脚本的默认输出目录保持一致。
- 显存策略：
  - `STEADYDANCER_DF11_CPU_OFFLOAD=1`（默认）时，DF11 权重常驻 CPU，仅在前向时按块解码到 GPU，适合 3080 等显存较小的卡；
  - `STEADYDANCER_DF11_DEVICE_MAP_AUTO=1`（实验性）时，DF11 主干会通过 Accelerate 的 `device_map=\"auto\"` 自动切分到多张 GPU 上，适合双 3080 等多卡环境，出现问题可关闭回退到单卡。

更细粒度的调度（如 FSDP + xDiT USP 多机多卡）仍建议直接参考上游 `generate_dancer.py` 与 Wan 官方文档，在专用脚本或服务中按需集成。

---

## 9. One-to-All-14b 的 DF11 导出（本仓库）

本仓库的 One-to-All-Animation 推理会读取：

- One-to-All 微调后的 transformer 权重：`models/<ONE_TO_ALL_MODEL_DIR>/`
- Wan base 资源：`models/<ONE_TO_ALL_MODEL_DIR>/pretrained_models/Wan2.1-T2V-14B-Diffusers/`

为了在不保留原始 BF16 分片的情况下导出一个“自洽可跑”的 DF11 目录，本仓库提供脚本：

- `scripts/compress_one_to_all_14b_dfloat11.py`

### 9.1 目标结构（示例）

```text
models/One-to-All-14b-DF11/
  configuration.json
  pretrained_models/
    Wan2.1-T2V-14B-Diffusers/
      text_encoder/           # 仅保留 config / tokenizer 相关文件（无 BF16 分片）
      text_encoder_df11/      # DF11 权重（safetensors + config.json: dfloat11_config）
      vae/
      tokenizer/
      scheduler/
    DWPose/
    process_checkpoint/
  transformer_df11/           # One-to-All transformer 的 DF11 权重
```

### 9.2 压缩/导出命令

在仓库根目录，使用 worker 环境运行：

```bash
uv sync --project apps/worker --extra one_to_all_animation --extra df11
uv run --project apps/worker python scripts/compress_one_to_all_14b_dfloat11.py \
  --ckpt-dir models/One-to-All-14b \
  --save-dir models/One-to-All-14b-DF11
```

### 9.3 推理侧启用

导出后，设置环境变量：

```bash
ONE_TO_ALL_MODEL_DIR=One-to-All-14b-DF11
ONE_TO_ALL_USE_DF11=1
```

然后按 `apps/worker` / `apps/api` 的正常方式启动推理即可。
