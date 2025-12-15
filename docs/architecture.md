## Architecture

本仓库是一个 monorepo，目标是把「可运行应用」与「共享库」解耦，并把模型/推理相关的运行时资源组织到统一结构里。

### 目录结构（顶层约定）

```text
repo_root/
├─ apps/         # 可运行应用（独立环境）
├─ libs/         # 共享代码库（被 apps 引用）
├─ infra/        # 基础设施（Docker / Compose 等）
├─ scripts/      # 自动化脚本（借用某个 app 环境运行）
├─ docs/         # 全局文档
├─ assets/       # 示例素材/静态资源
├─ models/       # 本地模型权重（Git 忽略）
├─ data/         # 本地临时缓存（Git 忽略）
├─ logs/         # 本地日志（Git 忽略）
├─ tests/        # 全局集成测试（可选）
└─ third_party/  # 上游仓库（只读，尽量不在此做业务逻辑）
```

### Apps / Libs 依赖规则

- `apps/*`：独立应用，必须有自己的依赖声明、README、env.example、tests。
- `libs/*`：共享代码，不允许反向 import `apps/*`。
- App 依赖 Libs 必须使用绝对导入：`from libs.xxx import ...`。
- 禁止 `apps/A` 横向 import `apps/B`；通用逻辑下沉到 `libs/`。

### third_party 策略

`third_party/` 只用于保存上游代码（例如 One-to-All-Animation），保持“纯粹/只读”：

- 业务逻辑、路径解析、运行时目录管理等迁移到 `apps/` / `libs/`。
- 本地运行产生的缓存/产物不写入 `third_party/`，统一写入 `DATA_DIR`（默认 `data/`）。

### 运行时目录（本地）

- `MODELS_DIR`（默认 `models/`）：本地模型权重与静态资源（不提交 Git）。
- `DATA_DIR`（默认 `data/`）：临时缓存/中转目录（不提交 Git，不作为最终产物存储）。
- `LOG_DIR`（如需要）：本地日志目录；生产环境更推荐 stdout。

### S3/MinIO（持久化产物）

任何需要持久化的文件（上传、生成结果、数据集等）建议走 S3/MinIO，而不是长期放在 `data/`。
关键环境变量：`S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_NAME`。

## One-to-All / One-to-All-Animation 模型与推理资源

目标：把推理所需的权重/资源集中在 `models/One-to-All-14b/`，并保持 `third_party/One-to-All-Animation` 仅作为上游代码（只读）。

### 最小必需内容（14b）

推理链路由两部分组成：

1. **One-to-All-14b（微调后的 transformer 权重）**
   - 目录：`models/One-to-All-14b/`
   - 典型文件：`model.safetensors.index.json` + `model-*.safetensors` + `configuration.json`

2. **One-to-All-Animation 推理依赖的“预训练资源”（不是 One-to-All 微调权重）**
   - 目录：`models/One-to-All-14b/pretrained_models/`
   - 包含：
     - `process_checkpoint/det/yolov10m.onnx`
     - `process_checkpoint/pose2d/vitpose_h_wholebody.onnx`
       - 也可能是一个目录，里面包含 `end2end.onnx`（同样可用）
     - `DWPose/`（若干 onnx 文件）
     - `Wan2.1-T2V-14B-Diffusers/` 下的子目录：
       - `vae/`
       - `text_encoder/`
       - `tokenizer/`
       - `scheduler/`

关键点：**不需要** `Wan2.1-T2V-14B-Diffusers/transformer/` 的 base 权重，因为 One-to-All 会加载 `models/One-to-All-14b/` 的微调 transformer 权重。

### 推荐目录示意

```text
models/
└─ One-to-All-14b/
   ├─ configuration.json
   ├─ model.safetensors.index.json
   ├─ model-00001-of-....safetensors
   └─ pretrained_models/
      ├─ DWPose/
      ├─ process_checkpoint/
      │  ├─ det/yolov10m.onnx
      │  └─ pose2d/vitpose_h_wholebody.onnx  (file or dir containing end2end.onnx)
      └─ Wan2.1-T2V-14B-Diffusers/
         ├─ scheduler/
         ├─ text_encoder/
         ├─ tokenizer/
         └─ vae/
```

### 下载与迁移（推荐命令）

所有脚本都应借用 App 环境运行。

1. 下载 One-to-All 14b checkpoint（ModelScope）：
   - `uv sync --project apps/api --extra model_download`
   - `uv run --project apps/api scripts/download_modelscope_one_to_all.py --model 14b`

2. 下载 One-to-All-Animation 需要的预训练资源（默认精简版，不含 Wan transformer）：
   - `uv sync --project apps/api --extra model_download`
   - `uv run --project apps/api scripts/download_one_to_all_animation_pretrained.py --with-wan-14b`
   - 如确实需要 base Wan `transformer/`：加 `--with-wan-transformer`（通常不需要）

3. 如果历史上已经下载到 `models/One-to-All-Animation/pretrained_models/`，迁移到新结构：
   - `uv run --project apps/api scripts/prepare_one_to_all_14b_model_repo.py`

4. 如果 Wan Diffusers 目录里残留了 `transformer/`（体积巨大且冗余），可清理：
   - `uv run --project apps/api scripts/slim_wan_diffusers_repo.py --apply`

### 验证与清理

验证 `models/One-to-All-14b/` 是否包含推理所需资源：

- `uv run --project apps/worker scripts/verify_one_to_all_14b_model_repo.py`

验证通过后，可以删除历史遗留目录：

- `rm -rf models/One-to-All-Animation`

### 推荐环境变量（.env）

- `MODELS_DIR=models`
- `ONE_TO_ALL_MODEL_DIR=One-to-All-14b`（或 FP8：`One-to-All-14b-FP8`）
- `ONE_TO_ALL_PRETRAINED_DIR=models/One-to-All-14b/pretrained_models`
- `WAN_T2V_14B_DIFFUSERS_DIR=models/One-to-All-14b/pretrained_models/Wan2.1-T2V-14B-Diffusers`
- `ONE_TO_ALL_RUNTIME_DIR=data/one_to_all_animation_runtime`

