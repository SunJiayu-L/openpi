#!/bin/bash
# 使用 bash 解释器执行该脚本。

#SBATCH -p yukaichenglab
# 提交到名为 yukaichenglab 的 SLURM 分区。
#SBATCH --gres=gpu:1
# 申请 1 张 GPU。
#SBATCH --nodelist=gnho034
# 指定运行节点为 gnho034。
#SBATCH -J pi05_libero_eval_all
# 设置作业名，方便在 squeue 中识别。
#SBATCH -o trainlogs/pi05_libero_eval_all.%j.log
# 标准输出日志文件，%j 会替换为作业 ID。
#SBATCH -e trainlogs/pi05_libero_eval_all.%j.err
# 标准错误日志文件，%j 会替换为作业 ID。
#SBATCH --cpus-per-task=16
# 给该任务分配 16 个 CPU 核。
#SBATCH --mem=64G
# 给该任务分配 64GB 内存。

echo "Job ID: $SLURM_JOB_ID"
# 打印当前 SLURM 作业 ID。
echo "Node: $SLURMD_NODENAME"
# 打印实际运行节点名称。
echo "Start time: $(date)"
# 打印脚本开始时间。

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
# 切换到项目根目录，确保后续相对路径都正确。

# Offline mode
export HF_HOME=~/.cache/huggingface
# 指定 Hugging Face 缓存目录。
export HF_HUB_OFFLINE=1
# 开启离线模式，只使用本地缓存，不访问外网。

# Headless GPU rendering
export MUJOCO_GL=egl
# MuJoCo 使用 EGL 后端，支持无显示器渲染。
export MUJOCO_EGL_DEVICE_ID=0
# 指定 EGL 使用第 0 号 GPU。
export PYOPENGL_PLATFORM=egl
# PyOpenGL 也使用 EGL 平台，和 MuJoCo 保持一致。

# ============================================================
# Step 1: Start policy server in background
# ============================================================
CKPT_DIR="checkpoints/pi05_libero/my_experiment/29999"
# 指定评估所用 checkpoint 目录。
echo "Starting policy server with checkpoint: $CKPT_DIR"
# 打印将要加载的 checkpoint 路径。

uv run scripts/serve_policy.py \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$CKPT_DIR" &
# 后台启动策略服务：
# - 监听 8000 端口
# - 以 checkpoint 模式加载策略
# - 使用 pi05_libero 配置
# - 从 CKPT_DIR 读取参数
SERVER_PID=$!
# 记录后台服务进程 PID，供后续检查/清理。
echo "Policy server PID: $SERVER_PID"
# 打印服务进程 PID。

# Wait for server to be ready
echo "Waiting for policy server to start..."
# 提示正在等待服务就绪。
for i in $(seq 1 300); do
    # 最多等待 300 秒（每秒一次探测）。
    if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
        # 访问健康检查接口成功，说明服务已就绪。
        echo "Policy server is ready! (waited ${i}s)"
        # 打印实际等待时长。
        break
        # 跳出等待循环，进入评估阶段。
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        # 若进程已不存在，说明服务启动失败或异常退出。
        echo "ERROR: Policy server process died"
        # 打印错误信息。
        exit 1
        # 直接退出脚本并返回非 0 状态。
    fi
    sleep 1
    # 每轮等待 1 秒，避免忙轮询。
done

# ============================================================
# Step 2: Run LIBERO evaluation on all 4 suites
# ============================================================
LIBERO_VENV=examples/libero/.venv
# LIBERO 评估环境的虚拟环境路径。
source "$LIBERO_VENV/bin/activate"
# 激活该虚拟环境。
export PYTHONPATH=$PWD/third_party/libero:$PYTHONPATH
# 把本地 third_party/libero 加入 Python 模块搜索路径。

NUM_TRIALS=50
# 每个 task 跑 50 次 rollout。
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
# 要评估的 4 个 LIBERO suite 列表。

for SUITE in "${SUITES[@]}"; do
    # 依次评估每个 suite。
    VIDEO_OUT="data/libero/videos/${SUITE}"
    # 每个 suite 的视频输出目录。
    echo ""
    # 输出空行，便于日志分段阅读。
    echo "=========================================="
    # 打印分隔线。
    echo "Evaluating: $SUITE (${NUM_TRIALS} trials per task)"
    # 打印当前 suite 和 trial 数。
    echo "Start: $(date)"
    # 打印当前 suite 开始时间。
    echo "=========================================="
    # 打印分隔线。

    python examples/libero/main.py \
        --args.task-suite-name "$SUITE" \
        --args.num-trials-per-task "$NUM_TRIALS" \
        --args.video-out-path "$VIDEO_OUT" \
        --args.host 0.0.0.0 \
        --args.port 8000
    # 运行 LIBERO 客户端评估：
    # - 指定 suite
    # - 指定每个 task rollout 次数
    # - 指定视频输出路径
    # - 连接本机 8000 端口上的策略服务

    echo "Finished $SUITE at $(date)"
    # 打印当前 suite 评估完成时间。
done

deactivate
# 退出虚拟环境。

# ============================================================
# Cleanup
# ============================================================
echo ""
# 输出空行。
echo "Stopping policy server..."
# 提示开始关闭后台策略服务。
kill $SERVER_PID 2>/dev/null
# 向策略服务发送终止信号（忽略错误输出）。
wait $SERVER_PID 2>/dev/null
# 等待策略服务进程退出（忽略错误输出）。

echo "=========================================="
# 打印分隔线。
echo "All evaluations complete!"
# 提示所有 suite 评估结束。
echo "End time: $(date)"
# 打印整体结束时间。
echo "Videos saved to: data/libero/videos/{libero_spatial,libero_object,libero_goal,libero_10}/"
# 打印视频保存路径提示。
echo "=========================================="
# 打印分隔线。
