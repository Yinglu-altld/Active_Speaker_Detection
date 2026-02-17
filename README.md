# Active Speaker Detection（CNN + MediaPipe + Audio DOA + Fusion）

这个项目用于**实时检测谁在说话**（Active Speaker Detection）。

核心思路是把两路信息融合：

- 视觉：从人脸关键点判断“这个用户在说话的概率”（CNN）
- 音频：从麦克风阵列判断“声音来自哪个方向、可信度多高”（DOA）
- 融合：给每个 `user_id` 计算一个 `overall score`，得分最高的就是当前说话人

---

## 1. 项目逻辑（先理解再跑）

### 1.1 CNN（视觉）这块

相关脚本：

- `scripts/step3_extract_landmarks.py`
- `scripts/step3_batch_extract_landmarks.py`
- `scripts/step4_build_windows.py`
- `scripts/step5_train_cnn.py`
- `scripts/step6_realtime_infer.py`

流程：

1. 用 MediaPipe 提取人脸关键点（嘴唇、脸部轮廓等）
2. 拼成时间窗口特征（window）
3. 训练 `TemporalCNN`，输出每个用户的说话概率
4. 实时阶段按 `user_id` 输出 `cnn_prob`

---

### 1.2 Audio（DOA）这块

相关脚本：

- `audio_doa/doa_core.py`
- `audio_doa/srp_phat.py`

流程：

1. 读取多通道麦克风音频（如 ReSpeaker 6 通道）
2. VAD + 能量门控判断是否有语音
3. SRP-PHAT 估计 `azimuth_deg`（声源角度）
4. 输出 `conf_doa`、`conf_doa_srp`、`audio_conf`、`sigma_deg` 等 JSON 字段

---

### 1.3 Integration（融合）这块

相关脚本：

- `audio_doa/fusion.py`（融合打分函数）
- `audio_doa/run_live_fusion.py`（一键实时融合主入口）
- `audio_doa/fusion_stub.py`（调试/管线验证）

融合输出：

- 每个 `user_id` 一个 `overall score`
- `speaker_id` = 最高分用户
- 可选发送 Furhat `attend.user`

---

## 2. 从 clone 到跑通（Step-by-step）

### Step 0：clone 项目

```bash
git clone https://github.com/Yinglu-altld/Active_Speaker_Detection.git
cd Active_Speaker_Detection
```

### Step 1：创建环境并安装依赖

```bash
python -m venv venv
source venv/bin/activate
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -r audio_doa/requirements.txt
```

### Step 2：准备必须文件

至少准备以下内容：

- MediaPipe 模型：`data/models/face_landmarker_v2.task`
- 视频：`data/videos/<video_id>.mp4`
- 标注：`data/labels/<video_id>-activespeaker.csv`

如果你已经有训练好的模型，还需要：

- `data/models/cnn_vvad/best.pt`
- `data/models/cnn_vvad/config.json`
- `data/models/cnn_vvad/threshold.json`
- `data/windows/windows_info.json`

---

## 3. 最快跑全项目（推荐）

一条命令跑完整实时链路（CNN + DOA + Fusion + Furhat）：

```bash
./venv/bin/python audio_doa/run_live_fusion.py \
  --furhat-ip 192.168.1.109 \
  --attend-furhat \
  --audio-device 1 --audio-channels 6 --mic-channels 1,2,3,4 \
  --step6-extra "--show --window-width 960 --window-height 540"
```

这条命令内部会自动启动：

- `scripts/step6_realtime_infer.py`
- `audio_doa/doa_core.py`
- 融合与 Furhat 控制逻辑

---

## 4. 实时窗口怎么看

每个 bbox 上会显示：

- `c`：CNN 分数
- `d`：该用户对应的 DOA 分数
- `o`：融合后的 overall 分数
- `a`：是否为当前 active speaker（`1`/`0`）

颜色规则：

- 绿色：当前 active speaker
- 橙色：非 active speaker（但参与融合）

---

## 5. 如果你要重新训练 CNN（可选）

### 5.1 批量提取关键点

```bash
./venv/bin/python scripts/step3_batch_extract_landmarks.py --min-rows 50
```

### 5.2 构建训练窗口

```bash
./venv/bin/python scripts/step4_build_windows.py --window-sec 1.5 --hop-sec 0.5 --target-fps 25
```

### 5.3 训练和评估

```bash
./venv/bin/python scripts/step5_train_cnn.py --val-video WwoTG3_OjUg --epochs 40 --batch-size 128
./venv/bin/python scripts/step5_train_cnn.py --eval-only --test-video Ag-pXiLrd48
```

常用可选项：

- `--no-delta`：禁用 delta 特征
- `--filter-extremes --neg-max 0.1 --pos-min 0.6`：只用更“干净”的极值标签

---

## 6. 仅跑视觉（不融合）

Furhat 相机源：

```bash
./venv/bin/python scripts/step6_realtime_infer.py --source furhat --furhat-ip 192.168.1.109 --show
```

本地摄像头：

```bash
./venv/bin/python scripts/step6_realtime_infer.py --source opencv --video-device 0 --show
```

---

## 7. 参考数据与当前指标

当前仓库主要使用的 6 个视频：

- `2bxKkUgcqpk`
- `9bK05eBt1GM`
- `Ag-pXiLrd48`
- `B1MAUxpKaV8`
- `WwoTG3_OjUg`
- `a5mEmM6w_ks`

参考测试指标（held-out: `Ag-pXiLrd48`）：

- `test_f1=0.744`
- `test_acc=0.843`
- `test_pos_rate=0.251`
- `threshold=0.30`

---

## 8. 说明

- `data/` 存放大文件和中间产物，默认不纳入 git。
- 更细的音频模块说明见：`audio_doa/README.md`。
