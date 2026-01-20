# 全量预测任务说明

## 📊 任务信息

### 任务配置
- **总药物数**: 2,588 个
- **输出文件**: `/mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/data/result.json`
- **日志文件**: `/mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/code/full_prediction.log`
- **PID 文件**: `/mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/code/prediction_pid.txt`

### 预计时间
- **每个药物**: ~14 秒（包含 LLM 生成 reasoning）
- **总时间**: ~10 小时 (2,588 × 14秒 ≈ 36,232秒 ≈ 10.1小时)
- **预计完成**: 2025-12-04 下午 14:00 左右

## 🔍 监控方法

### 方法 1: 查看日志最后几行
```bash
cd /mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/code
tail -30 full_prediction.log
```

### 方法 2: 实时监控日志
```bash
cd /mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/code
tail -f full_prediction.log
```
按 `Ctrl+C` 退出监控

### 方法 3: 查看当前进度
```bash
cd /mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/code
grep "进度:" full_prediction.log | tail -1
```

### 方法 4: 查看最后处理的药物
```bash
cd /mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/code
grep "处理药物" full_prediction.log | tail -1
```

### 方法 5: 检查进程是否运行
```bash
cd /mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/code
ps aux | grep predict_efficiency_rdkit_with_reasoning | grep -v grep
```

### 方法 6: 定时自动监控（每60秒刷新一次）
```bash
cd /mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/code
watch -n 60 'tail -20 full_prediction.log'
```

## 📈 进度查看示例

日志中会显示类似以下内容：
```
处理药物 1 (分数: 7.5) - 生成理由中...
处理药物 2 (分数: 7.0) - 生成理由中...
...
进度: 100/2588 (3.9%) - 速度: 14.2 药物/秒 - ETA: 9.5小时
```

## 📁 输出文件格式

`result.json` 将包含所有药物的完整信息：
```json
[
  {
    "number": 2173,
    "smiles": "O=C(OC)C(C(C)O)N(CCC(OCCCCOC(CCCCC)=O)=O)...",
    "efficiency_score": 8.5,
    "molecular_weight": 618.0,
    "logP": 3.9,
    "hbd": 1,
    "hba": 11,
    "tpsa": 155.0,
    "rotatable_bonds": 27,
    "reasoning": "Due to the presence of... at the head and... at the tail..."
  },
  ...
]
```

## ⚠️ 注意事项

1. **不要中断**: 任务运行在后台，不要关闭终端或重启服务器
2. **磁盘空间**: 确保有足够空间（预计 result.json 约 2-3 MB）
3. **GPU 占用**: 任务会占用 GPU 资源，预计使用 ~100GB VRAM
4. **网络**: 使用离线模式（HF_HUB_OFFLINE=1），不需要网络

## 🛑 如果需要停止任务

```bash
cd /mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/code
if [ -f prediction_pid.txt ]; then
    kill $(cat prediction_pid.txt)
    echo "任务已停止"
fi
```

## ✅ 任务完成后

### 检查结果
```bash
cd /mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/data
ls -lh result.json

# 查看药物数量
python3 << EOF
import json
with open('result.json', 'r') as f:
    data = json.load(f)
print(f"总共处理: {len(data)} 个药物")
print(f"分数范围: {min(d['efficiency_score'] for d in data)} - {max(d['efficiency_score'] for d in data)}")
EOF
```

### 查看 Top 10 药物
```bash
cd /mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/data
python3 << EOF
import json
with open('result.json', 'r') as f:
    data = json.load(f)
print("Top 10 药物:")
for i, drug in enumerate(data[:10], 1):
    print(f"{i}. 药物 {drug['number']}: {drug['efficiency_score']}/10")
EOF
```

## 📞 监控命令速查

```bash
# 进入工作目录
cd /mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/code

# 查看最新进度
tail -20 full_prediction.log

# 实时监控
tail -f full_prediction.log

# 查看进度百分比
grep "进度:" full_prediction.log | tail -1

# 检查任务状态
ps aux | grep predict_efficiency | grep -v grep

# 查看已生成的部分结果（如果有）
cd ../data && ls -lh result.json 2>/dev/null
```

---

**任务启动时间**: 2025-12-04 约 04:00  
**预计完成时间**: 2025-12-04 约 14:00  
**状态**: 运行中 🚀
