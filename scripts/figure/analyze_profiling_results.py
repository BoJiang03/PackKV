#!/usr/bin/env python3
"""
分析脚本用于解析profiling结果并生成可读报告
该脚本帮助理解CUDA kernel与PyTorch操作的对应关系
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def parse_kernel_name(kernel_name: str) -> Dict[str, str]:
    """
    解析CUDA kernel名称，提取有用信息
    """
    info = {
        'original_name': kernel_name,
        'operation_type': 'unknown',
        'data_type': 'unknown',
        'kernel_type': 'unknown'
    }
    
    # 检测GEMV/GEMM操作
    if 'gemv' in kernel_name.lower():
        info['operation_type'] = 'GEMV (Matrix-Vector Multiplication)'
        info['kernel_type'] = 'cuBLAS'
    elif 'gemm' in kernel_name.lower():
        info['operation_type'] = 'GEMM (Matrix-Matrix Multiplication)'  
        info['kernel_type'] = 'cuBLAS'
    elif 'sgemm' in kernel_name.lower():
        info['operation_type'] = 'SGEMM (Single Precision GEMM)'
        info['kernel_type'] = 'cuBLAS'
    elif 'hgemm' in kernel_name.lower():
        info['operation_type'] = 'HGEMM (Half Precision GEMM)'
        info['kernel_type'] = 'cuBLAS'
    
    # 检测数据类型
    if '__nv_bfloat16' in kernel_name:
        info['data_type'] = 'bfloat16'
    elif 'half' in kernel_name.lower():
        info['data_type'] = 'float16'
    elif 'float' in kernel_name.lower():
        info['data_type'] = 'float32'
    
    # 检测Flash Attention
    if 'flash' in kernel_name.lower() or 'fmha' in kernel_name.lower():
        info['operation_type'] = 'Flash Attention'
        info['kernel_type'] = 'Flash Attention'
    
    # 检测softmax
    if 'softmax' in kernel_name.lower():
        info['operation_type'] = 'Softmax'
        info['kernel_type'] = 'Custom'
    
    # 检测transpose
    if 'transpose' in kernel_name.lower():
        info['operation_type'] = 'Transpose'
        info['kernel_type'] = 'cuBLAS/Custom'
    
    return info

def map_pytorch_operation_to_kernel(pytorch_op: str) -> str:
    """
    将PyTorch操作映射到预期的kernel类型
    """
    mapping = {
        'ATTENTION_QK_MATMUL': '通常对应cuBLAS GEMM kernel (Q @ K^T)',
        'ATTENTION_WEIGHTS_VALUE_MATMUL': '通常对应cuBLAS GEMV/GEMM kernel (Attention @ V)',
        'GEMM_QUERY_KEY_TRANSPOSE': 'cuBLAS GEMM kernel for Q@K^T计算',
        'GEMM_ATTENTION_VALUES': 'cuBLAS GEMM/GEMV kernel for Attention@V计算',
        'ATTENTION_SOFTMAX': 'Softmax kernel',
        'ATTENTION_LAYER_FORWARD': '包含多个attention相关kernels',
        'MODEL_FORWARD_DECODE': '包含整个模型前向传播的kernels',
        'PREFILL_PHASE': '预填充阶段，通常包含大量并行计算',
        'DECODE_STEP': '单步解码，通常kernel较小'
    }
    
    for op_pattern, description in mapping.items():
        if op_pattern in pytorch_op:
            return description
    
    return '未知操作'

def create_kernel_analysis_report():
    """
    创建kernel分析报告
    """
    report = """
# CUDA Kernel 与 PyTorch 操作对应关系分析指南

## 常见的Attention计算中的Kernel类型

### 1. cuBLAS GEMV Kernels
**特征**: `internal::gemvx::kernel`, `cublasGemvParamsEx`
**对应PyTorch操作**: 
- Query @ Key^T 矩阵乘法
- Attention_weights @ Value 矩阵乘法

**示例kernel名称**:
```
std::enable_if<true, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, ...>
```

**如何识别**:
- 在TensorBoard中查找 `ATTENTION_QK_MATMUL_*` 或 `ATTENTION_WEIGHTS_VALUE_MATMUL_*` 标记
- 这些标记包含形状信息: B(batch), H(heads), S(sequence), D(dimension)

### 2. cuBLAS GEMM Kernels  
**特征**: `sgemm`, `hgemm`, `gemm`
**对应PyTorch操作**: 通用矩阵乘法

### 3. Flash Attention Kernels
**特征**: `fmha`, `flash_attn`
**对应PyTorch操作**: 优化的attention计算

### 4. Softmax Kernels
**特征**: `softmax`
**对应PyTorch操作**: attention权重归一化

## 如何在TensorBoard中追踪

### 1. 按操作类型过滤
在TensorBoard的"TRACE"视图中：
- 搜索 `ATTENTION_QK_MATMUL` 找到Q@K^T计算
- 搜索 `ATTENTION_WEIGHTS_VALUE_MATMUL` 找到Attention@V计算
- 搜索 `GEMM_` 找到具体的GEMM操作

### 2. 查看调用栈
- 展开trace中的操作
- 查看"Call Stack"了解调用层次
- 匹配自定义标记与底层kernel

### 3. 分析性能热点
- 按时间排序找到最耗时的kernel
- 对比不同context length的性能差异
- 关注memory bandwidth vs compute utilization

## 性能优化建议

### 1. GEMV vs GEMM
- 短序列(decode阶段): 主要是GEMV操作
- 长序列(prefill阶段): 主要是GEMM操作
- GEMV通常memory-bound, GEMM通常compute-bound

### 2. 数据类型影响
- bfloat16: 平衡精度和性能
- float16: 更快但可能有数值问题  
- float32: 精度最高但最慢

### 3. 批处理大小
- 较大batch size有利于GEMM性能
- 但会增加内存占用

## 常见问题诊断

### Q: 为什么看不到Flash Attention kernels?
A: 可能因为:
1. 模型使用了eager attention实现
2. 序列长度不满足Flash Attention的要求
3. 硬件不支持

### Q: 为什么GEMV kernel占用很多时间?
A: 在decode阶段很正常，因为:
1. Decode是逐token生成，matrix-vector操作居多
2. Memory bandwidth成为瓶颈
3. 并行度相对较低

### Q: 如何判断性能瓶颈?
A: 查看:
1. Kernel occupancy (GPU利用率)
2. Memory bandwidth utilization  
3. 计算vs内存传输比例
"""
    
    return report

def analyze_profiling_directory(profile_dir: str):
    """
    分析指定目录下的profiling结果
    """
    print(f"\n分析目录: {profile_dir}")
    print("="*60)
    
    if not os.path.exists(profile_dir):
        print(f"❌ 目录不存在: {profile_dir}")
        return
    
    # 查找trace文件
    trace_files = list(Path(profile_dir).glob("*.pt.trace.json"))
    
    if not trace_files:
        print("❌ 未找到trace文件，请先运行profiling")
        return
    
    print(f"✅ 找到 {len(trace_files)} 个trace文件")
    
    # 检查TensorBoard事件文件
    tb_files = list(Path(profile_dir).glob("*.tensorboard.pt.trace.json"))
    if tb_files:
        print(f"✅ TensorBoard文件可用: {len(tb_files)} 个")
        print(f"   运行命令查看: tensorboard --logdir={profile_dir}")
    
    print("\n📊 分析建议:")
    print("1. 在TensorBoard中查看 'TRACE' 视图")
    print("2. 搜索以下关键标记:")
    print("   - ATTENTION_QK_MATMUL_* (Q@K^T计算)")
    print("   - ATTENTION_WEIGHTS_VALUE_MATMUL_* (Attention@V计算)")
    print("   - DECODE_STEP_* (单步token生成)")
    print("3. 查看kernel详细信息和调用栈")
    print("4. 对比不同context length的性能差异")

def main():
    """
    主函数 - 分析所有profiling结果
    """
    print("🔍 CUDA Kernel 分析工具")
    print("="*80)
    
    # 创建分析报告
    report = create_kernel_analysis_report()
    
    # 保存报告到文件
    report_file = "profiling_analysis_guide.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 分析指南已保存到: {report_file}")
    
    # 分析现有的profiling目录
    base_dir = "./profiling_logs"
    if os.path.exists(base_dir):
        print(f"\n📁 检查profiling结果目录: {base_dir}")
        
        # 查找所有context length目录
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith('ctx_len_'):
                analyze_profiling_directory(item_path)
    else:
        print(f"\n❌ Profiling目录不存在: {base_dir}")
        print("请先运行 mat_vec_mul_inference_profiling.py 生成profiling数据")
    
    print("\n" + "="*80)
    print("🎯 快速开始指南:")
    print("1. 运行profiling: python mat_vec_mul_inference_profiling.py")
    print("2. 启动TensorBoard: tensorboard --logdir=./profiling_logs")
    print("3. 在浏览器打开: http://localhost:6006")
    print("4. 切换到 'TRACE' 视图查看详细的kernel调用")
    print("5. 使用搜索功能找到特定的operation标记")
    print("="*80)

if __name__ == "__main__":
    main() 