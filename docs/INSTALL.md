# 缅A亏钱理论指南 by doctorbun 安装说明

## 环境要求

- Windows 10 / 11
- Miniconda 或 Anaconda
- Python 3.11

## 方式一：本地环境运行

### 1. 创建环境

```powershell
conda env create -f environment.yml
```

### 2. 激活环境

```powershell
conda activate stocklab
```

### 3. 启动软件

```powershell
python -m streamlit run .\app.py
```

或者：

```powershell
.\run-stocklab.bat
```

### 4. 指定端口

```powershell
.\run-stocklab.bat 8051
```

如果 8051 被占用，脚本会自动尝试后续端口。

## 方式二：构建桌面可运行包

### 1. 激活环境

```powershell
conda activate stocklab
```

### 2. 执行打包脚本

```powershell
powershell -ExecutionPolicy Bypass -File .\build-package.ps1
```

### 3. 打开输出目录

构建结果在：

```text
dist\StockLab\
```

压缩包在：

```text
dist\StockLab-portable.zip
```

## 常见问题

### 1. 页面打不开

- 优先看终端里输出的本地地址
- 默认是 `http://127.0.0.1:端口`
- 软件只监听本机，不对公网开放

### 2. 端口被占用

- 直接换端口启动：`.\run-stocklab.bat 8052`
- 或者让脚本自动寻找空闲端口

### 3. 行情抓取失败

程序已经接入多级回退源，但网络波动时仍可能短暂失败。一般重试一次即可。

### 4. 为什么没有“保证赚钱”

因为不存在可靠的保证盈利工具。这个软件做的是研究流程、信号确认和风险控制。
