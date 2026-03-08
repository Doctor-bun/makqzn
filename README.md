# 缅A亏钱理论指南 by doctorbun

“缅A亏钱理论指南 by doctorbun” 是一个偏 A 股研究场景的本地股票分析工具。它把技术面、量价关系、财务摘要、估值、多层消息面、主线板块、资金分配和规则回测放到同一个界面里。

它不是“稳赚软件”。它的目标是把选股、买点、仓位和风险控制流程固定下来。

## 主要功能

- 单票分析
- K 线、成交量、均线、回归趋势线
- 支撑位 / 压力位识别
- 财务摘要和估值分析
- 巴菲特收益法、格雷厄姆公式、业务市值法、历史估值分位
- 公司新闻、公告、券商研报、社交舆情、行业新闻
- 全A主板排名
- 主线板块排序和板块内个股排名
- 资金分配和仓位控制
- 持仓决策和做T参考
- 规则回测
- 本地结果落库，可回看历史快照

## 项目文件

- `app.py`: Streamlit 图形界面
- `analysis_engine.py`: 单票分析、估值、回测、仓位等核心逻辑
- `market_overview.py`: 全市场预筛、主线板块、主题排名
- `local_store.py`: 本地 CSV / JSON 落库与偏好保存
- `run-stocklab.bat`: 一键启动，支持端口自动切换
- `desktop_launcher.py`: 桌面打包启动器
- `build-package.ps1`: 生成桌面可运行包

## 快速开始

### 方式一：已有 Conda 环境

```powershell
conda activate stocklab
cd /d S:\work\personal\gpt
python -m streamlit run .\app.py
```

### 方式二：一键启动

```powershell
.\run-stocklab.bat
```

指定端口：

```powershell
.\run-stocklab.bat 8051
```

如果指定端口被占用，脚本会自动往后找可用端口。

## 本地保存

程序会把以下结果保存到 `data_store/`：

- 全主板排名 CSV
- 主线板块汇总 CSV
- 板块内个股排名 CSV
- 使用偏好 JSON

你可以在界面里直接切换历史快照，并选择展示 Top 10 / 20 / 50 / 100。

## 安装说明

详细安装步骤见 [docs/INSTALL.md](docs/INSTALL.md)。

## 功能说明

详细功能说明见 [docs/FEATURES.md](docs/FEATURES.md)。

## 生成可直接运行的软件包

```powershell
conda activate stocklab
cd /d S:\work\personal\gpt
powershell -ExecutionPolicy Bypass -File .\build-package.ps1
```

生成结果在 `dist\StockLab\`，压缩包在 `dist\StockLab-portable.zip`。

## 风险边界

- 不能保证盈利。
- 估值模型是情景分析，不是确定性预测。
- 新闻和政策是程序化归纳，不替代人工研报。
- 回测验证的是规则纪律，不代表未来一定有效。
