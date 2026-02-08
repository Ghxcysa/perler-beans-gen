# Perler-Gen（拼豆图纸生成器）技术文档（MVP 4–6 周）

> 项目目标：用户上传一张图片，自动生成对应的拼豆（Perler / Hama Beads）**可打印图纸**与**分步摆豆说明**，并输出**颜色用量清单**，形成“图片 → 图纸 → 可按步骤完成作品”的 MVP 闭环。

---

## 1. 背景与问题定义

拼豆作品本质是一个二维网格：每个网格点放置一颗指定颜色的豆。  
用户常见痛点：

- 想把照片/插画变成拼豆图纸，但手工像素化、配色、统计用量耗时
- 图纸没有步骤拆分，摆豆时容易错位、漏放
- 颜色数量过多，材料准备困难

本项目通过规则化流程，自动完成：**缩放 → 像素化 → 颜色量化 → 图纸与步骤输出 → 用量清单**。

---

## 2. MVP 范围（Scope Definition）

### 2.1 Scope Statement（范围声明）
在 4–6 周内完成一个可运行的 MVP：  
输入单张图片，输出一份可打印的拼豆 Pattern PDF（含分步页）+ 颜色用量清单 + 预览图。

### 2.2 In Scope（MVP 要做什么）

**输入**
- 单张 RGB 图片（jpg/png）
- 可选参数：裁剪、背景处理、输出网格尺寸、最大颜色数、步骤拆分方式

**核心处理**
- 网格尺寸缩放（如 29×29 / 48×48 / 58×58 等）
- 颜色量化：将像素映射到“拼豆色卡”（有限颜色集合）
- 步骤生成：将全图拆成若干步（MVP 推荐 *逐行* 或 *分区* 二选一）

**输出**
- Pattern PDF（包含：封面、预览、图例、坐标、分步页）
- Bead List（用量清单：每种颜色需要多少颗）
- Preview PNG（量化后的像素预览图）
- 可选：SVG（矢量网格图，方便二次编辑）

### 2.3 Out of Scope（MVP 不做什么）
- 不做照片级拟真（效果以“可摆豆”为准）
- 不做高质量智能抠图（MVP 只做简单背景处理或不做）
- 不做复杂摆豆路径优化（最少手部移动、防错策略等）
- 不做 3D 立体拼豆、多层结构、熨烫收缩仿真
- 不引入深度学习风格化/生成式重绘（以规则驱动为主）

### 2.4 Assumptions（假设）
- 用户使用标准拼豆与拼豆板（如 5mm 豆）
- 色卡是离散集合（MVP 先支持一套固定色表，后续可扩展）
- 用户自行完成摆豆和熨烫，本项目只负责图纸与清单

### 2.5 Constraints（约束）
- 固定网格尺寸（只提供少数档位或允许自定义但需限范围）
- 颜色数上限（默认 ≤ 24/32 色）
- 步骤拆分方式 MVP 仅支持一种（建议：逐行）
- 输出格式：PDF 为主，PNG 为辅

### 2.6 Acceptance Criteria（验收标准）
- 任意输入图片（jpg/png）能成功生成：
  - 1 份可打印 Pattern PDF（含分步摆豆页）
  - 1 份颜色用量清单（CSV 或 JSON）
  - 1 张预览 PNG
- PDF 页面内容清晰可读（网格、符号/颜色、坐标、图例齐全）
- 清单统计数量与图纸一致（可抽样验证）

---

## 3. 用户故事（User Stories）

1. **新手用户**
   - 我上传一张图片，选择网格大小与颜色数量上限
   - 我下载 PDF，按步骤页逐步摆豆即可完成

2. **进阶用户**
   - 我想指定色卡（只用我已有的颜色）
   - 我想导出 SVG 便于自己再调整细节

3. **创作者/卖图纸**
   - 我需要可打印的标准格式（带坐标、图例、用量清单）
   - 我需要不同尺寸版本（48×48、58×58 等）

---

## 4. 总体流程（Pipeline）

```
Input Image
   │
   ├─(可选) 裁剪/去背景/增强
   │
   ├─缩放到目标网格 (W×H)
   │
   ├─颜色量化 (Color Quantization)
   │     ├─像素颜色 → 色卡最近邻映射
   │     └─限制最大颜色数（可选：聚类/抖动）
   │
   ├─步骤拆分 (Steps)
   │     ├─逐行：Row 1 → Row 2 → ...
   │     └─或分区：Quadrant 1 → ...
   │
   ├─输出
   │     ├─Pattern PDF（封面+预览+图例+分步页）
   │     ├─Bead List（颜色→数量）
   │     └─Preview PNG / SVG
   │
Done
```

---

## 5. 关键模块设计

### 5.1 图像预处理（preprocess）
- 读取图片（Pillow / OpenCV）
- 可选：
  - 裁剪：居中裁剪 / 手动 bbox
  - 背景处理：简单阈值、基于边缘/颜色分割（MVP 先简化）

输出：统一的 RGB 图像

### 5.2 网格缩放（resample_to_grid）
- 将图片缩放到目标尺寸 `W×H`
- 建议使用 `NEAREST` 或 `BILINEAR` + 后续量化  
  - 若希望更像像素画：`NEAREST`
  - 若希望更平滑：`BILINEAR` / `BICUBIC`

输出：`W×H` 像素图

### 5.3 颜色量化与色卡映射（quantize）
目标：减少颜色数量，并保证最终颜色来自拼豆色卡。

策略（MVP 推荐最简单稳的一种）：
- 预先定义色卡 `palette = [(r,g,b,name,code), ...]`
- 对每个像素 `(r,g,b)` 计算到色卡颜色的距离（如 RGB 欧氏距离）
- 取最近邻颜色作为映射结果
- 可选：先做 K-Means 得到 `K` 个代表色，再映射到色卡（提升效果）

输出：
- `grid[H][W]`：每格一个色卡索引
- `counts[color]`：用量统计

### 5.4 步骤生成（step_planner）
MVP 方案：
- **逐行**：每一步输出一行（或 N 行一组）
  - 优点：实现简单，用户容易照做
  - 缺点：步骤页可能较多（可用“每步 2–4 行”缓解）
- 或 **分区**：左上、右上、左下、右下
  - 优点：页数少
  - 缺点：用户可能更容易错位（需要更明显坐标/边界）

输出：`steps = [step1_cells, step2_cells, ...]`

### 5.5 PDF 生成（export_pdf）
PDF 页面建议结构：
1. 封面：标题、网格尺寸、颜色数、预览图
2. 图例：颜色名/编号 ↔ 网格符号（建议每色一个字母/图形）
3. 全图总览（可选）
4. 分步页：每页显示当前步需要摆的格子（其他格子淡化/空白），带坐标轴

实现建议：
- Python：ReportLab（可控性强）或 FPDF2（更易上手）
- 符号策略：
  - 每个颜色分配一个短符号：A, B, C... 或 1,2,3...
  - PDF 内打印符号，避免彩色打印依赖

输出：`pattern.pdf`

### 5.6 输出清单（bead_list）
- CSV：`color_code,color_name,count`
- JSON：`{ "colors": [{"code":..., "name":..., "count":...}], "grid": {"w":..., "h":...}}`

---

## 6. 技术选型（Tech Stack）

- Python 3.10+
- 图像处理：Pillow（优先）或 OpenCV（可选）
- 数学/聚类：NumPy + scikit-learn（若使用 KMeans，可选）
- PDF：ReportLab（推荐）
- CLI：argparse / Typer（可选）
- （可选）Web Demo：Streamlit / Gradio

---

## 7. 工程结构（建议）

```
perler-gen/
  README.md
  requirements.txt
  assets/
    palettes/
      perler_basic.json
  src/
    perler_gen/
      __init__.py
      cli.py
      preprocess.py
      quantize.py
      step_planner.py
      export_pdf.py
      export_assets.py
      utils.py
  examples/
    input/
    output/
  tests/
```

---

## 8. CLI 设计（MVP）

示例：
```bash
python -m perler_gen.cli \
  --input examples/input/cat.jpg \
  --outdir examples/output/cat_48 \
  --grid 48 48 \
  --max-colors 24 \
  --palette assets/palettes/perler_basic.json \
  --steps row \
  --rows-per-step 2
```

参数说明：
- `--input`：输入图片路径
- `--outdir`：输出目录
- `--grid W H`：目标网格尺寸
- `--max-colors`：颜色上限（可选）
- `--palette`：色卡文件（JSON）
- `--steps`：`row` 或 `quadrant`（MVP 推荐 `row`）
- `--rows-per-step`：逐行模式每步包含的行数（默认 1 或 2）
- `--no-preview`：不导出 PNG（可选）
- `--export-svg`：导出 SVG（可选）

输出文件：
- `pattern.pdf`
- `preview.png`
- `bead_list.csv`（或 `.json`）

---

## 9. 色卡格式（Palette JSON）

示例：
```json
{
  "name": "Perler Basic",
  "colors": [
    {"code": "P01", "name": "White", "rgb": [255,255,255]},
    {"code": "P02", "name": "Black", "rgb": [0,0,0]}
  ]
}
```

---

## 10. 测试与验证

### 10.1 单元测试（建议）
- `quantize`：同一输入像素映射到正确色卡
- `counts`：统计总数等于 `W×H`
- `step_planner`：所有格子被覆盖且不重复（除非设计允许）

### 10.2 端到端样例
准备至少 3 张图片：
1. 高对比度卡通
2. 人像或照片
3. 复杂背景场景

验证：
- PDF 可打印、网格与符号清晰
- 清单颜色数量合理
- 用户按步骤摆豆可完成（抽样验证即可）

---

## 11. 里程碑（4–6 周）

- **第 1 周**：读取/裁剪/缩放到网格 + 输出 preview
- **第 2 周**：色卡定义 + 最近邻映射 + 用量统计
- **第 3 周**：步骤拆分（逐行）+ 生成每步数据
- **第 4 周**：PDF 导出（图例+坐标+分步页）+ 完整 CLI
- **第 5 周（可选）**：效果优化（抖动/聚类）、SVG 输出、体验增强
- **第 6 周（缓冲）**：文档、示例、测试、bugfix

---

## 12. 风险与缓解

- **颜色映射效果不理想**
  - 缓解：限制颜色数 + 可选 KMeans + 可选抖动（Floyd–Steinberg）
- **PDF 可读性不足**
  - 缓解：符号优先（不依赖彩打），坐标轴加粗清晰
- **步骤过多**
  - 缓解：`rows-per-step` 支持 2–4 行合并

---

## 13. 后续扩展（Post-MVP）

- 多色卡选择（Perler/Hama/自定义）
- 交互式编辑（手动替换某些格子颜色、橡皮擦/画笔）
- 更智能的步骤规划（减少错误、提高效率）
- Web 端上传与在线预览/编辑
- 导出“成品渲染图”（更接近真实拼豆外观）

---

## 14. 附：最小 Demo 交付清单（Checklist）

- [ ] `quantize` 可运行，输出 `preview.png`
- [ ] `bead_list.csv` 正确统计
- [ ] `pattern.pdf` 生成成功（至少包含：预览+图例+分步页）
- [ ] `examples/` 下有输入与输出示例
- [ ] README 包含安装与运行命令
