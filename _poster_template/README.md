# PDCAT 2026 紫白色 LaTeX 海报模板

这是基于此前 `3.pdf` / PDCAT2026 Visio 模板布局重新编写的 LaTeX 作者模板。保留 A0 竖版、顶部标题与会议信息，以及 Motivation、Method、Experimental Results 三个分区和原有图表区域。内容沿用当前 PDCAT2026 作者模板的占位文字，供作者填写自己的研究。

页面为 **841 × 1189 mm（宽 × 高）**。顶部标题与会议 logo 区域使用白底、深色文字和紫色会议名称，让猫咪的黑色线条和紫色眼镜清晰显示。分区条与页脚使用最初版本的紫色 `#741B5B`，搭配白色内容区和极浅的紫色背景；所有色块与细线均由 TikZ 绘制。会议猫咪 logo 使用透明背景的矢量 PDF，保留原来的黑色线条、紫色眼镜和红色项圈。标题、正文、图框和 logo 的大小与位置沿用上一版。

## 在 Overleaf 上使用

1. 下载 `PDCAT2026_LaTeX_Purple_Overleaf.zip`。
2. 在 Overleaf 新建项目时选择上传项目 ZIP。
3. 将主文件设为 `main.tex`，编译器选择 **XeLaTeX**。
4. 修改 `main.tex` 顶部的标题、作者、单位、Paper ID 和联系方式。
5. 修改三个正文分区的文字，将图表放入 `figures/`，重新编译并下载 PDF。

工程只用 TeX Live 常见宏包和随发行版提供的字体，不需要安装 Visio、额外下载字体或启用 shell escape。支持中文姓名、中文正文和常见数学公式。

## 修改文字与图片

- `main.tex`：作者通常只需要编辑此文件。正文在三个编号分区中，可直接使用 `\textbf{...}`、`\(...\)`、`itemize` 等 LaTeX 语法。
- `pdcatposter.sty`：配色、字体、图框和分区条样式。保持统一版式时无需修改。
- `assets/PDCAT_Logo_Vector.pdf`：模板使用的透明背景会议矢量 logo。
- `assets/PDCAT_Logo_Vector_Transparent.svg`：相同图案的透明背景 SVG，可用于其他设计软件。
- `figures/`：作者自己的图片与图表。

默认图文件如下；放入同名文件后，占位框会自动替换为图片。

| 文件 | 图表用途 | 最大放置区域，宽 × 高 |
| --- | --- | --- |
| `figures/motivation.pdf` | 动机示例 | 297 × 146 mm |
| `figures/method.pdf` | 方法图 / 系统架构 | 522 × 258 mm |
| `figures/results.pdf` | 主要实验结果 | 579 × 173 mm |
| `figures/comparison.pdf` | 对比实验 / 表格 | 529 × 118 mm |
| `figures/qualitative.pdf` | 定性结果 / 消融实验 | 229 × 150 mm |
| `figures/qr-code.pdf` | 论文或代码二维码 | 123 × 123 mm |

也可以使用 PNG / JPG，只需修改 `main.tex` 顶部对应的文件名。图片按比例适配区域，保持完整，不拉伸。图框中的二维码只是占位文字，作者需要放入真实二维码。

标题区域预留两行，标题会自动换行。正文长度增加时，请留意编译日志中的 `Content box ... exceeds ...` 或 `Overfull` 提示；模板固定为单页 A0，不会把多余内容自动排到第二页。优先精简文字，必要时调整对应 `\PosterText` 的字号。其参数依次为：左边距、上边距、宽、最大高（均为 mm），字号、行距（均为 pt），正文。

文字中的特殊符号按 LaTeX 规则输入，例如 `\&`、`\%`、`\_`、`\#`。不要通过插入大段空白、负间距或强制换行挤内容。

## 本地编译与导出

在工程目录运行：

```sh
latexmk -xelatex main.tex
```

或直接运行 `xelatex main.tex`。输出为 `main.pdf`，尺寸已经是 A0；交给印刷方时按实际尺寸 / 100% 输出。附带的 `PDCAT2026_Poster_Purple.pdf` 是本模板已编译的预览。

文字、装饰线、图框和会议 logo 为矢量对象。作者插入的图若本身是矢量 PDF，导出时可继续保留矢量；位图放入 PDF 后不会自动变成矢量。科研曲线和框图优先从原始软件导出为 PDF。

## 验证范围

本地用 XeLaTeX 编译并检查了单页 A0 尺寸、字体嵌入、文字可提取和默认 PDF 无位图对象；另测试了两行长标题、中英文姓名、公式和实际矢量图替换。没有在 Overleaf 实际上传，也未进行纸质印刷打样。
