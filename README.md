# 圆明园数字复原计划

[![Deploy](https://github.com/Ivan666jjj/yuanmingyuan-restoration/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ivan666jjj/yuanmingyuan-restoration/actions/workflows/deploy.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![Images: CC BY-SA 4.0](https://img.shields.io/badge/images-CC%20BY--SA%204.0-lightgrey.svg)](LICENSE-IMAGES)
[![Scenes](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2FIvan666jjj%2Fyuanmingyuan-restoration%2Fmain%2Fsrc%2Fdata%2Fscenes.json&query=%24.scenes.length&label=scenes%20listed&color=a52a20)](https://ivan666jjj.github.io/yuanmingyuan-restoration/)

把 1860 年前的圆明园，一块一块拼回来。

一个开放的、可核验的圆明园数字复原图册：以《圆明园四十景图咏》绢本彩绘（1744）、《西洋楼二十景》铜版画（1786）、样式雷图档与考古照片为依据，逐景复原已毁的园林与建筑。每一景都附带参考图出处、提示词与生成模型，并标注“AI 考据级复原，非考古测绘”。

- 站点：https://Ivan666jjj.github.io/yuanmingyuan-restoration/
- 图册：`/gallery`
- 方法：`/method`
- 参与：`/contribute`

## 数据与结构

```
src/data/scenes.json        景点清单（坐标、状态、参考、提示词）
public/images/restorations  复原图 WebP（1600px）
public/images/thumbs        缩略图 WebP（720px）
src/components/SceneMap.astro  景区分区地图与热区
src/pages/*.astro           首页 / 图册 / 方法 / 参与
```

新增一个景点 = 在 `scenes.json` 增加一条记录，并把两张 WebP 放进对应目录。

## 本地开发

```bash
npm install
npm run dev      # http://localhost:4321/yuanmingyuan-restoration/
npm run build
```

## 参与

见 [CONTRIBUTING.md](./CONTRIBUTING.md)。不会写代码也可以：开一个“认领景点”Issue，附上成图、提示词与出处即可。

## 许可

- 代码：MIT（见 LICENSE）
- 复原图：CC BY-SA 4.0（见 LICENSE-IMAGES）
- 历史参考图：公有领域，来源在每景详情标注

## 免责声明

本站图像为 AI 生成的历史考据复原，非考古测绘成果，不可作为修复施工或学术论文插图使用。形制与配色力求有据，细节不保证逐寸对应。