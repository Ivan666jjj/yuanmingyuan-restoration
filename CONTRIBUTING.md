# 参与共建

## 三种方式

### A. 不会写代码（推荐先用这个）

1. 打开 [认领 Issue](https://github.com/Ivan666jjj/yuanmingyuan-restoration/issues/new?template=claim-scene.yml)。
2. 填写景点名、依据、制作计划。
3. 附成图（可选）。图片请先把文件存到本机，再任选一种上传：
   - 复制图片后直接在文本框里粘贴；
   - 点文本框下方的「Paste, drop, or click to add files」，选本地文件；
   - 把图片文件拖进文本框。
   上传时会出现 `<!-- Uploading "xxx.jpg"... -->`，几秒后变成 `github.com/user-attachments/...` 的链接，属于正常。
   **不要**粘贴小红书、微博、微信公众号的图片地址：这类链接有防盗链且会过期，GitHub 抓不到图，框里就会留下 `<!-- Failed to upload -->`。图片在网页上时，先存到本地或直接给出 Commons／馆藏页链接。
4. 完成后提交，维护者会合入并署名。

### B. 会 Git

1. Fork 仓库并新建分支。
2. 在 `src/data/scenes.json` 增加或更新一条记录。
3. 把图片放到：
   - `public/images/restorations/<id>.webp`（长边 1600）
   - `public/images/thumbs/<id>.webp`（长边 720）
4. 提交 PR，说明参考图出处与提示词。

### C. 三维复原（手工或程序化建模）

三维与二维同等收录：只交渲染图也算完成；多一份 glb，就能在图册里「旋转查看」。

1. 先开 [提交三维复原 Issue](https://github.com/Ivan666jjj/yuanmingyuan-restoration/issues/new?template=submit-3d.yml)，写明景点、建模方式、软件版本与依据。
2. 准备交付物：
   - 主图 1 张（用于图册卡片）+ 正立面、侧面、斜俯视三视图，长边 ≥1600 的 WebP；
   - 可选 `glb`，**≤10MB**，单位米，原点在台基中心，导出 Y-up。Draco 可压到十分之一，但 `<model-viewer>` 的解码器默认从境外 CDN 取，国内可能加载失败，所以**优先不压缩**、靠精简几何控体积；确实要用 Draco 请在 Issue 里说明；
   - 材质说明：程序材质或公有领域/自制贴图，不使用来源不明的扫描图与网络贴图；
   - 逐项列出**哪些尺寸来自图档、哪些是估算**（柱网、开间、进深、檐高、装饰）。
3. 会 Git 的按 B 的方式提 PR，新增字段：

   ```json
   "model3d": {
     "glb": "/models/<id>.glb",
     "hero": "/images/3d/<id>-hero.webp",
     "views": ["/images/3d/<id>-w.webp", "/images/3d/<id>-s.webp"],
     "tool": "Blender 4.2 LTS · 手工建模",
     "basis": "柱网依样式雷图档 018-7；进深与檐高按图估算",
     "license": "CC BY-SA 4.0"
   }
   ```

   没有 glb 就省掉 `glb` 字段，只留 `hero`/`views`/`tool`/`basis`。
4. 不会 Git 的，把渲染图与 glb 附在 Issue 里，维护者替你合入并署名。

三维复原的读法：几何可检验，材质与彩画仍是推断。所以「像不像照片」不是标准，**柱网、开间、尺度能不能对回图档**才是。任何情况下都不得把模型宣称为测绘成果。

## 收录标准

- **尺寸**：母版 2940×1912 或长边 ≥1600。
- **必填**：`id`、`name`、`en`（英文名，英文页用）、`garden`、`category`、`x`/`y`（地图坐标）、`status`、`ref`、`model`、`prompt`。三维投稿另加可选 `model3d`（见上）。
- **参考图**：必须为公有领域或已获授权，并在 `ref` 中写明来源。
- **提示词**：可复现，包含形制参数而非泛泛描述。
- **声明**：不得宣称是照片、测绘或“原貌确证”；统一标注 AI 生成。

## 物理自查清单

- 喷泉从明确出水口喷出，呈抛物线落回水面；
- 水面低于池沿、驳岸；
- 单一光源，阴影方向一致；
- 石构件有厚度与接缝，无悬空、无穿模；
- 瓦垄逐片，屋面有坡度；
- 水中倒影为水平镜像；
- 人物、动物比例正常；
- 画面无文字、印章、水印（匾额需能说明出处）。

## 许可

提交即表示你同意：复原图以 **CC BY-SA 4.0** 发布，代码以 **MIT** 发布，并对参考图的版权状态负责。