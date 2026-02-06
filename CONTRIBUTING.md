# 贡献指南

感谢您对《大模型数据工程》项目的关注！本文档提供了向项目贡献的指南。

## 📝 如何贡献

### 报告问题

如果您发现了错误、内容不准确或有改进建议：

1. 前往 [GitHub Issues](https://github.com/datascale-ai/data_engineering_book/issues)
2. 点击 "New Issue"
3. 提供详细信息：
   - 问题所在的章节和页面
   - 具体问题描述
   - 期望的正确内容（如适用）
   - 截图（如适用）

### 提交内容改进

我们欢迎以下类型的贡献：
- 修正错别字和语法错误
- 改进代码示例
- 添加新的图表或说明
- 更新过时的技术信息
- 优化文档结构

#### 贡献流程

1. **Fork 仓库**
   ```bash
   # 在 GitHub 页面点击 Fork 按钮
   ```

2. **克隆到本地**
   ```bash
   git clone https://github.com/YOUR_USERNAME/data_engineering_book.git
   cd data_engineering_book
   ```

3. **创建特性分支**
   ```bash
   git checkout -b improve/chapter-x-topic
   ```

4. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

5. **本地预览**
   ```bash
   mkdocs serve
   ```
   访问 http://127.0.0.1:8000 预览您的更改

6. **提交更改**
   ```bash
   git add .
   git commit -m "改进第X章：简要描述您的修改"
   git push origin improve/chapter-x-topic
   ```

7. **提交 Pull Request**
   - 前往您 Fork 的仓库页面
   - 点击 "Compare & pull request"
   - 填写 PR 描述，说明您做了什么改进
   - 等待维护者审核

## 📐 内容规范

### Markdown 格式

- 使用标准 Markdown 语法
- 标题层级：章节用 `#`，小节用 `##`，子小节用 `###`
- 代码块必须指定语言：
  ```markdown
  \```python
  print("Hello, World!")
  \```
  ```

### 代码示例

- Python 代码遵循 [PEP 8](https://pep8.org/) 规范
- Shell 脚本遵循 [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)
- 包含必要的注释说明
- 确保代码可运行（或明确标注为伪代码）

### 图片规范

- 图片存放在 `docs/images/第X章/` 目录
- 命名格式：`图X-Y_描述.png`
  - X: 章节号
  - Y: 图序号
  - 描述: 简短中文描述
- 示例：`图3-1_数据采集流程.png`
- 建议格式：PNG（图表）或 JPEG（照片）
- 建议分辨率：宽度 800-1200px

### 引用规范

- 引用论文或文章时提供链接
- 格式：`[论文标题](URL)`
- 示例：
  ```markdown
  根据 [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) 的研究...
  ```

## 🎯 内容质量标准

### 技术准确性
- 确保技术描述准确无误
- 引用权威来源
- 包含实际可运行的代码示例

### 可读性
- 使用通俗易懂的语言
- 避免过度使用术语（或提供解释）
- 使用场景化的例子帮助理解

### 完整性
- 代码示例包含必要的导入语句
- 提供完整的执行步骤
- 说明环境要求和依赖

## 🚫 不接受的贡献

- 广告或营销内容
- 与书籍主题无关的内容
- 未经验证的代码或错误信息
- 侵犯版权的内容

## ❓ 问题咨询

如果您对贡献流程有任何疑问：
- 提交 [GitHub Issue](https://github.com/datascale-ai/data_engineering_book/issues)
- 在您的 Pull Request 中提问

## 📄 许可证

通过向本项目贡献内容，您同意您的贡献将按照项目的 [MIT License](LICENSE) 进行许可。

---

再次感谢您的贡献！让我们一起打造更好的技术书籍！ 🎉
