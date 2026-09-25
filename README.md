# 📰 本地新闻问答 RAG 系统

一个适合应届生练手的**最小知识库（Minimal Knowledge Base）项目**。

基于中国新闻网滚动新闻数据，使用 **ChromaDB + sentence-transformers + OpenAI 兼容接口 + Gradio** 实现本地向量检索与问答。大模型只根据召回的新闻内容作答，找不到相关信息时会明确回答"没有"。

---

## 🎯 项目亮点

- **端到端 RAG 流程**：数据预处理 → 向量入库 → Top-K 召回 → 大模型总结 → 网页聊天界面
- **本地向量库**：使用 ChromaDB 本地持久化，无需安装额外数据库
- **轻量 Embedding**：使用 `all-MiniLM-L6-v2`，普通电脑即可运行
- **OpenAI 兼容接口**：只需配置 API key 和 base_url，可接入任意兼容接口
- **引用可追溯**：回答下方可查看引用的原始新闻标题、时间和链接
- **安全合规**：敏感配置从 `.env` 读取，不会提交到 Git 仓库

---

## 🛠 技术栈

| 模块 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 向量数据库 | ChromaDB（本地持久化） |
| Embedding 模型 | sentence-transformers / all-MiniLM-L6-v2 |
| 大模型调用 | OpenAI 兼容接口 |
| 前端界面 | Gradio |
| 配置管理 | python-dotenv |

---

## 📁 项目结构

```
.
├── news_rag.py          # 主程序：RAG 全流程代码
├── news.json            # 爬取的新闻数据（需自行准备，不提交到 Git）
├── chroma_db/           # ChromaDB 本地向量库（自动生成，不提交到 Git）
├── .env                 # 环境变量：API key 等（不提交到 Git）
├── .env.example         # 环境变量示例模板
├── .gitignore           # Git 忽略规则
└── README.md            # 本文件
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/你的用户名/news-rag.git
cd news-rag
```

### 2. 安装依赖

```bash
pip install chromadb sentence-transformers openai gradio python-dotenv
```

### 3. 准备新闻数据

将你爬取好的新闻 JSON 文件命名为 `news.json`，放在项目根目录。

JSON 格式要求：每条新闻包含 `title`、`category`、`time`、`url`、`content` 字段，例如：

```json
[
  {
    "title": "示例新闻标题",
    "category": "科技",
    "time": "2026-09-25",
    "url": "https://www.chinanews.com.cn/...",
    "content": "新闻正文内容..."
  }
]
```

> 如果你还没有爬虫，可以先用中国新闻网滚动新闻页自行抓取：`https://www.chinanews.com.cn/scroll-news/news1.html`

### 4. 配置环境变量

复制示例文件：

```bash
cp .env.example .env
```

编辑 `.env`，填入你的真实配置：

```env
OPENAI_API_KEY=sk-你的key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4o-mini
```

- `OPENAI_API_KEY`：你的 API key
- `OPENAI_BASE_URL`：OpenAI 兼容接口地址，第三方中转就改成中转地址
- `OPENAI_MODEL_NAME`：模型名，按你的接口支持的填写

### 5. 运行项目

```bash
python news_rag.py
```

首次运行会自动下载 Embedding 模型（约 90MB），之后本地可用。

启动成功后，浏览器访问：

```
http://127.0.0.1:7860
```

---

## 💬 使用示例

在网页输入框中提问：

- `今天有哪些科技相关的新闻？`
- `最近有什么体育新闻？`
- `关于经济的新闻有哪些？`

系统会：
1. 在本地向量库中召回最相关的 5 条新闻
2. 把召回结果喂给大模型，生成基于新闻内容的回答
3. 点击 **"显示引用来源"** 可查看答案引用的原始新闻标题、时间和链接

---

## 🔒 安全说明

- `.env` 文件包含 API key，**已加入 `.gitignore`，不会提交到 Git**
- 请勿在代码中硬编码 API key
- 如需分享项目，只提交 `.env.example`（占位值），不提交 `.env`

---

## ⚙️ 主要配置项

所有配置都在 `news_rag.py` 顶部的配置区，也可通过环境变量覆盖：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `NEWS_JSON_PATH` | `./news.json` | 新闻数据文件路径 |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB 本地存储目录 |
| `COLLECTION_NAME` | `news_db` | ChromaDB 集合名 |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding 模型名 |
| `TOP_K` | `5` | 每次召回的新闻条数 |
| `OPENAI_API_KEY` | 从 `.env` 读取 | API key |
| `OPENAI_BASE_URL` | 从 `.env` 读取 | OpenAI 兼容接口地址 |
| `LLM_MODEL_NAME` | 从 `.env` 读取 | 大模型名 |

---

## 📝 注意事项

1. **Embedding 模型中文效果**
   
   `all-MiniLM-L6-v2` 主要面向英文语料，中文语义召回效果一般。如果中文问答效果不理想，可在配置区替换为：
   
   ```python
   EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
   ```
   
   接口完全兼容，无需改动其他代码。

2. **向量库重复入库**
   
   程序会检测 `chroma_db/` 目录是否存在：
   - 已存在：直接加载现有向量库，**跳过入库**
   - 不存在：创建新库并批量插入新闻向量
   
   更新新闻数据后，如需重新入库，请删除 `chroma_db/` 目录再运行。

3. **大模型回答"
   
   如果召回的新闻中没有相关信息，系统会直接回复"当前新闻库中没有相关信息"，不会编造答案。

---

## 📚 适合学习/面试的知识点

通过这个项目可以理解和实践：

- RAG 的基本流程：检索 + 增强生成
- 文本块（Chunk）与元数据（Metadata）的设计
- 向量相似度检索与余弦相似度
- 本地向量数据库 ChromaDB 的使用
- sentence-transformers Embedding 模型
- OpenAI 兼容接口的调用方式
- Gradio 快速搭建 demo 界面
- 环境变量与 `.env` 管理敏感配置

---

## 📄 License

本项目仅用于学习交流。
