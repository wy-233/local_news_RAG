# -*- coding: utf-8 -*-
"""
本地新闻问答 RAG 系统
=====================
数据源：中国新闻网滚动新闻（爬取好的 news.json）
流程：数据预处理 -> ChromaDB 向量入库 -> Top5 召回 -> 大模型总结 -> Gradio 问答界面

依赖安装（Python 3.10+，直接 pip install）：
    pip install chromadb sentence-transformers openai gradio python-dotenv

说明：
1. 第一次运行会自动下载 all-MiniLM-L6-v2 模型（约 90MB），需联网，之后本地可用。
2. 把爬好的 news.json 放在本文件同目录下，运行 python news_rag.py 即可启动。

【最小知识库（Minimal Knowledge Base）的设计思路】

RAG = Retrieval（检索） + Augmented Generation（增强生成）。
所谓"最小知识库"，核心就是把"你手里已有的数据"变成"大模型能查的资料"，
让大模型在回答问题时只参考你的数据，而不是用它的通用知识瞎编。

本文件是这个最小知识库的 5 个必要零件：
┌─────────────────────────────────────────────────────────────────┐
│  1. 数据预处理  │  把原始数据切成统一格式的"文档块"（Document）     │
│  2. 向量入库    │  把文档块转成向量，存进本地 ChromaDB             │
│  3. 召回函数    │  把用户问题也转成向量，找最相似的 Top-K 个文档块 │
│  4. 大模型总结  │  把召回的文档块喂给 LLM，让它只根据资料回答      │
│  5. Gradio 界面 │  给用户一个可交互的网页聊天入口                  │
└─────────────────────────────────────────────────────────────────┘

想换成你自己的数据源（比如论文、公司文档、聊天记录），通常只需要改：
- NEWS_JSON_PATH：换成你的数据文件
- preprocess_news()：改成解析你的数据格式
- EMBED_MODEL_NAME / LLM_MODEL_NAME：按需要换模型
其他模块基本不用动。
"""

import os
import json

# 加载同级目录下的 .env 文件，把里面的 KEY=VALUE 注入到环境变量中。
# 这样可以把敏感配置（API key）放在 .env 里，既方便本地开发，又不会提交到 Git。
try:
    from dotenv import load_dotenv
    load_dotenv()  # 默认读取脚本同目录的 .env 文件
except ImportError:
    # 如果用户没装 python-dotenv，也不影响运行，只是不支持 .env 文件
    pass

# ============================ 配置区（按需修改） ============================
# 把配置放在文件最开头，是为了"一次修改，全局生效"。
# 面试/写简历时可以说：配置与业务逻辑解耦，便于不同环境复用。

# 爬好的新闻 JSON 文件路径（与本脚本同目录）
# os.path.dirname(os.path.abspath(__file__)) 表示"本脚本所在目录"，
# 这样无论你从哪个目录运行 python xxx/news_rag.py，都能找到同目录下的 news.json。
NEWS_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news.json")

# ChromaDB 本地持久化目录 & 集合名
# "./chroma_db" 会在当前目录下创建一个文件夹，里面是 Chroma 的持久化数据。
# Collection（集合）相当于关系型数据库里的"表"，用来存放同一类向量。
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "news_db"

# Embedding 模型（sentence-transformers，本地运行）
# Embedding 模型负责把"文本"变成"高维向量"，向量之间的距离就代表了语义相似度。
# all-MiniLM-L6-v2 是英文为主的轻量模型，中文效果一般；想中文更好可换成
# "paraphrase-multilingual-MiniLM-L12-v2"，接口完全兼容。
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# 召回条数：用户提问后，从向量库中找出语义最相关的 5 条新闻片段喂给大模型。
# 数字越大，大模型看到的上下文越多；数字越小，噪音越少。5 是常见起点。
TOP_K = 5

# 大模型配置（OpenAI 兼容接口，改成你自己的）
# ⚠️ 安全提示：key 从环境变量读取，不会提交到 Git 仓库。
# 推荐做法：在脚本同目录下创建 .env 文件，写入：
#   OPENAI_API_KEY=sk-你的key
#   OPENAI_BASE_URL=https://aihub.top/v1
#   OPENAI_MODEL_NAME=gpt-5.6-sol
# 程序启动时会自动读取。也可以手动设置环境变量：
#   Windows PowerShell: $env:OPENAI_API_KEY="sk-你的key"
#   Windows CMD:        set OPENAI_API_KEY=sk-你的key
#   Linux/macOS:        export OPENAI_API_KEY=sk-你的key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://aihub.top/v1")   # 第三方中转就改成中转地址
LLM_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.6-sol")            # 模型名，按你的接口支持的填

# Gradio 服务配置
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860

# ============================ 模块1：数据预处理 ============================
# 知识库的第一件事：把"原始数据"整理成"计算机能理解的文档块"。
# 每个文档块（document / chunk）就是后面被向量化、被检索的最小单位。

def preprocess_news(json_path):
    """
    读取爬好的新闻 JSON，拼成统一格式的文本块。
    输入 JSON 每条新闻字段：title, category, time, url, content
    返回：
        texts     —— 文本块列表
        metadatas —— 元数据字典列表（title, category, time, url）

    最小知识库里的"文本块"为什么要拼成固定格式？
    - 向量化模型看到的是纯文本，格式统一后语义更清晰；
    - 元数据不进入向量，但会随向量一起存/一起查回来，用来展示引用来源。
    """
    # 检查文件是否存在：友好的错误处理，让应届生朋友一眼知道是不是 news.json 放错位置。
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到新闻数据文件：{json_path}，请确认 news.json 放在脚本同目录下。")

    # 读取 JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_news = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 文件解析失败，请检查文件格式：{e}")

    # 兼容两种常见结构：直接是列表，或 {"data": [...]} 包了一层
    # 做兼容是因为爬虫保存 JSON 的习惯不同，提前处理能减少后面出错。
    if isinstance(raw_news, dict):
        raw_news = raw_news.get("data", [])
    if not isinstance(raw_news, list):
        raise ValueError("新闻 JSON 结构不正确：应为新闻列表，或包含 data 列表字段。")

    texts = []
    metadatas = []

    for item in raw_news:
        # 缺字段时给默认值，避免拼接报错。
        # 这是做最小知识库的常用技巧：对上游数据做"防御式处理"。
        title = str(item.get("title", "")).strip()
        category = str(item.get("category", "未分类")).strip()
        pub_time = str(item.get("time", "未知")).strip()
        url = str(item.get("url", "")).strip()
        content = str(item.get("content", "")).strip()

        # 正文只取前 500 字，控制文本块长度。
        # 原因：
        # 1. 向量化模型对超长文本的表征效果会下降；
        # 2. 召回后给 LLM 的上下文太长会消耗更多 token / 稀释重点。
        # 更专业的做法是用滑动窗口切分，但应届生练手项目"取前 N 字"够用了。
        content_cut = content[:500]

        # 按要求的格式拼成一个文本块。
        # "\n" 是换行，把多个字段隔开，向量化模型能更容易区分不同字段。
        text_block = (
            f"【标题】{title}\n"
            f"【分类】{category}\n"
            f"【时间】{pub_time}\n"
            f"【正文】{content_cut}"
        )

        texts.append(text_block)
        # 元数据（入库后随向量一起返回，用于展示引用来源）
        # 注意：元数据不参与向量相似度计算，只是"挂在"每条向量旁边的附加信息。
        metadatas.append({
            "title": title,
            "category": category,
            "time": pub_time,
            "url": url,
        })

    print(f"[预处理] 共读取 {len(texts)} 条新闻。")
    return texts, metadatas


# ============================ 模块2：向量入库 ============================
# 知识库的核心：把"文本"变成"向量"，并持久化保存。
# ChromaDB 是本地向量数据库，不需要额外安装 MySQL/PostgreSQL，适合最小知识库。

def _build_embedding_function():
    """
    构造一个基于 sentence-transformers 的 Chroma 嵌入函数，
    让 Chroma 在入库和查询时自动调用本地 all-MiniLM-L6-v2 模型。

    为什么要自定义 EmbeddingFunction？
    - Chroma 默认可能会自动下载一个 ONNX 模型，体积较大且不可控；
    - 自己包一层 sentence-transformers，可以明确指定模型、做归一化，且接口零改动。
    """
    from sentence_transformers import SentenceTransformer

    # 不同版本 chromadb 的 EmbeddingFunction 位置略有差异，做兼容导入。
    # 这是工程里的小技巧：优先用新版导入路径，失败再回退旧版。
    try:
        from chromadb import EmbeddingFunction
    except ImportError:
        from chromadb.api.types import EmbeddingFunction

    class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
        """自定义嵌入函数：把文本列表转成向量列表。"""

        def __init__(self, model_name):
            # 首次运行会自动下载模型到本地缓存目录（默认 ~/.cache/torch/sentence_transformers）。
            self.model = SentenceTransformer(model_name)

        def __call__(self, input):
            # normalize_embeddings=True 表示把向量长度归一化到 1，
            # 这样两个向量之间的"距离"就等价于"余弦相似度"，是语义检索最常用的度量方式。
            vectors = self.model.encode(
                list(input),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.tolist()

    return SentenceTransformerEmbeddingFunction(EMBED_MODEL_NAME)


def get_or_create_db(texts=None, metadatas=None):
    """
    创建或加载本地持久化 Chroma 库。
    - chroma_db 文件夹已存在：直接加载现有库，跳过入库（避免重复插入）；
    - 不存在：批量插入预处理好的文本块和元数据。
    返回 Chroma Collection 对象。

    为什么"避免重复插入"很重要？
    - 重复插入会让向量库里有大量重复文档，召回时会浪费 Top-K 位置；
    - 对本地持久化库来说，每次启动都重建索引也会浪费时间。
    """
    import chromadb

    # 初始化嵌入函数（会下载/加载模型）。
    embedding_function = _build_embedding_function()

    # 文件夹已存在 -> 默认直接加载，不重复入库。
    if os.path.isdir(CHROMA_DB_PATH):
        print(f"[入库] 检测到已有向量库目录 {CHROMA_DB_PATH}，直接加载，跳过入库。")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )
        # 兜底：目录在但是空的（比如上次入库中断），补一次入库。
        # 这是一个工程鲁棒性处理，避免"目录存在但库里没数据"导致界面一问三不知。
        if collection.count() == 0 and texts:
            print("[入库] 现有库为空，重新执行入库。")
            _add_documents(collection, texts, metadatas)
        return collection

    # 文件夹不存在 -> 新建库并批量入库。
    print(f"[入库] 未发现向量库，创建新库：{CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    _add_documents(collection, texts, metadatas)
    return collection


def _add_documents(collection, texts, metadatas):
    """
    把文本块批量插入 Chroma，每条带 id、文本和元数据。

    Chroma 的 collection.add() 接收三个关键参数：
    - ids: 唯一标识，用于后续删除/更新；
    - documents: 原始文本，Chroma 会调用 embedding_function 自动转成向量；
    - metadatas: 附加信息，查询时会随结果一起返回。
    """
    if not texts:
        print("[入库] 没有可入库的新闻数据。")
        return

    # 给每条新闻生成唯一 id，简单场景用 news_0, news_1, ... 即可。
    ids = [f"news_{i}" for i in range(len(texts))]

    # 批量插入（chroma 会通过自定义嵌入函数自动转向量）。
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"[入库] 完成，共插入 {collection.count()} 条新闻向量。")


# ============================ 模块3：召回函数 ============================
# 用户提问时，我们不能把全库文本都塞给大模型（太长、太贵、太慢）。
# 召回（Retrieval）的作用：只挑"和用户问题最相关"的那几条。

def retrieve_news(query, collection, top_k=TOP_K):
    """
    根据用户问题在向量库中召回最相关的新闻片段。
    返回：
        results —— [{"text": 文本块, "metadata": 元数据}, ...]，最多 top_k 条

    召回的本质是"语义最近邻搜索"：
    1. 把 query 也转成向量（用同一个 Embedding 模型）；
    2. 在 Chroma 里找到与 query 向量最接近的 top_k 个文档向量；
    3. 返回这些文档的原文和元数据。
    """
    if not query or not query.strip():
        return []

    # query_texts 会自动调用嵌入函数把问题转向量，再做相似度检索。
    try:
        query_result = collection.query(
            query_texts=[query.strip()],
            n_results=top_k,
        )
    except Exception as e:
        print(f"[召回] 向量检索出错：{e}")
        return []

    results = []
    # query 返回的是嵌套列表（按 query 分组），这里只有一个 query，所以取 [0]。
    docs = query_result.get("documents", [[]])[0]
    metas = query_result.get("metadatas", [[]])[0]

    for doc, meta in zip(docs, metas):
        results.append({
            "text": doc,
            "metadata": meta or {},
        })

    print(f"[召回] 问题「{query}」召回 {len(results)} 条新闻。")
    return results


# ============================ 模块4：大模型总结 ============================
# 这就是 RAG 里"增强生成"的部分：大模型不依赖自己的记忆，
# 而是依赖我们刚刚召回出来的几条新闻片段来回答。

def build_prompt(query, retrieved_results):
    """
    按指定模板拼 system prompt：用户问题 + 召回的新闻内容。

    prompt 设计的核心原则：
    - 明确告诉模型"只能根据下面的新闻片段回答"，防止它使用外部知识瞎编；
    - 明确告诉模型"找不到就直说没有"，这是最小知识库防幻觉的关键；
    - 把召回的新闻编号列出，让模型知道有几条证据。
    """
    # 把召回的新闻逐条编号拼接
    news_context = "\n\n".join(
        f"【新闻片段 {i + 1}】\n{item['text']}"
        for i, item in enumerate(retrieved_results)
    )

    prompt = (
        "你是一个新闻问答助手。请只根据下面提供的新闻片段回答用户问题，不要使用外部知识。"
        "如果新闻里没有相关信息，直接说没有。\n\n"
        f"新闻片段：\n{news_context}\n\n"
        f"用户问题：{query}\n\n"
        "请用简洁的中文回答："
    )
    return prompt


def generate_answer(query, retrieved_results):
    """
    调用 OpenAI 兼容接口，让大模型只根据召回新闻回答问题。
    返回大模型生成的回答文本；失败时返回友好的错误提示。

    为什么这里要用 OpenAI 兼容接口？
    - 国内很多中转平台都提供 OpenAI 格式的 API；
    - openai 包的 OpenAI 客户端可以只改 base_url 就切换到不同服务商；
    - 对练手项目来说，这是接入大模型最轻量的方式。
    """
    # 一条都没召回：直接按要求回复，不浪费一次 API 调用。
    if not retrieved_results:
        return "当前新闻库中没有相关信息。"

    # 延迟导入，避免没用到大模型时也强依赖 openai 包。
    # 这叫"懒加载"，启动速度更快，依赖也更松耦合。
    try:
        from openai import OpenAI
    except ImportError:
        return "未安装 openai 库，请先执行：pip install openai"

    # 检查 key 是否配置：防止用户忘了设置环境变量就直接运行。
    if not OPENAI_API_KEY:
        return "尚未配置 OPENAI_API_KEY 环境变量，请先设置后再运行。"

    prompt = build_prompt(query, retrieved_results)

    # 初始化 OpenAI 兼容客户端（base_url 可指向任意兼容接口）。
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=30,
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                # system prompt 放"约束条件 + 参考资料"，user 只放问题本身。
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.3,   # 低温度让回答更稳定、更忠实于资料。
        )
        # 兼容标准 OpenAI 响应，以及部分中转接口直接返回字符串/字典的情况。
        if isinstance(response, str):
            answer = response.strip()
        elif isinstance(response, dict):
            choices = response.get("choices") or []
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            answer = str(content).strip()
        else:
            content = response.choices[0].message.content or ""
            answer = content.strip()

        if not answer:
            raise ValueError("大模型接口返回了空内容")
        return answer
    except Exception as e:
        # 认证失败、网络不通、接口报错等统一给友好提示。
        print(f"[大模型] 调用失败：{e}")
        return f"抱歉，大模型服务调用失败（{type(e).__name__}），请检查 API key、base_url 和网络后重试。"


# ============================ 模块5：Gradio 界面 ============================
# 最小知识库需要一个"门面"。Gradio 用几行代码就能搭出网页聊天界面，
# 适合快速验证项目效果，也适合写到简历里展示 Demo。

def format_sources(retrieved_results):
    """
    把召回新闻的标题、时间、链接整理成可展示的 Markdown 文本。

    显示引用来源有两个好处：
    1. 用户可追溯答案出处，增强可信度；
    2. 面试官/阅卷人能看到你确实做了 RAG，而不是直接调大模型。
    """
    if not retrieved_results:
        return "本次回答没有引用任何新闻来源。"

    lines = ["**本次回答引用的新闻来源：**"]
    for i, item in enumerate(retrieved_results):
        meta = item.get("metadata", {})
        title = meta.get("title", "无标题")
        pub_time = meta.get("time", "未知时间")
        url = meta.get("url", "")
        if url:
            lines.append(f"{i + 1}. {title}（{pub_time}）\n   [查看原文]({url})")
        else:
            lines.append(f"{i + 1}. {title}（{pub_time}）")
    return "\n".join(lines)


def launch_gradio(collection):
    """启动 Gradio 聊天界面。"""
    import gradio as gr

    def respond(message, chat_history):
        """
        点击发送 / 回车：先召回，再让大模型总结，并更新对话与引用来源。

        Gradio 的事件函数返回几个值，就会更新几个组件。
        这里返回 4 个值：
        1. 清空输入框；
        2. 更新聊天记录；
        3. 提示"点击下方按钮查看引用来源"；
        4. 把召回结果暂存到 State 里。
        """
        message = (message or "").strip()
        if not message:
            return "", chat_history, gr.update(value="请先提问，再查看引用来源。"), []

        # 第一步：召回 Top5
        retrieved_results = retrieve_news(message, collection)
        # 第二步：大模型总结
        answer = generate_answer(message, retrieved_results)

        # 更新聊天记录（messages 格式：每条记录带 role 字段）
        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        # 输入框清空；引用来源暂存到状态里，等用户点按钮再展示。
        return "", chat_history, gr.update(value="点击下方按钮查看本次回答的引用来源。"), retrieved_results

    def show_sources(retrieved_results):
        """显示本次回答引用的新闻来源。"""
        return format_sources(retrieved_results or [])

    with gr.Blocks(title="本地新闻问答 RAG") as demo:
        gr.Markdown("# 📰 本地新闻问答系统\n基于中国新闻网滚动新闻的 RAG 问答，回答仅来自本地新闻库。")

        chatbot = gr.Chatbot(label="新闻问答", height=450)

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="请输入问题，例如：今天有哪些科技相关的新闻？",
                scale=8,
                show_label=False,
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)

        source_btn = gr.Button("显示引用来源")
        source_box = gr.Markdown(value="提问后可在此查看引用来源。")

        # 暂存最近一次召回结果。State 不会显示在界面上，但可以在事件之间传递数据。
        last_sources = gr.State([])

        # 绑定事件：按钮点击 & 输入框回车
        send_btn.click(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, source_box, last_sources],
        )
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, source_box, last_sources],
        )
        source_btn.click(
            show_sources,
            inputs=[last_sources],
            outputs=[source_box],
        )

    # 启动网页界面
    demo.launch(server_name=SERVER_HOST, server_port=SERVER_PORT)


# ============================ 主流程 ============================
# main() 只做三件事：预处理、入库、启动界面。
# 这就是整个最小知识库的"启动开关"。

def main():
    """串联整个流程：加载数据 -> 入库/加载库 -> 启动 Gradio 界面。"""
    print("=" * 50)
    print("本地新闻问答 RAG 系统启动中...")
    print("=" * 50)

    # 1. 数据预处理：把新闻 JSON 变成文本块 + 元数据。
    texts, metadatas = preprocess_news(NEWS_JSON_PATH)

    # 2. 向量入库（已存在则直接加载）：把文本块变成向量，存到本地 Chroma。
    collection = get_or_create_db(texts, metadatas)

    # 3. 启动 Gradio 问答界面（召回 + 大模型总结在界面交互时触发）。
    print(f"[启动] Gradio 界面地址：http://{SERVER_HOST}:{SERVER_PORT}")
    launch_gradio(collection)


if __name__ == "__main__":
    # 当直接运行 python news_rag.py 时执行 main()；
    # 当作为模块被 import 时不会自动运行，便于测试/复用。
    main()
