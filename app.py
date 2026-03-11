"""
Gemini 2.5 Flash 多模態聊天室 — Streamlit Web GUI
支援圖片 (JPG/PNG)、PDF、純文字 (.txt) 檔案上傳與 AI 對話。
具備歷史對話紀錄瀏覽功能。
"""

import base64
import glob
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# ──────────────────────────────────────────────
# 頁面設定
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Gemini 2.5 Flash 聊天室",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 自訂 CSS 樣式
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── 全域字體 ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;600;700&display=swap');
html, body, [class*="st-"] {
    font-family: 'Noto Sans TC', 'Inter', sans-serif;
    color: #1e1e2e;
}

/* ── 主容器背景 ── */
.stApp {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 50%, #f0ebff 100%);
}

/* ── 側邊欄 ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f7ff 0%, #ede9fe 100%);
    border-right: 1px solid rgba(139, 92, 246, 0.12);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #2d2b55;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #3d3b65;
}

/* ── 標題樣式 ── */
h1 {
    background: linear-gradient(90deg, #7c3aed, #6366f1, #4f46e5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

/* ── 聊天訊息 ── */
div[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(139, 92, 246, 0.1);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    backdrop-filter: blur(12px);
    transition: background 0.2s ease;
    color: #1e1e2e;
}
div[data-testid="stChatMessage"]:hover {
    background: rgba(255, 255, 255, 0.9);
    border-color: rgba(139, 92, 246, 0.2);
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] li {
    color: #1e1e2e !important;
}

/* ── 聊天輸入框 ── */
div[data-testid="stChatInput"] textarea {
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 12px !important;
    color: #1e1e2e !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(139, 92, 246, 0.6) !important;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.1) !important;
}

/* ── 按鈕 ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.4) !important;
}

/* ── 下載按鈕 ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(16, 185, 129, 0.4) !important;
}

/* ── 檔案上傳區 ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(139, 92, 246, 0.25) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(139, 92, 246, 0.5) !important;
}

/* ── Metric 卡片 ── */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(139, 92, 246, 0.1);
    border-radius: 12px;
    padding: 0.8rem;
}
div[data-testid="stMetric"] label {
    color: #5b5880 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #4f46e5 !important;
}

/* ── 檔案標籤 ── */
.file-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(99, 102, 241, 0.1));
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.82rem;
    color: #5b21b6;
    margin-bottom: 6px;
}

/* ── 歡迎卡片 ── */
.welcome-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(237, 233, 254, 0.6));
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.08);
}
.welcome-card h2 {
    color: #4f46e5;
    margin-bottom: 0.5rem;
}
.welcome-card p {
    color: #5b5880;
    font-size: 0.95rem;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-top: 1.5rem;
}
.feature-item {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(139, 92, 246, 0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.feature-item .icon {
    font-size: 1.8rem;
    margin-bottom: 0.3rem;
}
.feature-item .label {
    color: #3d3b65;
    font-size: 0.85rem;
}

/* ── 歷史紀錄卡片 ── */
.history-card {
    background: rgba(255, 255, 255, 0.5);
    border: 1px solid rgba(139, 92, 246, 0.1);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    transition: all 0.2s ease;
    cursor: pointer;
}
.history-card:hover {
    background: rgba(255, 255, 255, 0.8);
    border-color: rgba(139, 92, 246, 0.3);
}
.history-card .history-time {
    color: #6b6890;
    font-size: 0.75rem;
}
.history-card .history-preview {
    color: #2d2b55;
    font-size: 0.85rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 2px;
}
.history-card .history-count {
    color: #7c3aed;
    font-size: 0.72rem;
    margin-top: 2px;
}

/* ── 歷史檢視標題 ── */
.history-viewer-header {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.7), rgba(237, 233, 254, 0.5));
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
    text-align: center;
}
.history-viewer-header h3 {
    color: #4f46e5;
    margin: 0;
    font-size: 1rem;
}
.history-viewer-header .sub {
    color: #6b6890;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LLM & 工具函式
# ──────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt"}


@st.cache_resource
def create_llm() -> ChatGoogleGenerativeAI:
    """建立 Gemini 2.5 Flash LLM（快取避免重複建立）。"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ 請在 `.env` 檔案中設定 `GEMINI_API_KEY`")
        st.stop()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
    )


def detect_file_type(filename: str) -> str | None:
    """根據副檔名判斷檔案類型。"""
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in PDF_EXTENSIONS:
        return "pdf"
    elif ext in TEXT_EXTENSIONS:
        return "text"
    return None


def load_pdf_text(file_bytes: bytes, filename: str) -> str:
    """使用 PyPDFLoader 讀取 PDF 內容。"""
    from langchain_community.document_loaders import PyPDFLoader

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        return "\n\n".join(
            f"[第 {i + 1} 頁]\n{page.page_content}" for i, page in enumerate(pages)
        )
    finally:
        os.unlink(tmp_path)


def build_message_with_file(
    file_type: str, file_bytes: bytes, filename: str, prompt: str
) -> tuple[HumanMessage, str]:
    """根據檔案類型組合 HumanMessage，回傳 (message, 紀錄描述文字)。"""
    import mimetypes

    default_prompts = {
        "image": "請描述這張圖片的內容。",
        "pdf": "請摘要這份 PDF 文件的內容。",
        "text": "請分析以下文字內容。",
    }
    if not prompt:
        prompt = default_prompts.get(file_type, "請分析此檔案。")

    if file_type == "image":
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = "image/jpeg"
        b64 = base64.standard_b64encode(file_bytes).decode("utf-8")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
        ]
        return HumanMessage(content=content), f"[圖片: {filename}] {prompt}"

    elif file_type == "pdf":
        pdf_text = load_pdf_text(file_bytes, filename)
        full_prompt = f"{prompt}\n\n---\n以下是 PDF 文件內容：\n{pdf_text}"
        return HumanMessage(content=[{"type": "text", "text": full_prompt}]), f"[PDF: {filename}] {prompt}"

    elif file_type == "text":
        text_content = file_bytes.decode("utf-8", errors="replace")
        full_prompt = f"{prompt}\n\n---\n以下是文字檔內容：\n{text_content}"
        return HumanMessage(content=[{"type": "text", "text": full_prompt}]), f"[TXT: {filename}] {prompt}"

    raise ValueError(f"不支援的檔案類型: {file_type}")


def sanitize_text(text: str) -> str:
    """移除 surrogate 字元。"""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def export_records_json(records: list[dict]) -> str:
    """將對話紀錄匯出為 JSON 字串。"""
    clean = []
    for r in records:
        entry = {
            "timestamp": r["timestamp"],
            "role": r["role"],
            "content": sanitize_text(r["content"]),
        }
        if "file" in r:
            entry["file"] = r["file"]
        clean.append(entry)
    return json.dumps(clean, ensure_ascii=False, indent=2)


def save_records_to_file(records: list[dict]) -> str:
    """將對話紀錄存成 JSON 檔案，回傳檔名。"""
    filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(export_records_json(records))
    return filename


def get_history_files() -> list[dict]:
    """掃描目前目錄下的歷史對話 JSON 檔案，回傳摘要列表。"""
    files = sorted(glob.glob("chat_*.json"), reverse=True)
    history_list = []
    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data:
                continue
            # 解析檔名中的時間
            basename = Path(filepath).stem  # e.g. "chat_20260311_095427"
            time_str = basename.replace("chat_", "")
            try:
                dt = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
                display_time = dt.strftime("%Y/%m/%d %H:%M:%S")
            except ValueError:
                display_time = time_str

            # 取得對話預覽（第一則使用者訊息）
            preview = ""
            for msg in data:
                if msg.get("role") == "user":
                    preview = msg.get("content", "")[:50]
                    break
            if not preview:
                preview = data[0].get("content", "")[:50]

            msg_count = len(data)
            history_list.append({
                "filepath": filepath,
                "display_time": display_time,
                "preview": preview,
                "msg_count": msg_count,
                "data": data,
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return history_list


# ──────────────────────────────────────────────
# Session State 初始化
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []
if "records" not in st.session_state:
    st.session_state.records = []
if "pending_file" not in st.session_state:
    st.session_state.pending_file = None
if "viewing_history" not in st.session_state:
    st.session_state.viewing_history = None  # 正在檢視的歷史紀錄


# ──────────────────────────────────────────────
# 側邊欄
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ✨ Gemini 聊天室")
    st.caption("由 Gemini 2.5 Flash + LangChain 驅動")

    st.divider()

    # ── 檔案上傳 ──
    st.markdown("### 📎 上傳檔案")
    uploaded_file = st.file_uploader(
        "拖放或選擇檔案",
        type=["jpg", "jpeg", "png", "pdf", "txt"],
        label_visibility="collapsed",
        key="file_uploader",
    )
    if uploaded_file:
        file_type = detect_file_type(uploaded_file.name)
        if file_type:
            st.session_state.pending_file = {
                "name": uploaded_file.name,
                "type": file_type,
                "bytes": uploaded_file.getvalue(),
            }
            type_icons = {"image": "🖼️", "pdf": "📄", "text": "📝"}
            st.success(f"{type_icons.get(file_type, '📎')} 已載入：{uploaded_file.name}")
            st.caption("在下方輸入框輸入問題，即可針對此檔案提問")
        else:
            st.error("不支援的檔案格式")

    st.divider()

    # ── 對話統計 ──
    st.markdown("### 📊 對話統計")
    user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    ai_msgs = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
    col1, col2 = st.columns(2)
    col1.metric("使用者", user_msgs)
    col2.metric("AI 回覆", ai_msgs)

    st.divider()

    # ── 對話管理 ──
    st.markdown("### 💾 對話管理")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.session_state.records:
            st.download_button(
                "⬇️ 匯出紀錄",
                data=export_records_json(st.session_state.records),
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
    with col_b:
        if st.button("🗑️ 清除對話", use_container_width=True):
            if st.session_state.records:
                save_records_to_file(st.session_state.records)
            st.session_state.messages = []
            st.session_state.langchain_history = []
            st.session_state.records = []
            st.session_state.pending_file = None
            st.session_state.viewing_history = None
            st.rerun()

    st.divider()

    # ── 歷史對話紀錄 ──
    st.markdown("### 📜 歷史對話紀錄")
    history_files = get_history_files()

    if not history_files:
        st.caption("尚無歷史對話紀錄")
    else:
        st.caption(f"共 {len(history_files)} 筆紀錄")

        # 返回當前對話按鈕
        if st.session_state.viewing_history is not None:
            if st.button("🔙 返回當前對話", use_container_width=True):
                st.session_state.viewing_history = None
                st.rerun()

        for i, hist in enumerate(history_files):
            st.markdown(
                f"""<div class="history-card">
                    <div class="history-time">🕐 {hist['display_time']}</div>
                    <div class="history-preview">{hist['preview']}</div>
                    <div class="history-count">💬 {hist['msg_count']} 則訊息</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(
                f"📖 檢視此對話",
                key=f"hist_{i}",
                use_container_width=True,
            ):
                st.session_state.viewing_history = hist
                st.rerun()


# ──────────────────────────────────────────────
# 主聊天區
# ──────────────────────────────────────────────

# ═══ 模式一：檢視歷史紀錄 ═══
if st.session_state.viewing_history is not None:
    hist = st.session_state.viewing_history
    st.markdown("# 📜 歷史對話紀錄")
    st.markdown(
        f"""<div class="history-viewer-header">
            <h3>🕐 {hist['display_time']}</h3>
            <div class="sub">📁 {hist['filepath']}　｜　💬 {hist['msg_count']} 則訊息</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # 下載此歷史紀錄
    st.download_button(
        "⬇️ 下載此對話紀錄",
        data=json.dumps(hist["data"], ensure_ascii=False, indent=2),
        file_name=hist["filepath"],
        mime="application/json",
    )

    # 顯示歷史對話內容
    for msg in hist["data"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")

        # 將 JSON 中的 "ai" 角色轉為 streamlit 的 "assistant"
        display_role = "assistant" if role == "ai" else role

        with st.chat_message(display_role):
            # 顯示時間戳記
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_display = dt.strftime("%H:%M:%S")
                except ValueError:
                    time_display = timestamp
                st.caption(f"🕐 {time_display}")

            # 顯示檔案標籤
            if "file" in msg:
                file_info = msg["file"]
                file_path = file_info.get("path", "")
                file_type = file_info.get("type", "")
                type_icons = {"image": "🖼️", "pdf": "📄", "text": "📝"}
                icon = type_icons.get(file_type, "📎")
                st.markdown(
                    f'<div class="file-badge">{icon} {file_path}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(content)

    st.info("💡 這是歷史對話紀錄的唯讀檢視。點選側邊欄「🔙 返回當前對話」繼續聊天。")

# ═══ 模式二：正常聊天 ═══
else:
    st.markdown("# ✨ Gemini 2.5 Flash 多模態聊天室")

    # 歡迎畫面
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <h2>👋 歡迎使用 Gemini 聊天室</h2>
            <p>支援多輪對話、檔案上傳分析、對話紀錄匯出與歷史瀏覽</p>
            <div class="feature-grid">
                <div class="feature-item">
                    <div class="icon">💬</div>
                    <div class="label">智慧對話</div>
                </div>
                <div class="feature-item">
                    <div class="icon">📎</div>
                    <div class="label">多模態檔案</div>
                </div>
                <div class="feature-item">
                    <div class="icon">💾</div>
                    <div class="label">紀錄匯出</div>
                </div>
                <div class="feature-item">
                    <div class="icon">📜</div>
                    <div class="label">歷史瀏覽</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 顯示目前的聊天訊息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "file" in msg:
                type_icons = {"image": "🖼️", "pdf": "📄", "text": "📝"}
                icon = type_icons.get(msg["file"]["type"], "📎")
                st.markdown(
                    f'<div class="file-badge">{icon} {msg["file"]["name"]}</div>',
                    unsafe_allow_html=True,
                )
                if msg["file"]["type"] == "image" and "image_bytes" in msg["file"]:
                    st.image(msg["file"]["image_bytes"], width=300)
            st.markdown(msg["content"])

    # 聊天輸入
    if prompt := st.chat_input("輸入訊息...（側邊欄可上傳檔案）"):
        llm = create_llm()
        timestamp = datetime.now().isoformat()
        pending = st.session_state.pending_file

        if pending:
            # ── 有檔案附帶的訊息 ──
            file_type = pending["type"]
            file_bytes = pending["bytes"]
            filename = pending["name"]

            display_msg = {
                "role": "user",
                "content": prompt,
                "file": {"name": filename, "type": file_type},
            }
            if file_type == "image":
                display_msg["file"]["image_bytes"] = file_bytes
            st.session_state.messages.append(display_msg)

            with st.chat_message("user"):
                type_icons = {"image": "🖼️", "pdf": "📄", "text": "📝"}
                icon = type_icons.get(file_type, "📎")
                st.markdown(
                    f'<div class="file-badge">{icon} {filename}</div>',
                    unsafe_allow_html=True,
                )
                if file_type == "image":
                    st.image(file_bytes, width=300)
                st.markdown(prompt)

            human_msg, record_text = build_message_with_file(
                file_type, file_bytes, filename, prompt
            )
            st.session_state.langchain_history.append(human_msg)
            st.session_state.records.append({
                "timestamp": timestamp,
                "role": "user",
                "content": record_text,
                "file": {"path": filename, "type": file_type},
            })

            st.session_state.pending_file = None

        else:
            # ── 純文字訊息 ──
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            human_msg = HumanMessage(content=prompt)
            st.session_state.langchain_history.append(human_msg)
            st.session_state.records.append({
                "timestamp": timestamp,
                "role": "user",
                "content": prompt,
            })

        # ── 呼叫 AI ──
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    response = llm.invoke(st.session_state.langchain_history)
                    ai_text = response.content
                except Exception as e:
                    ai_text = f"❌ API 呼叫失敗：{e}"
                st.markdown(ai_text)

        # 記錄 AI 回覆
        timestamp = datetime.now().isoformat()
        ai_msg = AIMessage(content=ai_text)
        st.session_state.langchain_history.append(ai_msg)
        st.session_state.messages.append({"role": "assistant", "content": ai_text})
        st.session_state.records.append({
            "timestamp": timestamp,
            "role": "ai",
            "content": ai_text,
        })

        # 即時儲存 JSON
        save_records_to_file(st.session_state.records)
