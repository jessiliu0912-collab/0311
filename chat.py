"""
Gemini 2.5 Flash 多模態聊天程式
支援圖片 (JPG/PNG)、PDF、純文字 (.txt) 檔案輸入。
具備對話記憶與 JSON 持久化功能。
"""

import base64
import json
import mimetypes
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

# 修正 Windows 終端中文編碼問題
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

# 支援的檔案類型
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {".txt"}


def create_llm() -> ChatGoogleGenerativeAI:
    """建立 Gemini 2.5 Flash LLM 實例。"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("請在 .env 檔案中設定 GEMINI_API_KEY")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
    )


def detect_file_type(filepath: str) -> str | None:
    """根據副檔名判斷檔案類型，回傳 'image' / 'pdf' / 'text' / None。"""
    ext = Path(filepath).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in PDF_EXTENSIONS:
        return "pdf"
    elif ext in TEXT_EXTENSIONS:
        return "text"
    return None


def load_image_as_base64(filepath: str) -> dict:
    """讀取圖片並轉為 base64 格式的 LangChain content block。"""
    mime_type, _ = mimetypes.guess_type(filepath)
    if not mime_type:
        mime_type = "image/jpeg"

    with open(filepath, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
    }


def load_pdf_text(filepath: str) -> str:
    """使用 PyPDFLoader 讀取 PDF 全文。"""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(filepath)
    pages = loader.load()
    full_text = "\n\n".join(
        f"[第 {i + 1} 頁]\n{page.page_content}" for i, page in enumerate(pages)
    )
    return full_text


def load_text_file(filepath: str) -> str:
    """讀取純文字檔案內容。"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def parse_file_command(user_input: str) -> tuple[str | None, str]:
    """
    解析使用者輸入，偵測 /file 指令。
    回傳 (檔案路徑, 使用者問題文字)。
    若不是 /file 指令，回傳 (None, 原始輸入)。
    """
    if not user_input.startswith("/file "):
        return None, user_input

    # 移除 /file 前綴
    rest = user_input[6:].strip()

    # 嘗試解析檔案路徑（支援帶引號的路徑和不帶引號的路徑）
    if rest.startswith('"'):
        # 帶引號的路徑：/file "C:\path with spaces\file.pdf" 問題
        end_quote = rest.find('"', 1)
        if end_quote == -1:
            return None, user_input
        filepath = rest[1:end_quote]
        question = rest[end_quote + 1:].strip()
    else:
        # 不帶引號：取第一個空格前的部分作為路徑
        parts = rest.split(" ", 1)
        filepath = parts[0]
        question = parts[1] if len(parts) > 1 else ""

    return filepath, question


def build_multimodal_message(
    file_type: str, filepath: str, question: str
) -> tuple[HumanMessage, str]:
    """
    根據檔案類型組合 HumanMessage。
    回傳 (HumanMessage 物件, 用於紀錄的文字描述)。
    """
    default_prompts = {
        "image": "請描述這張圖片的內容。",
        "pdf": "請摘要這份 PDF 文件的內容。",
        "text": "請分析以下文字內容。",
    }
    prompt = question if question else default_prompts.get(file_type, "請分析此檔案。")

    if file_type == "image":
        # 圖片：使用多模態格式
        image_block = load_image_as_base64(filepath)
        content = [
            {"type": "text", "text": prompt},
            image_block,
        ]
        record_text = f"[圖片: {filepath}] {prompt}"
        return HumanMessage(content=content), record_text

    elif file_type == "pdf":
        # PDF：提取文字後作為上下文放入 prompt
        pdf_text = load_pdf_text(filepath)
        full_prompt = f"{prompt}\n\n---\n以下是 PDF 文件內容：\n{pdf_text}"
        content = [{"type": "text", "text": full_prompt}]
        record_text = f"[PDF: {filepath}] {prompt}"
        return HumanMessage(content=content), record_text

    elif file_type == "text":
        # 純文字檔：讀取內容放入 prompt
        text_content = load_text_file(filepath)
        full_prompt = f"{prompt}\n\n---\n以下是文字檔內容：\n{text_content}"
        content = [{"type": "text", "text": full_prompt}]
        record_text = f"[TXT: {filepath}] {prompt}"
        return HumanMessage(content=content), record_text

    else:
        raise ValueError(f"不支援的檔案類型: {file_type}")


def sanitize_text(text: str) -> str:
    """移除無法編碼的 surrogate 字元。"""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def save_history_to_json(records: list[dict]) -> str:
    """將對話紀錄儲存為 JSON 檔案，回傳檔名。"""
    filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    clean_records = []
    for record in records:
        clean = {
            "timestamp": record["timestamp"],
            "role": record["role"],
            "content": sanitize_text(record["content"]),
        }
        if "file" in record:
            clean["file"] = record["file"]
        clean_records.append(clean)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(clean_records, f, ensure_ascii=False, indent=2)
    return filename


def main():
    llm = create_llm()
    history = InMemoryChatMessageHistory()
    records: list[dict] = []

    print("=" * 50)
    print("  Gemini 2.5 Flash 多模態聊天室")
    print("=" * 50)
    print("  指令說明：")
    print("  • 直接輸入文字進行對話")
    print("  • /file <路徑> [問題]  傳送檔案")
    print("    支援格式：JPG, PNG, PDF, TXT")
    print("  • exit  結束對話並儲存紀錄")
    print("=" * 50)
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        timestamp = datetime.now().isoformat()
        filepath, question = parse_file_command(user_input)

        if filepath:
            # === 檔案模式 ===
            # 驗證檔案存在
            if not os.path.isfile(filepath):
                print(f"\n⚠ 找不到檔案: {filepath}\n")
                continue

            file_type = detect_file_type(filepath)
            if not file_type:
                ext = Path(filepath).suffix
                print(f"\n⚠ 不支援的檔案格式: {ext}")
                print("  支援的格式：JPG, PNG, PDF, TXT\n")
                continue

            print(f"  📎 載入 {file_type.upper()} 檔案: {filepath}")

            try:
                human_msg, record_text = build_multimodal_message(
                    file_type, filepath, question
                )
            except Exception as e:
                print(f"\n⚠ 讀取檔案失敗: {e}\n")
                continue

            # 加入歷史（多模態 message 直接加入）
            history.add_message(human_msg)
            records.append({
                "timestamp": timestamp,
                "role": "user",
                "content": record_text,
                "file": {
                    "path": filepath,
                    "type": file_type,
                },
            })

        else:
            # === 純文字模式 ===
            history.add_user_message(user_input)
            records.append({
                "timestamp": timestamp,
                "role": "user",
                "content": user_input,
            })

        # 呼叫 Gemini
        try:
            response = llm.invoke(history.messages)
            ai_text = response.content
        except Exception as e:
            ai_text = f"[錯誤] 呼叫 API 失敗: {e}"

        # 記錄 AI 回覆
        timestamp = datetime.now().isoformat()
        history.add_ai_message(ai_text)
        records.append({
            "timestamp": timestamp,
            "role": "ai",
            "content": ai_text,
        })

        print(f"\nAI: {ai_text}\n")

    # 儲存對話紀錄
    if records:
        filename = save_history_to_json(records)
        print(f"\n對話紀錄已儲存至: {filename}")
    else:
        print("\n沒有對話紀錄需要儲存。")


if __name__ == "__main__":
    main()
