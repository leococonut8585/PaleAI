import mimetypes
import os
from typing import Dict, Tuple, Optional, Any
from fastapi.concurrency import run_in_threadpool
from fastapi import Request
import io

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from openpyxl import load_workbook

# --- 定数定義 ---
MAX_TEXT_FILE_SIZE_MB = 10  # TXT, MD, code files
MAX_PDF_SIZE_MB = 50       # PDF files for text extraction
MAX_DOCX_SIZE_MB = 20      # DOCX files
MAX_XLSX_SIZE_MB = 20      # XLSX files
MAX_IMAGE_SIZE_MB = 20     # Images for Gemini Vision
MAX_AUDIO_SIZE_MB = 25     # Audio for Whisper

MB_TO_BYTES = 1024 * 1024

SUPPORTED_TEXT_EXT = ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.csv', '.log', '.rtf']
SUPPORTED_PDF_EXT = ['.pdf']
SUPPORTED_DOCX_EXT = ['.docx']
SUPPORTED_XLSX_EXT = ['.xlsx']
SUPPORTED_IMAGE_MIMETYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif', 'image/bmp']
SUPPORTED_AUDIO_MIMETYPES = ['audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/x-m4a', 'audio/ogg', 'audio/flac', 'audio/webm']

# --- ヘルパー関数 ---
def get_file_details(filename: str, content: bytes) -> Tuple[str, str, int, Optional[str]]:
    """ファイル名、拡張子、MIMEタイプ、サイズを取得"""
    name_no_ext, extension = os.path.splitext(filename)
    extension = extension.lower()
    mime_type, _ = mimetypes.guess_type(filename)
    size_bytes = len(content)
    return name_no_ext, extension, mime_type, size_bytes


def create_error_response(message: str, status_code: int = 400, ai_used: str = "N/A") -> Dict[str, Any]:
    return {
        "processed_content": None,
        "processing_ai": ai_used,
        "error": message,
        "status_code": status_code,
    }


def create_success_response(processed_content: str, ai_used: str, original_filename: str, content_type: Optional[str], size_bytes: int) -> Dict[str, Any]:
    return {
        "processed_content": processed_content,
        "original_filename": original_filename,
        "content_type": content_type,
        "size_bytes": size_bytes,
        "processing_ai": ai_used,
        "error": None,
        "status_code": 200,
    }


# --- 各ファイルタイプ処理関数 ---
async def process_text_file(filename: str, content: bytes, extension: str, size_bytes: int) -> Dict[str, Any]:
    if size_bytes > MAX_TEXT_FILE_SIZE_MB * MB_TO_BYTES:
        return create_error_response(
            f"テキストファイル ({extension}) は {MAX_TEXT_FILE_SIZE_MB}MB までしかアップロードできません。", 413
        )
    try:
        text_content = ""
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text_content = content.decode('shift-jis')
            except UnicodeDecodeError:
                text_content = content.decode('latin-1', errors='replace')

        return create_success_response(text_content, "Direct Read", filename, mimetypes.guess_type(filename)[0], size_bytes)
    except Exception as e:  # pragma: no cover - unexpected decode issues
        return create_error_response(f"テキストファイル処理エラー: {str(e)}", 500)


async def process_pdf_file(filename: str, content: bytes, size_bytes: int) -> Dict[str, Any]:
    if size_bytes > MAX_PDF_SIZE_MB * MB_TO_BYTES:
        return create_error_response(
            f"PDFファイルは {MAX_PDF_SIZE_MB}MB までしかアップロードできません。", 413
        )
    try:
        def extract_text_from_pdf():
            doc = fitz.open("pdf", content)  # type: ignore
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text", sort=True)
                if page_num < len(doc) - 1:
                    text += "\n--- Page Break ---\n"
            doc.close()
            return text

        extracted_text = await run_in_threadpool(extract_text_from_pdf)
        if not extracted_text.strip():
            return create_success_response(
                "[このPDFからはテキストが抽出できませんでした。画像ベースのPDFである可能性があります。]",
                "PyMuPDF (fitz)",
                filename,
                "application/pdf",
                size_bytes,
            )

        return create_success_response(extracted_text, "PyMuPDF (fitz)", filename, "application/pdf", size_bytes)
    except Exception as e:
        return create_error_response(f"PDFファイル処理エラー: {str(e)}", 500, "PyMuPDF (fitz)")


async def process_docx_file(filename: str, content: bytes, size_bytes: int) -> Dict[str, Any]:
    if size_bytes > MAX_DOCX_SIZE_MB * MB_TO_BYTES:
        return create_error_response(
            f"Word (docx) ファイルは {MAX_DOCX_SIZE_MB}MB までしかアップロードできません。", 413
        )
    try:
        def extract_text_from_docx():
            doc = DocxDocument(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text

        extracted_text = await run_in_threadpool(extract_text_from_docx)
        return create_success_response(extracted_text, "python-docx", filename, mimetypes.guess_type(filename)[0], size_bytes)
    except Exception as e:
        return create_error_response(f"Word (docx) ファイル処理エラー: {str(e)}", 500, "python-docx")


async def process_xlsx_file(filename: str, content: bytes, size_bytes: int) -> Dict[str, Any]:
    if size_bytes > MAX_XLSX_SIZE_MB * MB_TO_BYTES:
        return create_error_response(
            f"Excel (xlsx) ファイルは {MAX_XLSX_SIZE_MB}MB までしかアップロードできません。", 413
        )
    try:
        def extract_text_from_xlsx():
            workbook = load_workbook(io.BytesIO(content))
            text_parts = []
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text_parts.append(f"--- Sheet: {sheet_name} ---")
                for row in sheet.iter_rows(values_only=True):
                    row_texts = [str(cell) if cell is not None else "" for cell in row]
                    text_parts.append(", ".join(row_texts))
                text_parts.append("\n")
            return "\n".join(text_parts)

        extracted_text = await run_in_threadpool(extract_text_from_xlsx)
        return create_success_response(extracted_text, "openpyxl", filename, mimetypes.guess_type(filename)[0], size_bytes)
    except Exception as e:
        return create_error_response(f"Excel (xlsx) ファイル処理エラー: {str(e)}", 500, "openpyxl")


async def process_image_file(request: Request, filename: str, content: bytes, mime_type: str, size_bytes: int) -> Dict[str, Any]:
    if size_bytes > MAX_IMAGE_SIZE_MB * MB_TO_BYTES:
        return create_error_response(
            f"画像ファイルは {MAX_IMAGE_SIZE_MB}MB までしかアップロードできません。", 413
        )
    try:
        gemini_client = request.app.state.gemini_vision_client
        if not gemini_client:
            return create_error_response("Gemini Visionクライアントが利用できません。", 503, "Gemini Vision")

        image_part = {"mime_type": mime_type, "data": content}
        prompt = "この画像に何が写っているか詳細に説明してください。画像内のテキストも読み取ってください。"

        async def generate_image_description():
            model = gemini_client
            response = await model.generate_content_async([prompt, image_part])
            return response.text

        description = await generate_image_description()

        return create_success_response(description, "Gemini Vision", filename, mime_type, size_bytes)
    except Exception as e:
        print(f"Gemini Vision API error: {e}")
        return create_error_response(f"画像処理エラー (Gemini Vision): {str(e)}", 500, "Gemini Vision")


async def process_audio_file(request: Request, filename: str, content: bytes, mime_type: str, size_bytes: int) -> Dict[str, Any]:
    if size_bytes > MAX_AUDIO_SIZE_MB * MB_TO_BYTES:
        return create_error_response(
            f"音声ファイルは {MAX_AUDIO_SIZE_MB}MB までしかアップロードできません。", 413
        )
    try:
        openai_client = request.app.state.openai_client
        if not openai_client:
            return create_error_response("OpenAIクライアントが利用できません。", 503, "OpenAI Whisper")

        audio_file_like = io.BytesIO(content)

        async def transcribe_audio():
            response = await openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=(filename, audio_file_like, mime_type),
            )
            return response.text

        transcript_text = await transcribe_audio()

        return create_success_response(transcript_text, "OpenAI Whisper", filename, mime_type, size_bytes)
    except Exception as e:
        print(f"OpenAI Whisper API error: {e}")
        return create_error_response(f"音声処理エラー (OpenAI Whisper): {str(e)}", 500, "OpenAI Whisper")


# --- メイン処理関数 ---
async def stage0_process_file(request: Request, filename: str, content: bytes) -> Dict[str, Any]:
    """アップロードされたファイルを処理し、テキストコンテンツまたはエラー情報を返す。"""
    name_no_ext, extension, mime_type, size_bytes = get_file_details(filename, content)

    if extension in SUPPORTED_TEXT_EXT:
        return await process_text_file(filename, content, extension, size_bytes)
    elif extension in SUPPORTED_PDF_EXT:
        return await process_pdf_file(filename, content, size_bytes)
    elif extension in SUPPORTED_DOCX_EXT:
        return await process_docx_file(filename, content, size_bytes)
    elif extension in SUPPORTED_XLSX_EXT:
        return await process_xlsx_file(filename, content, size_bytes)

    if mime_type:
        if mime_type in SUPPORTED_IMAGE_MIMETYPES:
            return await process_image_file(request, filename, content, mime_type, size_bytes)
        elif mime_type in SUPPORTED_AUDIO_MIMETYPES:
            return await process_audio_file(request, filename, content, mime_type, size_bytes)

    unsupported_message = f"この形式のファイル ({extension if extension else mime_type if mime_type else '不明'}) は扱えません。"
    return create_error_response(unsupported_message, 415)
