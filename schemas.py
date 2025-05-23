from pydantic import BaseModel, Field, conlist
from typing import Optional, List, Dict, Any
from datetime import datetime

# --- Auth & User Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: str
    username: str = Field(..., min_length=4, max_length=20, pattern=r"^[A-Za-z0-9ぁ-んァ-ヶ一-龠ー]+$")
    gender: str
    color1: str = Field(..., pattern=r"^#[0-9a-fA-F]{6}$")
    color2: str = Field(..., pattern=r"^#[0-9a-fA-F]{6}$")

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    profile_image_url: Optional[str] = None
    is_active: bool

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    gender: Optional[str] = None
    color1: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    color2: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")

# --- Chat Message Schemas ---
class ChatMessageBase(BaseModel):
    role: str
    content: str
    ai_model: Optional[str] = None

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessageResponse(ChatMessageBase):
    id: int
    chat_session_id: int
    user_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True

# --- Chat Session Schemas ---
class ChatSessionBase(BaseModel):
    title: Optional[str] = None
    starred: bool = False
    tags: Optional[str] = None
    mode: str = "chat"
    is_complete: bool = True
    status: Optional[str] = "complete"

class ChatSessionCreate(ChatSessionBase):
    pass

class ChatSessionTitleUpdate(BaseModel):
    title: str = Field(..., min_length=1)

class ChatSessionStarUpdate(BaseModel):
    starred: bool

class ChatSessionTagsUpdate(BaseModel):
    tags: Optional[str] = None

class ChatSessionResponse(ChatSessionBase):
    id: int
    user_id: int
    folder_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    mode: str
    is_complete: bool
    status: str

    class Config:
        from_attributes = True

class ChatSessionDetailResponse(ChatSessionResponse):
    messages: List[ChatMessageResponse] = []

    class Config:
        from_attributes = True

# --- Folder Schemas ---
class FolderBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    position: Optional[int] = None

class FolderCreate(FolderBase):
    pass

class FolderUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=50)

class FolderResponse(FolderBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# --- User Memory Schemas ---
class UserMemoryBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=5000)
    priority: int = Field(0, ge=0, le=10)


class UserMemoryCreate(UserMemoryBase):
    pass


class UserMemoryUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=100)
    content: Optional[str] = Field(None, min_length=1, max_length=5000)
    priority: Optional[int] = Field(None, ge=0, le=10)


class UserMemoryResponse(UserMemoryBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# --- Prompt Template Schemas ---
class PromptTemplateBase(BaseModel):
    title: str
    content: str
    category: Optional[str] = None


class PromptTemplateCreate(PromptTemplateBase):
    pass


class PromptTemplateUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None


class PromptTemplateResponse(PromptTemplateBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# --- AI Response Schemas ---
class IndividualAIResponse(BaseModel):
    source: str
    query: Optional[str] = None
    intent: Optional[str] = None
    response: Optional[str] = None
    links: Optional[List[str]] = None
    error: Optional[str] = None
    # --- 検索結果や断片情報に付随するメタデータ ---
    source_url: Optional[str] = None
    published_date: Optional[str] = None # 日付は文字列として格納 (例: "2024-05-22")
    author: Optional[str] = None
    content_type: Optional[str] = None # 例: "news_article", "sns_post", "research_paper_abstract", "review", "web_page"
    # --- 超検索モード用 ---
    reliability_label: Optional[str] = None # 例: "High", "Medium", "Low", "Unverified"
    bias_label: Optional[str] = None # 例: "Neutral", "Positive Bias", "Negative Bias", "Contains Opinions"
    issues_detected: Optional[List[str]] = None # 例: ["Contradictory Information", "Outdated", "Needs Fact-check"]

class PromptRequestWithHistory(BaseModel):
    prompt: str
    mode: str
    session_id: Optional[int] = None
    char_count: Optional[int] = None
    user_memories: Optional[List[UserMemoryResponse]] = None

class CollaborativeResponseV2(BaseModel):
    prompt: str
    mode_executed: Optional[str] = None
    processed_session_id: Optional[int] = None
    # --- バランスモードなどのステップ別詳細 ---
    step1_initial_draft_openai: Optional[IndividualAIResponse] = None
    step2_review_claude: Optional[IndividualAIResponse] = None
    step3_improved_draft_cohere: Optional[IndividualAIResponse] = None
    step4_comprehensive_answer_perplexity: Optional[IndividualAIResponse] = None
    step5_final_answer_gemini: Optional[IndividualAIResponse] = None
    step6_review2_claude: Optional[IndividualAIResponse] = None
    # --- 全モード共通の最終的な整形済み回答 (検索モードでは整形された断片リスト本体) ---
    step7_final_answer_v2_openai: Optional[IndividualAIResponse] = None

    # --- 検索モード専用フィールド ---
    search_fragments: List[IndividualAIResponse] = Field(default_factory=list) # 収集された全ての生の情報断片リスト
    search_summary_text: Optional[str] = None # 検索モードの最後に付与される非常に短いまとめ
    search_mode_warnings: Optional[Dict[str, Any]] = Field(default_factory=dict) # 超検索モード用の信頼性・バイアス警告など (例: {"reliability_concerns": ["古い情報が含まれます"], "bias_alerts": ["強い意見が含まれるソースあり"]})

    # --- 他モード専用フィールド (既存のものを確認しつつ、必要なら追加) ---
    code_mode_details: Optional[List[IndividualAIResponse]] = None # コード生成の各ステップ詳細
    writing_mode_details: Optional[List[IndividualAIResponse]] = None # 通常執筆の各ステップ詳細
    ultra_writing_mode_details: Optional[List[IndividualAIResponse]] = None # 超長文執筆の各ステップ詳細

    overall_error: Optional[str] = None

    class Config:
        from_attributes = True


class ChatSessionClone(BaseModel):
    folder_id: Optional[int] = None
    title: Optional[str] = None


class MessageSearchResult(BaseModel):
    session_id: int
    message_id: int
    content: str


class TranslationRequest(BaseModel):
    text: str
    target_lang: str = "JA"


class TranslationResponse(BaseModel):
    translated_text: str


class ImageGenerationRequest(BaseModel):
    prompt: str
    count: int = Field(..., ge=1, le=10)
    api: Optional[str] = "openai"


class ImageGenerationResponse(BaseModel):
    urls: List[str]
    error: Optional[str] = None



