from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Text, text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(20), unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    gender = Column(String, nullable=False, server_default="未回答")
    color1 = Column(String(7), nullable=False, server_default="#000000")
    color2 = Column(String(7), nullable=False, server_default="#ffffff")
    profile_image_url = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)

    chat_sessions = relationship("ChatSession", back_populates="owner")
    messages = relationship("ChatMessage", back_populates="author")
    folders = relationship(
        "Folder", back_populates="owner", cascade="all, delete-orphan"
    )  # ★★★ 追加: UserとFolderのリレーション ★★★
    memories = relationship(
        "UserMemory", back_populates="owner", cascade="all, delete-orphan"
    )
    templates = relationship(
        "PromptTemplate", back_populates="owner", cascade="all, delete-orphan"
    )


class Folder(Base):  # ★★★ 新規モデル追加 ★★★
    __tablename__ = "folders"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    position = Column(Integer, nullable=True)

    owner = relationship("User", back_populates="folders")
    chat_sessions = relationship(
        "ChatSession", back_populates="folder"
    )  # FolderとChatSessionのリレーション


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, index=True)
    starred = Column(Boolean, default=False)
    tags = Column(String, nullable=True)
    mode = Column(String, nullable=False, server_default="chat")
    is_complete = Column(Boolean, nullable=False, server_default=text('1'))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    folder_id = Column(
        Integer, ForeignKey("folders.id"), nullable=True
    )  # ★★★ 追加: Folderへの外部キー ★★★

    owner = relationship("User", back_populates="chat_sessions")
    messages = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan"
    )
    folder = relationship(
        "Folder", back_populates="chat_sessions"
    )  # ★★★ 追加: ChatSessionとFolderのリレーション ★★★


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    ai_model = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")
    author = relationship("User", back_populates="messages")


class UserMemory(Base):
    __tablename__ = "user_memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    priority = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    owner = relationship("User", back_populates="memories")


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    owner = relationship("User", back_populates="templates")
