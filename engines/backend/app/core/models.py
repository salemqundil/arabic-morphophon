"""
Core data models for the API
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


class TextRequest(BaseModel):
    """Base request model for text processing"""

    text: str = Field(..., description="Text to process")


class ProcessedResponse(BaseModel):
    """Base response model for processed text"""

    original: str = Field(..., description="Original text")
    processed: str = Field(..., description="Processed text")
    processing_time: float = Field(..., description="Processing time in seconds")


class ErrorResponse(BaseModel):
    """Error response model"""

    status: str = Field(default="error", description="Error status")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")


class UserBase(BaseModel):
    """Base user model"""

    email: str = Field(..., description="User email")
    full_name: Optional[str] = Field(None, description="User full name")


class UserCreate(UserBase):
    """User creation model"""

    password: str = Field(..., description="User password")


class UserInDB(UserBase):
    """User database model"""

    id: int = Field(..., description="User ID")
    hashed_password: str = Field(..., description="Hashed password")
    is_active: bool = Field(default=True, description="Whether the user is active")
    is_admin: bool = Field(default=False, description="Whether the user is an admin")


class User(UserBase):
    """User response model"""

    id: int = Field(..., description="User ID")
    is_active: bool = Field(..., description="Whether the user is active")


class Token(BaseModel):
    """Token model"""

    access_token: str = Field(..., description="Access token")
    token_type: str = Field(default="bearer", description="Token type")


class TokenData(BaseModel):
    """Token data model"""

    username: Optional[str] = None
