"""
multi-agent-fits-dev-02/app/core/constants.py
"""

from enum import Enum
from typing import Final

# ==================================
# Agent Names
# ==================================

class AgentNames:
    """Agent identifiers for registration and routing"""
    CLASSIFICATION: Final[str] = "classification_parameter_agent"
    ANALYSIS: Final[str] = "analysis_agent"
    ASTROSAGE: Final[str] = "astrosage_client"
    REWRITE: Final[str] = "rewrite_agent"

    @classmethod
    def all(cls) -> list[str]:
        """Get all agent names"""
        return [
            cls.CLASSIFICATION,
            cls.ANALYSIS,
            cls.ASTROSAGE,
            cls.REWRITE
        ]

    @classmethod
    def validate(cls, agent_name: str) -> bool:
        """Check if agent name is valid"""
        return agent_name in cls.all()

# ==================================
# Routing Strategies
# ==================================
class RoutingStrategy(str, Enum):
    """Workflow routing strategies"""
    ASTROSAGE = "astrosage"         # Classification → AstroSage → Rewrite
    ANALYSIS = "analysis"           # Classification → Analysis → Rewrite
    MIXED = "mixed"                 # Classification → Analysis → AstroSage → Rewrite


# ==================================
# Workflow StatusS
# ==================================
class WorkflowStatusType(str, Enum):
    """Workflow execution status"""
    QUEUED = "queued"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# ==================================
# Analysis Types
# ==================================
class AnalysisType(str, Enum):
    """Types of FITS analysis"""
    STATISTICS = "statistics"
    PSD = "psd"
    POWER_LAW = "power_law"
    BENDING_POWER_LAW = "bending_power_law"
    METADATA = "metadata"
    VALIDATION = "validation"

# ==================================
# Intent Categories
# ==================================
# class IntentCategory(str, Enum):
#     """User intent cagetories"""
#     QUESTION = ""

# ==================================
# Model Names
# ==================================
class PlotType(str, Enum):
    """Types of generated plots"""
    PSD = "psd"
    POWER_LAW = "power_law"
    BENDING_POWER_LAW = "bending_power_law"

# ==================================
# Model Names
# ==================================
class ModelNames:
    """LLM model identifiers"""
    GPT4: Final[str] = "gtp-4"
    GPT35_TURBO: Final[str] = "gpt-3.5-turbo"
    ASTROSAGE_LLAMA31_8B: Final[str] = "AstroSage-Llama-3.1-8B"
    ASTROSAGE_LLAMA31_70B: Final[str] = "AstroSage-Llama-3.1-70B"

# ==================================
# API Pesponse Codes
# ==================================
class ResponseCode:
    """Standard response codes"""
    SUCCESS: Final[int] = 200
    CREATED: Final[int] = 201
    ACCEPTED: Final[int] = 202
    BAD_REQUEST: Final[int] = 400
    UNAUTHORIZED: Final[int] = 401
    FORBIDDEN: Final[int] = 403
    NOT_FOUND: Final[int] = 404
    CONFLICT: Final[int] = 409
    INTERNAL_ERROR: Final[int] = 500
    SERVICE_UNAVAILABLE: Final[int] = 503

# ==================================
# Resource Limits
# ==================================
class ResourceLimits:
    MAX_GPT_CONCURRENT: Final[int] = 3
    MAX_ASTROSAGE_CONCURRENT: Final[int] = 1
    MAX_WORKFLOW_MEMORY: Final[int] = 100
    MAX_WORKER_CONCURRENT: Final[int] = 20
    MAX_FILE_SIZE_MB: Final[int] = 512
    MAX_UPLOAD_SIZE_BYTES: Final[int] = 536_870_912 # 512 MB


# ==================================
# File Extensions
# ==================================
class FileExtensions:
    """Allowed file extensions"""
    FITS: Final[tuple[str, ...]] = (".fits", ".fit")
    PLOTS: Final[str] = ".png"

# ============================================
# Error Messages
# ============================================
class ErrorMessages:
    """Standard error messages"""
    AGENT_NOT_FOUND: Final[str] = "Agent '{}' not found or not registered"
    WORKFLOW_NOT_FOUND: Final[str] = "Workflow '{}' not found"
    FILE_NOT_FOUND: Final[str] = "File '{}' not found"
    INVALID_FILE_TYPE: Final[str] = "Invalid file type. Expected: {}"
    FILE_TOO_LARGE: Final[str] = "File size exceeds maximum limit of {} MB"
    ORCHESTRATOR_NOT_READY: Final[str] = "Orchestrator not initialized"
    INVALID_ROUTING_STRATEGY: Final[str] = "Invalid routing strategy: {}"
    ANALYSIS_FAILED: Final[str] = "Analysis failed: {}"
    ASTROSAGE_UNAVAILABLE: Final[str] = "AstroSage service unavailable"


class TimeConstants:
    """Time-related constants for system operations"""
    
    # ==========================================
    # 1. FILE_RETENTION_DAYS
    # ==========================================
    # FILE_RETENTION_DAYS: Final[int] = 30
    """
    จำนวนวันที่เก็บไฟล์ก่อนจะลบอัตโนมัติ
    
    ใช้สำหรับ:
    - FITS files ที่ผู้ใช้อัปโหลด
    - Plot images ที่สร้างจากการ analysis
    - Temporary files และ cache
    
    หมายเหตุ:
    - ไฟล์ที่ไม่มีการเข้าถึงภายใน 30 วัน จะถูกลบ
    - ยกเว้นไฟล์ที่ผู้ใช้ "mark" ไว้
    - ช่วยประหยัด storage space
    
    ตัวอย่างการใช้งาน:
    >>> from datetime import datetime, timedelta
    >>> cutoff = datetime.now() - timedelta(days=TimeConstants.FILE_RETENTION_DAYS)
    >>> # ลบไฟล์ที่เก่ากว่า cutoff
    """
    
    # ==========================================
    # 2. SESSION_TIMEOUT_MINUTES
    # ==========================================
    # SESSION_TIMEOUT_MINUTES: Final[int] = 60
    """
    เวลาหมดอายุของ session (นาที)
    
    ใช้สำหรับ:
    - User session ที่ไม่มี activity
    - Chat context ที่ไม่ได้ใช้งาน
    - Temporary analysis state
    
    หมายเหตุ:
    - Session จะหมดอายุหลังจาก 60 นาทีไม่มีการใช้งาน
    - ช่วยปล่อย memory และ resources
    - User ต้อง refresh หรือเริ่ม session ใหม่
    
    ตัวอย่างการใช้งาน:
    >>> session_expiry = datetime.now() + timedelta(
    ...     minutes=TimeConstants.SESSION_TIMEOUT_MINUTES
    ... )
    >>> if datetime.now() > session_expiry:
    ...     # Session expired, need to create new one
    """
    
    # ==========================================
    # 3. REQUEST_TIMEOUT_SECONDS
    # ==========================================
    REQUEST_TIMEOUT_SECONDS: Final[int] = 300
    """
    เวลา timeout สำหรับการประมวลผล request (วินาที) = 5 นาที
    
    ใช้สำหรับ:
    - Analysis workflows ที่ใช้เวลานาน
    - LLM API calls (GPT-4, AstroSage)
    - FITS file processing
    
    หมายเหตุ:
    - หากการประมวลผลเกิน 5 นาที จะถูก cancel
    - ป้องกัน hanging requests
    - ป้องกัน resource exhaustion
    
    ตัวอย่างการใช้งาน:
    >>> import asyncio
    >>> try:
    ...     result = await asyncio.wait_for(
    ...         long_running_task(),
    ...         timeout=TimeConstants.REQUEST_TIMEOUT_SECONDS
    ...     )
    ... except asyncio.TimeoutError:
    ...     logger.error("Request timed out")
    """
    
    # ==========================================
    # 4. RETRY_DELAY_SECONDS
    # ==========================================
    RETRY_DELAY_SECONDS: Final[int] = 5
    """
    ระยะเวลารอก่อน retry เมื่อเกิด error (วินาที)
    
    ใช้สำหรับ:
    - External API calls (AstroSage, OpenAI)
    - Database connections
    - Network operations
    
    หมายเหตุ:
    - ใช้กับ exponential backoff strategy
    - ป้องกัน rate limiting
    - ลด load บน external services
    
    ตัวอย่างการใช้งาน:
    >>> for attempt in range(TimeConstants.MAX_RETRIES):
    ...     try:
    ...         result = await call_external_api()
    ...         break
    ...     except Exception as e:
    ...         if attempt < TimeConstants.MAX_RETRIES - 1:
    ...             await asyncio.sleep(
    ...                 TimeConstants.RETRY_DELAY_SECONDS * (2 ** attempt)
    ...             )  # Exponential backoff: 5s, 10s, 20s
    """
    
    # ==========================================
    # 5. MAX_RETRIES
    # ==========================================
    MAX_RETRIES: Final[int] = 3
    """
    จำนวนครั้งสูงสุดในการ retry เมื่อเกิด error
    
    ใช้สำหรับ:
    - Failed API calls
    - Network timeouts
    - Temporary service unavailability
    
    หมายเหตุ:
    - ลอง 3 ครั้ง (total 4 attempts รวมครั้งแรก)
    - หลังจากครบจำนวน retry จะ raise error
    - ใช้ร่วมกับ RETRY_DELAY_SECONDS
    
    ตัวอย่างการใช้งาน:
    >>> async def call_with_retry():
    ...     last_error = None
    ...     for attempt in range(TimeConstants.MAX_RETRIES + 1):
    ...         try:
    ...             return await external_call()
    ...         except Exception as e:
    ...             last_error = e
    ...             if attempt < TimeConstants.MAX_RETRIES:
    ...                 await asyncio.sleep(TimeConstants.RETRY_DELAY_SECONDS)
    ...     raise last_error  # After all retries failed
    """

# ==================================
# Analysis Result Status
# ==================================

class AnalysisStatus(str, Enum):
    """Status of analysis execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"  # Some analyses succeeded, some failed
    FAILED = "failed"
    CANCELLED = "cancelled"


# ==================================
# Parameter Source
# ==================================

class ParameterSource(str, Enum):
    """Source of analysis parameters"""
    USER_SPECIFIED = "user_specified"  # User explicitly provided
    DEFAULTS_USED = "defaults_used"    # Using default values
    INFERRED = "inferred"              # Inferred from context
    MIXED = "mixed"                    # Combination of above
    EXTRACT_ALL = "extract_all"       # For metadata extraction

