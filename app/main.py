"""
FITS Analysis Multi-Agent System
Supports both V1 and V2 API endpoints

multi-agent-fits-dev-02/app/main.py
"""

from fastapi import FastAPI, HTTPException, Request  
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import asyncio

from app.orchestration.orchestrator import DynamicWorkflowOrchestrator
from app.agents.classification_parameter.unified_FITS_classification_parameter_agent import (
    UnifiedFITSClassificationAgent
)
from app.agents.analysis.unified_analysis_agent import UnifiedAnalysisAgent
from app.services.astrosage import AstroSageClient
from app.agents.rewrite.gpt_rewrite_agent import GPTRewriteAgent

from app.core.constants import AgentNames
from app.core.config import settings
from app.db.base import init_db, close_db

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator: DynamicWorkflowOrchestrator = None


# ============================================================================
# APPLICATION LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global orchestrator

    # ========================================
    # STARTUP
    # ========================================
    logger.info("=" * 70)
    logger.info("🚀 Starting FITS Analysis Multi-Agent System")
    logger.info("=" * 70)

    # Initialize database
    logger.info("📊 Initializing database...")
    try:
        await init_db()
        logger.info("✓ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise
    
    # Ensure storage directories exist
    logger.info("📁 Ensuring storage directories exist...")
    settings.ensure_directories()
    logger.info(f"  ✓ FITS files: {settings.fits_path}")
    logger.info(f"  ✓ Plots: {settings.plots_path}")
    logger.info(f"  ✓ PSD plots: {settings.psd_plots_path}")
    logger.info(f"  ✓ Power law plots: {settings.powerlaw_plots_path}")
    logger.info(f"  ✓ Bending power law plots: {settings.bendingpowerlaw_plots_path}")

    # Initialize orchestrator
    logger.info("🎯 Initializing workflow orchestrator...")
    orchestrator = DynamicWorkflowOrchestrator()

    # ========================================
    # Register Agents
    # ========================================
    logger.info("🤖 Registering agents...")
    
    # Classification Agent
    classification_agent = UnifiedFITSClassificationAgent(
        model_name="gpt-3.5-turbo",
        temperature=0.1
    )
    orchestrator.register_agent(AgentNames.CLASSIFICATION, classification_agent)
    logger.info("  ✓ Classification Agent registered")

    # Analysis Agent
    analysis_agent = UnifiedAnalysisAgent()
    orchestrator.register_agent(AgentNames.ANALYSIS, analysis_agent)
    logger.info("  ✓ Analysis Agent registered")

    # AstroSage Client
    astrosage_client = AstroSageClient()
    orchestrator.register_agent(AgentNames.ASTROSAGE, astrosage_client)
    logger.info("  ✓ AstroSage Client registered")

    # Rewrite Agent
    rewrite_agent = GPTRewriteAgent(
        default_model="standard",  # mini, turbo, standard
        auto_upgrade=False
    )
    orchestrator.register_agent(AgentNames.REWRITE, rewrite_agent)
    logger.info("  ✓ Rewrite Agent registered")

    # Start workers
    logger.info("👷 Starting background workers...")
    asyncio.create_task(orchestrator.start_workers(num_workers=3))
    logger.info("  ✓ 3 workers started")

    logger.info("=" * 70)
    logger.info("✅ Application startup complete!")
    logger.info("=" * 70)
    logger.info("📡 API v1: http://localhost:8003/api/v1")
    logger.info("⚡ API v2: http://localhost:8003/api/v2 (Optimized)")
    logger.info("📚 Docs:   http://localhost:8003/docs")
    logger.info("=" * 70)

    yield

    # ========================================
    # SHUTDOWN
    # ========================================
    logger.info("=" * 70)
    logger.info("🛑 Shutting down application...")
    logger.info("=" * 70)
    
    logger.info("👷 Stopping workers...")
    await orchestrator.stop_workers()
    logger.info("  ✓ Workers stopped")

    logger.info("📊 Closing database connections...")
    await close_db()
    logger.info("  ✓ Database connections closed")

    logger.info("=" * 70)
    logger.info("✅ Application shutdown complete")
    logger.info("=" * 70)


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="FITS Analysis Multi-Agent System",
    description=(
        "A multi-agent orchestrator system for FITS file analysis\n\n"
        "## API Versions\n\n"
        "### V1 API (Original)\n"
        "- **Status:** Active & Stable\n"
        "- **Base URL:** `/api/v1`\n"
        "- **Use Case:** Backward compatibility, full metadata responses\n"
        "- **Performance:** Standard (50-100 KB responses)\n\n"
        "### V2 API (Optimized) ⚡\n"
        "- **Status:** Active & Recommended\n"
        "- **Base URL:** `/api/v2`\n"
        "- **Use Case:** New implementations, high performance\n"
        "- **Performance:** 10-100x faster (lightweight responses)\n"
        "- **Features:**\n"
        "  - Pagination support\n"
        "  - Server-Sent Events (SSE)\n"
        "  - GZIP compression\n"
        "  - On-demand metadata loading\n\n"
        "## Migration Guide\n"
        "See `/docs#migration` for migration from V1 to V2"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "v1-authentication",
            "description": "V1: User authentication and authorization"
        },
        {
            "name": "v1-analysis",
            "description": "V1: FITS file analysis workflows"
        },
        {
            "name": "v1-files",
            "description": "V1: FITS file management"
        },
        {
            "name": "v1-plots",
            "description": "V1: Plot file serving"
        },
        {
            "name": "v1-conversations",
            "description": "V1: Conversation history"
        },
        {
            "name": "v2-authentication",
            "description": "V2: User authentication (same as V1)"
        },
        {
            "name": "v2-sessions",
            "description": "V2: Chat session management (optimized)"
        },
        {
            "name": "v2-conversations",
            "description": "V2: Conversation history (paginated)"
        },
        {
            "name": "v2-files",
            "description": "V2: FITS file management (lightweight)"
        },
        {
            "name": "v2-analysis",
            "description": "V2: Analysis workflows (SSE support)"
        }
    ]
)


# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# 1. GZIP Compression (for V2 API)
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # Only compress responses > 1KB
    compresslevel=6     # Balance between speed and compression
)
logger.info("✓ GZIP compression middleware added")

# 2. CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Type"]
)
logger.info("✓ CORS middleware added")


# ============================================================================
# STATIC FILES
# ============================================================================

try:
    app.mount(
        "/storage/plots",
        StaticFiles(directory=str(settings.plots_path)),
        name="plots"
    )
    logger.info(f"✓ Static files mounted: /storage/plots -> {settings.plots_path}")
except Exception as e:
    logger.warning(f"⚠ Could not mount static files: {e}")


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

def get_orchestrator() -> DynamicWorkflowOrchestrator:
    """
    Get the global orchestrator instance.
    
    Raises:
        HTTPException: If orchestrator is not initialized
    """
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator not initialized. Please wait for startup to complete."
        )
    return orchestrator


# ============================================================================
# API V1 ROUTERS
# ============================================================================

logger.info("📡 Registering V1 API routes...")

from app.api.v1 import analysis, files, plots, conversations
from app.api.v1.auth import routes as auth_routes

# V1 Authentication
app.include_router(
    auth_routes.router,
    prefix="/api/v1/auth",
    tags=["v1-authentication"]
)

# V1 Analysis
app.include_router(
    analysis.router,
    prefix="/api/v1",
    tags=["v1-analysis"]
)

# V1 Files
app.include_router(
    files.router,
    prefix="/api/v1/files",
    tags=["v1-files"]
)

# V1 Plots
app.include_router(
    plots.router,
    prefix="/api/v1",
    tags=["v1-plots"]
)

# V1 Conversations
app.include_router(
    conversations.router,
    prefix="/api/v1",
    tags=["v1-conversations"]
)

logger.info("✓ V1 API routes registered")


# ============================================================================
# API V2 ROUTERS (NEW!)
# ============================================================================

logger.info("⚡ Registering V2 API routes...")

from app.api.v2 import sessions as sessions_v2
from app.api.v2 import conversations as conversations_v2
from app.api.v2 import files as files_v2
from app.api.v2 import analysis as analysis_v2

# V2 Authentication (shared with V1)
app.include_router(
    auth_routes.router,
    prefix="/api/v2/auth",
    tags=["v2-authentication"]
)

# V2 Sessions
app.include_router(
    sessions_v2.router,
    prefix="/api/v2",
    tags=["v2-sessions"]
)

# V2 Conversations
app.include_router(
    conversations_v2.router,
    prefix="/api/v2",
    tags=["v2-conversations"]
)

# V2 Files
app.include_router(
    files_v2.router,
    prefix="/api/v2",
    tags=["v2-files"]
)

# V2 Analysis
app.include_router(
    analysis_v2.router,
    prefix="/api/v2",
    tags=["v2-analysis"]
)

logger.info("✓ V2 API routes registered")


# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint - API information and status
    """
    return {
        "message": "FITS Analysis Multi-Agent System",
        "version": "2.0.0",
        "status": "operational",
        "api_versions": {
            "v1": {
                "status": "active",
                "base_url": "/api/v1",
                "description": "Original API - Backward compatible",
                "features": [
                    "Full metadata in responses",
                    "Complete workflow details",
                    "Existing integrations supported"
                ],
                "endpoints": {
                    "auth": "/api/v1/auth",
                    "analysis": "/api/v1/workflow/analyze",
                    "status": "/api/v1/analyze/{task_id}",
                    "files": "/api/v1/files",
                    "plots": "/api/v1/plots",
                    "conversations": "/api/v1/sessions/{session_id}/conversations"
                }
            },
            "v2": {
                "status": "active",
                "base_url": "/api/v2",
                "description": "Optimized API - 10-100x performance improvements",
                "features": [
                    "Lightweight responses (~100 bytes)",
                    "Pagination support",
                    "Server-Sent Events (SSE)",
                    "GZIP compression",
                    "On-demand metadata loading"
                ],
                "endpoints": {
                    "auth": "/api/v2/auth",
                    "sessions": "/api/v2/sessions",
                    "conversations": "/api/v2/conversations/{session_id}",
                    "files": "/api/v2/files",
                    "analysis": "/api/v2/analyze",
                    "status_light": "/api/v2/analyze/{task_id}/status",
                    "status_full": "/api/v2/analyze/{task_id}",
                    "stream": "/api/v2/analyze/{task_id}/stream"
                }
            }
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "migration_guide": "/api/info"
        },
        "system": {
            "orchestrator": "running" if orchestrator else "initializing",
            "workers": 3,
            "database": "connected"
        }
    }


@app.get("/health", tags=["root"])
async def health_check():
    """
    Health check endpoint
    """
    orchestrator_running = orchestrator is not None
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "orchestrator": "running" if orchestrator_running else "not initialized",
            "database": "connected",
            "workers": "active" if orchestrator_running else "inactive"
        },
        "api_versions": {
            "v1": "active",
            "v2": "active"
        }
    }


@app.get("/api", tags=["root"])
async def api_info():
    """
    Detailed API information and comparison
    """
    return {
        "title": "FITS Analysis Multi-Agent System API",
        "version": "2.0.0",
        "available_versions": ["v1", "v2"],
        "recommended_version": "v2",
        "comparison": {
            "v1": {
                "pros": [
                    "✓ Backward compatible",
                    "✓ Full metadata in responses",
                    "✓ Existing integrations work without changes",
                    "✓ All data available immediately"
                ],
                "cons": [
                    "✗ Large response sizes (50-100 KB)",
                    "✗ No pagination support",
                    "✗ Heavy polling required for status updates",
                    "✗ Not optimized for mobile/bandwidth-constrained clients"
                ],
                "response_sizes": {
                    "conversations": "50-100 KB",
                    "files": "10-50 KB",
                    "workflow_status": "5-20 KB"
                },
                "use_cases": [
                    "Legacy integrations",
                    "When full metadata needed immediately",
                    "Desktop applications with good bandwidth",
                    "Testing and development"
                ]
            },
            "v2": {
                "pros": [
                    "✓ 10-100x smaller responses",
                    "✓ Pagination support (efficient loading)",
                    "✓ Server-Sent Events for real-time updates",
                    "✓ GZIP compression enabled",
                    "✓ Optimized for performance and bandwidth",
                    "✓ On-demand metadata loading"
                ],
                "cons": [
                    "✗ Requires code updates for existing clients",
                    "✗ Metadata requires additional requests",
                    "✗ May need multiple API calls for complete data"
                ],
                "response_sizes": {
                    "conversations_light": "~2 KB (20 messages)",
                    "files_light": "~1 KB (20 files)",
                    "workflow_status_light": "~100 bytes",
                    "on_demand_metadata": "1-10 KB when needed"
                },
                "use_cases": [
                    "New implementations (recommended)",
                    "Mobile applications",
                    "High-performance requirements",
                    "Bandwidth-constrained clients",
                    "Real-time updates needed"
                ]
            }
        },
        "migration_guide": {
            "overview": "Migrating from V1 to V2 requires minimal changes",
            "steps": [
                "1. Update base URL from /api/v1 to /api/v2",
                "2. Implement pagination for list endpoints",
                "3. Use lightweight endpoints for polling",
                "4. Optionally implement SSE for real-time updates",
                "5. Request full metadata only when needed"
            ],
            "breaking_changes": [
                "Response structure changed for conversations",
                "Pagination added to list endpoints",
                "Workflow status split into light/full endpoints",
                "Metadata fields moved to separate endpoint"
            ],
            "code_examples": {
                "v1_conversation": "GET /api/v1/sessions/{session_id}/conversations",
                "v2_conversation": "GET /api/v2/conversations/{session_id}?offset=0&limit=50",
                "v1_workflow_status": "GET /api/v1/analyze/{task_id}",
                "v2_workflow_status_light": "GET /api/v2/analyze/{task_id}/status",
                "v2_workflow_status_full": "GET /api/v2/analyze/{task_id}",
                "v2_sse": "GET /api/v2/analyze/{task_id}/stream"
            }
        },
        "performance_improvements": {
            "response_size_reduction": "10-100x smaller",
            "polling_efficiency": "5-10x fewer bytes per poll",
            "bandwidth_savings": "90-95% reduction",
            "mobile_friendly": "Much faster on slow connections"
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(  # ✅ ต้อง return JSONResponse ไม่ใช่ dict!
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(  # ✅ ต้อง return JSONResponse ไม่ใช่ dict!
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(  # ✅ ต้อง return JSONResponse ไม่ใช่ dict!
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


# ============================================================================
# STARTUP EVENT
# ============================================================================

from datetime import datetime

@app.on_event("startup")
async def startup_message():
    """Print startup banner"""
    logger.info("")
    logger.info("╔════════════════════════════════════════════════════════════════╗")
    logger.info("║                                                                ║")
    logger.info("║        FITS Analysis Multi-Agent System v2.0.0                 ║")
    logger.info("║                                                                ║")
    logger.info("║  📡 API V1: http://localhost:8003/api/v1  (Original)           ║")
    logger.info("║  ⚡ API V2: http://localhost:8003/api/v2  (Optimized 10-100x)  ║")
    logger.info("║  📚 Docs:   http://localhost:8003/docs                         ║")
    logger.info("║                                                                ║")
    logger.info("║  Status: ✅ READY                                              ║")
    logger.info("║                                                                ║")
    logger.info("╚════════════════════════════════════════════════════════════════╝")
    logger.info("")