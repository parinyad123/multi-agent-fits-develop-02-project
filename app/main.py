"""
multi-agent-fits-dev-02/app/main.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.orchestration.orchestrator import DynamicWorkflowOrchestrator
from app.agents.classification_parameter.unified_FITS_classification_parameter_agent import (
    UnifiedFITSClassificationAgent
    )

from app.core.constants import AgentNames
from app.core.config import settings
from app.db.base import init_db, close_db

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Golbal orchestrator instance
orchestrator: DynamicWorkflowOrchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator

    # STARTUP
    logging.info("Starting application...")

    # Initialize database
    logger.info("nitializing database...")
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Ensure storage directories exist
    logger.info("Ensuring storage directories exist...")
    settings.ensure_directories()
    logger.info(f"  FITS files: {settings.fits_path}")
    logger.info(f"  Plots: {settings.plots_path}")
    logger.info(f"  PSD plots: {settings.psd_plots_path}")
    logger.info(f"  Power law plots: {settings.powerlaw_plots_path}")
    logger.info(f"  Bending power law plots: {settings.bendingpowerlaw_plots_path}")

    # Initialize the orchestrator
    orchestrator = DynamicWorkflowOrchestrator()

    # Register agents
    classification_agent = UnifiedFITSClassificationAgent(
        model_name="gpt-3.5-turbo",
        temperature=0.1
    )

    orchestrator.register_agent(AgentNames.CLASSIFICATION, classification_agent)

    # TODO: Add more agents as needed

    # Strat workers
    import asyncio
    # Create a task to run workers in the background
    asyncio.create_task(orchestrator.start_workers(num_workers=3))

    yield

    # SHUTDOWN
    logging.info("Shutting down application...")
    await orchestrator.stop_workers()

    # Close database connections
    await close_db()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="FITS Analysis Multi-Agent System",
    description="A multi-agent orchestrator system for FITS file analysis",
    version="0.1.0",
    lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_orchestrator() -> DynamicWorkflowOrchestrator:
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator

# Include routers
from app.api.v1 import analysis
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])

@app.get("/")
async def root():
    return {
        "message": "FITS Analysis Multi-Agent System is running.",
        "version": "0.1.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy",
            "orchestrator_status": "running" if orchestrator else "not running",
            "orchestrator": orchestrator is not None,
            "database_status": "connected"
            }