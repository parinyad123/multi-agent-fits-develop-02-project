"""
multi-agent-fits-dev-02/app/agents/analysis/unified_analysis_agent.py

Unified Analysis Agent - Sequential execution with partial results support
"""

import logging 
from datetime import datetime
from typing import Dict, Any
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.agents.analysis.models import AnalysisRequest, AnalysisResult, PlotInfo
from app.agents.analysis.capabilities.implementations import (
    StatisticsCapability,
    PSDCapability,
    FittingCapability,
    MetadataCapability
)
from app.services.file_service import FileService
from app.tools.fits.loader import load_fits_data
from app.db.models import AnalysisHistory, PlotFile
from app.db.models import Session as SessionModel

from app.core.constants import AnalysisStatus, AnalysisType

logger = logging.getLogger(__name__)

class UnifiedAnalysisAgent:
    """
    Unified Analysis Agent for FITS file analysis

    Features:
    - Sequential execution (one analysis at a time)
    - Partial results support (continue on failure)
    - Plot URLs (not bytes)
    - Uses defaults from Classification Agent
    - Dependency checking (fitting needs PSD)
    """

    def __init__(self):
        self.name = "UnifiedAnalysisAgent"
        self.logger = logging.getLogger(f"agent.{self.name}")

        # Initialze capabilities
        self.capabilities = {
            "statistics": StatisticsCapability(),
            "psd": PSDCapability(),
            "power_law": FittingCapability(AnalysisType.POWER_LAW),
            "bending_power_law": FittingCapability(AnalysisType.BENDING_POWER_LAW),
            "metadata": MetadataCapability()
        }

        self.logger.info("Unified Analysis Agent initialized")

    async def process_request(
            self,
            request: AnalysisRequest,
            session: AsyncSession
    ) -> AnalysisResult:
        """
        Main entry point for analysis processing

        Args:
            request: Analysis request from Orchestrator
            session: Databaser session
        
        Returns:
            AnalysisResult with partial results support
        """

        """
        
        Input:
            AnalysisRequest(
                file_id="123e4567-...",
                user_id="789e0123-...",
                session_id="session_abc",
                analysis_types=["statistics", "psd", "power_law"],  # ← ทำอะไรบ้าง
                parameters={                                         # ← พร้อม defaults แล้ว
                    "statistics": {"metrics": ["mean", "std"]},
                    "psd": {"low_freq": 1e-5, "high_freq": 0.05, "bins": 3500},
                    "power_law": {"A0": 1.0, "b0": 1.0, ...}
                }
            )

        Output:
            AnalysisResult(
                analysis_id=UUID(...),
                status="partial",  # หรือ "completed"
                results={
                    "statistics": {...},
                    "psd": {...}
                },
                errors={
                    "power_law": "Skipped: PSD not available"
                },
                plots=[
                    PlotInfo(plot_type="psd", plot_url="/storage/...")
                ],
                execution_time=2.34,
                completed_analyses=["statistics", "psd"],
                failed_analyses=[],
                skipped_analyses=["power_law"]
            )
        """

        start_time = datetime.now()
        self.logger.info(
            f"Processing analysis request: file={request.file_id}, "
            f"types={request.analysis_types}"
        )

        results = {}        # completed results
        errors = {}         # error
        plots = []          # plot URLs
        completed = []      # completed analysis' name
        failed = []         # failed analysis' name
        skipped = []        # skipped analysis' name

        # ========================================
        # STEP 1: Load FITS file
        # ========================================
        try:
            file_record, rate_data = await self._load_fits_file(
                request.file_id,
                session
            )
            self.logger.info(f"FITS file loaded: {len(rate_data)} data points")

        except Exception as e:
            self.logger.error(f"Failed to load FITS file: {e}")
            return self._create_error_result(
                request,
                "file_loading_failed",
                str(e),
                start_time
            )
        
        # ========================================
        # Ensure session exists before saving analysis
        # ========================================

        # try:
        #     # Check if session exists
        #     result = await session.execute(
        #         select(SessionModel).where(SessionModel.session_id == request.session_id)
        #     )
        #     existing_session = result.scalar_one_or_none()

        #     # If not exists, create it
        #     if not existing_session:
        #         self.logger.info("Creating session record: {request.session_id}")
        #         new_session = SessionModel(
        #         session_id=request.session_id,
        #         user_id=request.user_id,
        #         created_at=datetime.now(),
        #         last_activity_at=datetime.now(),
        #         is_active=True,
        #         session_metadata={}
        #     )
        #     session.add(new_session)
        #     await session.flush()  # Flush to make it available for FK
        #     self.logger.info(f"Session created: {request.session_id}")

        # except Exception as e:
        #     self.logger.error(f"Failed to ensure session exists: {e}")
        
        # ========================================
        # STEP 2: Sequential execution of analyses
        # ========================================
        for analysis_type in request.analysis_types:
            self.logger.info(f"Processing analysis: {analysis_type}")

            try:
                # Check dependencies
                if not self._check_dependencies(analysis_type, results, errors):
                    skip_msg = f"Skipped: required dependency not available"
                    self.logger.warning(f"{analysis_type}: {skip_msg}")
                    errors[analysis_type] = skip_msg
                    skipped.append(analysis_type)
                    continue

                # Get capability
                capability = self.capabilities.get(analysis_type)
                if not capability:
                    raise ValueError(f"Unknown analysis type: {analysis_type}")
                
                # Get parameters (already have defaults from Classification Agent)
                params = request.parameters.get(analysis_type, {})

                # Add filename to parameters for plot titles
                params["filename"] = file_record.metadata_filename or file_record.original_filename

                # Execute analysis
                result, plot_url = await capability.execute(
                    rate_data = rate_data,
                    parameters = params,
                    file_record = file_record
                )

                # Store result
                results[analysis_type] = result
                completed.append(analysis_type)

                # Store plot info if generated
                if plot_url:
                    plot_info = PlotInfo(
                        plot_id=uuid4(),
                        plot_type=analysis_type,
                        plot_url=plot_url,
                        created_at=datetime.now()
                    )
                    plots.append(plot_info)
                
                self.logger.info(f"{analysis_type}: completed successfully")

            except Exception as e:
                # Log error but continus to next analysis
                self.logger.error(f"{analysis_type} failed: {e}")
                errors[analysis_type] = str(e)
                failed.append(analysis_type)

        # ========================================
        # STEP 3: Save to database
        # ========================================
        try:
            analysis_id = await self._save_analysis_history(
                session=session,
                file_id=request.file_id,
                user_id=request.user_id,
                session_id=request.session_id,
                analysis_types=request.analysis_types,
                parameters=request.parameters,
                results=results,
                errors=errors,
                started_at=start_time
            )
            
            # Save plot records
            await self._save_plot_records(
                session=session,
                analysis_id=analysis_id,
                file_id=request.file_id,
                plots=plots
            )
            
            self.logger.info(f"Analysis saved to database: {analysis_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to save analysis to database: {e}")
            # Don't fail the entire analysis just because DB save failed
            analysis_id = uuid4()

        # ========================================
        # STEP 4: Return unified result
        # ========================================
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if not errors:
            status = AnalysisStatus.COMPLETED
        elif results:
            status = AnalysisStatus.PARTIAL
        else:
            status = AnalysisStatus.FAILED
        
        analysis_result = AnalysisResult(
            analysis_id=analysis_id,
            file_id=request.file_id,
            status=status,
            results=results,
            errors=errors,
            plots=plots,
            execution_time=execution_time,
            completed_analyses=completed,
            failed_analyses=failed,
            skipped_analyses=skipped,
            started_at=start_time,
            completed_at=datetime.now()
        )
        
        self.logger.info(
            f"Analysis completed: {len(completed)} succeeded, "
            f"{len(failed)} failed, {len(skipped)} skipped "
            f"(time: {execution_time:.2f}s)"
        )
        
        return analysis_result


    async def _load_fits_file(self, file_id: UUID, session: AsyncSession) -> tuple:
        """
        Load FITS file from database and filesystem

        Return: 
            Tuple of (file_record, rate_data)
        """

        # Quesry database
        # Get file record from database
        file_record = await FileService.get_file_by_id(file_id, session)
        
        # Validate
        if not file_record:
            raise FileNotFoundError(f"File not found: {file_id}")
        
        if not file_record.is_valid:
            raise ValueError(f"File is not valid: {file_record.validation_error}")
        
        if file_record.is_deleted:
            raise ValueError(f"File has been deleted: {file_id}")
        
        # Load actual FITS data filesystem
        rate_data = load_fits_data(file_record.storage_path)

        return file_record, rate_data
    
    def _check_dependencies(self, analysis_type: str, results: dict, errors: dict) -> bool:
        """
        Check if analysis dependencies are satisfied
        
        Args:
            analysis_type: Analysis type to check
            results: Completed results so far
            errors: Errors so far
        
        Returns:
            True if dependencies satisfied, False otherwise
        """

        """
        ตรวจสอบว่า analysis นี้ต้องการอะไรก่อนหรือไม่
        
        Example:
        - "statistics" → ไม่ต้องการอะไร → return True
        - "psd" → ไม่ต้องการอะไร → return True
        - "power_law" → ต้องการ "psd" ก่อน → ถ้าไม่มี return False
        """

        capability = self.capabilities.get(analysis_type)
        if not capability:
            return True
        
        dependencies = capability.get_dependencies()
        
        for dep in dependencies:
            # Dependency must be in results AND not in errors
            if dep not in results or dep in errors:
                return False
        
        return True

    async def _save_analysis_history(
            self,
            session: AsyncSession,
            file_id: UUID,
            user_id: UUID,
            session_id: str,
            analysis_types: list,
            parameters: dict,
            results: dict,
            errors: dict,
            started_at: datetime
    ) -> UUID:
        """
        Save analysis history to database
        
        Returns:
            analysis_id
        """

        analysis_record = AnalysisHistory(
            file_id = file_id,
            user_id = user_id,
            session_id = session_id,
            analysis_types = analysis_types,
            parameters = parameters,
            results={
                "completed": results,
                "errors": errors
            },
            status = AnalysisStatus.PARTIAL if errors else AnalysisStatus.COMPLETED,
            started_at = started_at,
            completed_at = datetime.now(),
            execution_time_seconds = int((datetime.now() - started_at).total_seconds())
        )

        session.add(analysis_record)
        await session.commit()
        await session.refresh(analysis_record)
        
        return analysis_record.analysis_id

    async def _save_plot_records(
        self,
        session: AsyncSession,
        analysis_id: UUID,
        file_id: UUID,
        plots: list
    ):
        """
        Save plot file records to database
        """
        for plot_info in plots:
            plot_record = PlotFile(
                plot_id = plot_info.plot_id,
                analysis_id = analysis_id,
                file_id = file_id,
                plot_type = plot_info.plot_type,
                storage_path = plot_info.plot_url,
                plot_url = plot_info.plot_url,
                plot_metadata = {
                    "created_at": plot_info.created_at.isoformat()
                }
            )
            session.add(plot_record)
        
        await session.commit()

    def _create_error_result(
            self, 
            request: AnalysisRequest, 
            error_type: str,
            error_message: str,
            start_time: datetime
    ) -> AnalysisResult:
        """
        Create error result when analysis fails eompletely
        """
        execution_time = (datetime.now() - start_time).total_seconds()

        return AnalysisResult(
            analysis_id = uuid4(),
            file_id = request.file_id,
            status = AnalysisStatus.FAILED,
            results = {},
            errors = {
                error_type: error_message
            },
            plots = [],
            execution_time = execution_time,
            completed_analyses = [],
            failed_analyses = request.analysis_types,
            skipped_analyses = [],
            started_at = start_time,
            completed_at = datetime.now()
        )