"""
Dynamic Workflow Orchestrator with 3 Routing Strategies
Routing workflows based on Classification Agent output

multi-agent-fits-dev-02/app/orchestration/orchestrator.py

"""

import asyncio
import hashlib
import json
import logging
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4, UUID
from datetime import datetime

from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import AsyncSessionLocal
from app.agents.analysis.models import AnalysisRequest, AnalysisResult
from app.core.constants import AnalysisStatus

from sqlalchemy import select
from app.db.models import Session as SessionModel

from app.agents.classification_parameter.unified_FITS_classification_parameter_agent import UnifiedFITSClassificationAgent

from app.core.constants import (
    AgentNames,
    RoutingStrategy,
    WorkflowStatusType,
    ResourceLimits,
    ErrorMessages
    )

logger = logging.getLogger(__name__)

# ==================================================
# Define the data models
# ==================================================

class UserRequest(BaseModel):
    user_id: UUID
    session_id: str | None = None
    request_id: str | None = None
    fits_file_id: str | None = None
    user_query: str
    context: Dict[str, Any] = {}


class WorkflowStatus(BaseModel):
    task_id: str
    status: WorkflowStatusType
    routing_strategy: RoutingStrategy | None = None # astrosage, analysis, mixed
    current_step: str | None = None
    completed_steps: List[Dict[str, Any]] = []
    progress: str
    created_at: datetime
    completed_at: datetime | None = None
    error: str | None = None

# ==================================================
# Dynamic Workflow Orchestrator
# ==================================================

class DynamicWorkflowOrchestrator:
    """
    Orchestrator with 3 dynamic routhing strategies:
    1. astrosage: Classification → AstroSage → Rewrite
    2. analysis: Classification → Analysis → Rewrite
    3. mixed: Classification  → Analysis → AstroSage → Rewrite
    """

    # Resource limits
    # MAX_GPT_CONCURRENT = 3
    # MAX_ASTROSAGE_CONCURRENT = 1
    # MAX_WORKFLOW_MEMORY = 100  # Max number of workflows to keep in memory
    # MAX_WORKER_CONCURRENT = 20  # Max number of concurrent workflows
    
    def __init__(self):

        # Main queue for incoming user requests
        self.main_queue = asyncio.Queue()

        #  Resource semaphores
        #  IMPORtANT: Classification and Rewrite use GPT-4, so they share the same semaphore
        self.shared_llm_semaphore = asyncio.Semaphore(ResourceLimits.MAX_GPT_CONCURRENT)
        self.astrosage_semaphore = asyncio.Semaphore(ResourceLimits.MAX_ASTROSAGE_CONCURRENT)


        # Storage for workflow statuses 
        self.workflow_results = OrderedDict() # In memory 
        self.workflow_lock = asyncio.Lock()

        # Agent
        self.agents = {} # Will be registered later

    def register_agent(self, name: str, agent: Any):
        """Resister an agent with validation"""
        if not AgentNames.validate(name):
            logger.warning(f"Registering agent with non-standard name: {name}")

        self.agents[name] = agent
        logger.info(f"Agent registered: {name}")

    async def submit_request(self, user_request: UserRequest) -> str:
        """
        Get request from API and built workflow
        Return: task_id for tracking
    
        """

        # Generate unique task_id
        task_id = str(uuid4())

        # Initialize workflow status
        workflow = {
            'task_id': task_id,
            'user_request': user_request,
            'status': WorkflowStatusType.QUEUED,
            'routing_strategy': None,
            'current_step': None,
            'completed_steps': [],
            'progress': '0%',
            'created_at': datetime.now(),
            'completed_at': None,
            'error': None
        }

        # Add to memory
        await self._add_to_memory(task_id, workflow)

        # Enqueue the workflow for processing
        await self.main_queue.put(task_id)
        logger.info(f"Workflow enqueued: {task_id}")

        return task_id

    async def _add_to_memory(self, task_id: str, workflow_data: dict):
        """Add/Update workflow result to memory."""
        # Lock workflow_lock
        async with self.workflow_lock:

            # Check if memory full
            if len(self.workflow_results) >= ResourceLimits.MAX_WORKFLOW_MEMORY:
                # Find older completed/failed workflow to remove
                for old_id in list(self.workflow_results.keys()):   # use list to avoid RuntimeError
                    if self.workflow_results[old_id]['status'] in [
                        WorkflowStatusType.COMPLETED,
                        WorkflowStatusType.FAILED
                    ]:
                        del self.workflow_results[old_id]
                        logger.info(f"Removed old workflow from memory: {old_id}")
                        break  # Remove only one

            # Add new workflow
            self.workflow_results[task_id] = workflow_data
            logger.info(f"Added workflow to memory: {task_id}") 
            # Move to end (mark as recently accessed)
            self.workflow_results.move_to_end(task_id)
            logger.debug(f"Updated workflow in memory: {task_id}")

    async def _get_from_memory(self, task_id: str) -> Optional[dict]:
        """Get workflow result from memory."""
        # Lock workflow_lock
        async with self.workflow_lock:

            # Get workflow
            workflow = self.workflow_results.get(task_id)
            if workflow:
                # Mark as recently accessed
                self.workflow_results.move_to_end(task_id)
            return workflow

    async def start_workers(self, num_workers: int = 5):
        """Start worker tasks to process the main queue."""
        workers = [
            asyncio.create_task(self._worker(f"worker-{i+1}"))
            for i in range(num_workers)
        ]

        # wait for all workers to finish (they won't, as they run indefinitely)
        await asyncio.gather(*workers)

    async def stop_workers(self):
        """Stop all workers gracefully."""
        # For now, just log - workers will stop when app shuts down
        logger.info("Stopping workers ...")

    async def _worker(self, worker_name: str):
        """
        Worker gets task from main queue and process the workflow
        """
        logger.info(f"{worker_name} started.")
        
        while True:
            try: 
                # Get task from main queue
                task_id = await self.main_queue.get()
                logger.info(f"{worker_name} picked up task: {task_id}")

                # Process the workflow
                await self._process_workflow(task_id)

                # Mark task as done
                self.main_queue.task_done()

            except Exception as e:
                logger.error(f"Error in {worker_name} while processing task {task_id}: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on error

    async def get_workflow_status(self, task_id: str) -> Optional[WorkflowStatus]:
        """Get status of a workflow by task_id."""
        workflow = await self._get_from_memory(task_id)
        if not workflow:
            return None
        
        return WorkflowStatus(**workflow)

    # Helper method to build AnalysisRequest
    def _build_analysis_request(self, workflow: dict) -> AnalysisRequest:
        """
        Build AnalysisRequest from workflow data

        Args:
            workflow : Workflow dictionay containing user_request and completed_steps

        Returns:
            AnalysisRequest ready for Analysis Agent

        Raises: 
            ValueError: If required data is missing
        """

        # Extract user request
        user_request = workflow['user_request']

        # Find classification result from complete_steps
        classification_step = None
        for step in workflow['completed_steps']:
            if step['step'] == 'classification':
                classification_step = step
                break
            
        if not classification_step:
            raise ValueError("Classification step not found in workflow")
        
        classification_result = classification_step.get('classification_result')
        if not classification_result:
            raise ValueError("Classification result is empty")

        # Extract analysis types and parameters
        analysis_types = classification_result.get('analysis_types', [])
        parameters = classification_result.get('parameters', {})

        if not analysis_types:
            raise ValueError("No analysis types specified by classification")
        
        # Convert fits_file_id string to UUID
        # Format: "5c006e62-36d2-4f16-82c7-f66861522a06.fits" → UUID
        fits_file_id = user_request.fits_file_id

        if not fits_file_id:
            raise ValueError("No FITS file ID provided")
        
        # Remove .fits extension if present
        file_id_str = fits_file_id.replace('.fits', '').replace('.fit', '')
        
        try:
            file_id = UUID(file_id_str)
        except ValueError as e:
            raise ValueError(f"Invalid FITS file ID format: {fits_file_id}") from e

        # Build AnalysisRequest
        analysis_request = AnalysisRequest(
            file_id=file_id,
            user_id=user_request.user_id,
            session_id=user_request.session_id or str(uuid4()),
            analysis_types=analysis_types,
            parameters=parameters,
            context=user_request.context
        )
        
        logger.info(
            f"Built AnalysisRequest: file_id={file_id}, "
            f"analysis_types={analysis_types}"
        )
        
        return analysis_request
    
    async def _ensure_session_exists(
        self,
        session_id: str,
        user_id: UUID,
        db_session: AsyncSession
    ) -> None:
        """
        Ensure session record exists in database.
        
        This method is called at the start of every workflow to guarantee
        that the session_id referenced in subsequent operations exists.
        
        Args:
            session_id: Session identifier (can be UUID or custom string)
            user_id: User who owns the session
            db_session: Database session for queries
            
        Raises:
            Exception: If session creation fails
        """
        try:
            # Check if session exists
            result = await db_session.execute(
                select(SessionModel).where(SessionModel.session_id == session_id)
            )
            existing_session = result.scalar_one_or_none()
            
            if not existing_session:
                logger.info(f"Creating new session record: {session_id}")
                
                new_session = SessionModel(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=datetime.now(),
                    last_activity_at=datetime.now(),
                    is_active=True,
                    session_metadata={}
                )
                
                db_session.add(new_session)
                await db_session.flush()
                
                logger.info(f"Session created successfully: {session_id}")
            else:
                # Update last activity
                existing_session.last_activity_at = datetime.now()
                await db_session.flush()
                
                logger.debug(f"Session already exists, updated activity: {session_id}")
                
        except Exception as e:
            logger.error(f"Failed to ensure session exists: {e}", exc_info=True)
            raise

    async def _process_workflow(self, task_id: str):
        """ 
        Process workflow with dynamic routing based on classification output.
        1. Classification Agent to determine routing strategy
        2. Route to AstroSage or Analysis or both based on strategy
        3. Rewrite Agent to finalize response
        4. Update workflow status in memory
        """

        
        try:
            # ========================================
            # STEP 0: Load Workflow (Database-backed)
            # ========================================
            workflow = await self._get_from_memory(task_id)

            if not workflow:
                logger.error(ErrorMessages.WORKFLOW_NOT_FOUND.format(task_id))
                return

            start_time = datetime.now()

            # ========================================
            # STEP 1: Database Session
            # ========================================
            # async with self.db.get_session() as session:
            async with AsyncSessionLocal() as session:
                
                # ========================================
                # STEP 2: Ensure Session Exists
                # ========================================
                # logger.info(f"Ensuring session exists for workflow {task_id}")
                
                # try:
                #     await self._ensure_session_exists(
                #         session_id=workflow['session_id'],
                #         user_id=workflow['user_id'],
                #         db_session=session
                #     )
                # except Exception as e:
                #     logger.error(f"Session management failed for {task_id}: {e}")
                #     workflow['status'] = WorkflowStatusType.FAILED
                #     workflow['error'] = f"Session error: {str(e)}"
                #     workflow['completed_at'] = datetime.now().isoformat()
                #     await self._add_to_memory(task_id, workflow)
                #     return


                user_request = workflow['user_request']
            
                # Extract session_id (handle both dict and object)
                if isinstance(user_request, dict):
                    session_id = user_request.get('session_id') or str(uuid4())
                    user_id = user_request.get('user_id')
                    user_query = user_request.get('user_query')
                    context = user_request.get('context', {})
                else:
                    # UserRequest object (Pydantic model)
                    session_id = user_request.session_id or str(uuid4())
                    user_id = user_request.user_id
                    user_query = user_request.user_query
                    context = user_request.context
                
                logger.info(f"Ensuring session exists for workflow {task_id}")
                
                try:
                    await self._ensure_session_exists(
                        session_id=session_id,
                        user_id=user_id,
                        db_session=session
                    )
                except Exception as e:
                    logger.error(f"Session management failed for {task_id}: {e}", exc_info=True)
                    workflow['status'] = WorkflowStatusType.FAILED
                    workflow['error'] = f"Session error: {str(e)}"
                    workflow['completed_at'] = datetime.now()
                    await self._add_to_memory(task_id, workflow)
                    return

                # ========================================
                # STEP 3: CLASSIFICATION AGENT
                # ========================================

                # Update status to in_progress
                workflow['status'] = WorkflowStatusType.IN_PROGRESS
                workflow['current_step'] = 'classification'
                workflow['progress'] = '10%'
                await self._add_to_memory(task_id, workflow)

                # Classification Agent
                # classification_agent: UnifiedFITSClassificationParameterAgent = self.agent.get('classification_parameter_agent')
                classification_agent = self.agents.get(AgentNames.CLASSIFICATION)
                if not classification_agent:
                    raise ValueError(ErrorMessages.AGENT_NOT_FOUND.format(AgentNames.CLASSIFICATION))

                # Run classification with shared LLM semaphore
                # async with self.shared_llm_semaphore:
                #     classification_result = await classification_agent.process_request(
                #         user_input=workflow['user_request'].user_query,
                #         context=workflow['user_request'].context
                #     )

                async with self.shared_llm_semaphore:
                    classification_result = await classification_agent.process_request(
                            user_input=user_query,  # ← ใช้ตัวแปรที่ extract แล้ว
                            context=context          # ← ใช้ตัวแปรที่ extract แล้ว
                        )

                # Extract routing strategy and record result
                routing_strategy = RoutingStrategy(classification_result.routing_strategy)
                workflow['routing_strategy'] = routing_strategy
                workflow['completed_steps'].append({
                    'step': 'classification',
                    # 'result': classification_result
                    'classification_result': {
                        'primary_intent': classification_result.primary_intent,
                        'analysis_types': classification_result.analysis_types,
                        'question_category': classification_result.question_category,
                        'routing_strategy': classification_result.routing_strategy,
                        'confidence': classification_result.confidence,
                        'parameters': classification_result.parameters,
                        'reasoning': classification_result.reasoning
                        },
                    'completed_at': datetime.now().isoformat()
                })
                # workflow['current_step'] = None
                workflow['progress'] = '30%'
                await self._add_to_memory(task_id, workflow)

                # ========================================
                # STEP 4: ROUTING BASED ON STRATEGY
                # ========================================
                if routing_strategy == RoutingStrategy.ASTROSAGE:
                    logger.info(f"Routing strategy: AstroSage only for task {task_id}")
                    workflow = await self._handle_astrosage(workflow, task_id)
                elif routing_strategy == RoutingStrategy.ANALYSIS:
                    logger.info(f"Routing strategy: Analysis only for task {task_id}")
                    workflow = await self._handle_analysis(workflow, task_id)
                elif routing_strategy == RoutingStrategy.MIXED:
                    logger.info(f"Routing strategy: Mixed for task {task_id}")
                    workflow = await self._handle_analysis(workflow, task_id)
                    workflow = await self._handle_astrosage(workflow, task_id)
                else:
                    raise ValueError(ErrorMessages.INVALID_ROUTING_STRATEGY.format(routing_strategy))
                    
                # ========================================
                # STEP 5: REWRITE AGENT (FINAL RESPONSE)
                # ========================================

                # Step 3: Rewrite Agent
                workflow['current_step'] = 'rewrite'
                workflow['progress'] = '90%'
                await self._add_to_memory(task_id, workflow)

                rewrite_agent = self.agents.get(AgentNames.REWRITE)
                if not rewrite_agent:
                    raise ValueError(ErrorMessages.AGENT_NOT_FOUND.format(AgentNames.REWRITE))
                
                if not rewrite_agent:
                    # Temporarily skip if Rewrite Agent not available
                    logger.warning(f"Rewrite Agent not available, skipping final rewrite")
                    workflow['completed_steps'].append({
                        'step': 'rewrite',
                        'result': 'skipped',
                        'reason': 'Rewrite Agent not available',
                        'completed_at': datetime.now().isoformat()
                    })
                else:
                    async with self.shared_llm_semaphore:
                        final_response = await rewrite_agent.rewrite_response(
                            user_input=workflow['user_request'].user_query,
                            context=workflow['user_request'].context,
                            intermediate_results=workflow['completed_steps']
                        )

                    workflow['completed_steps'].append({
                        'step': 'rewrite',
                        'result': final_response,
                        'completed_at': datetime.now().isoformat()
                    })

                # ========================================
                # COMPLETE WORKFLOW
                # ========================================
                # workflow['current_step'] = None
                workflow['status'] = WorkflowStatusType.COMPLETED
                workflow['progress'] = '100%'
                workflow['completed_at'] = datetime.now()
                await self._add_to_memory(task_id, workflow)

                duration = (workflow['completed_at'] - start_time).total_seconds()
                logger.info(f"Workflow {task_id} completed in {duration:.2f}s.")

        except Exception as e:
            logger.error(f"Error processing workflow {task_id}: {e}")
            workflow['status'] = WorkflowStatusType.FAILED
            workflow['error'] = str(e)
            workflow['completed_at'] = datetime.now()
            await self._add_to_memory(task_id, workflow)

    async def _handle_analysis(self, workflow: dict, task_id: str) -> dict:
        """ 
        Handle Analysis step in the workflow
        
        Process:
        1. Update status → 'analysis'
        2. Build AnalysisRequest from workflow
        3. Create new database session
        4. Call Analysis Agent
        5. Receive AnalysisResult
        6. Record result in completed_steps
        7. Handle partial/complete/failed statuses
        
        Args:
            workflow: Current workflow dictionary
            task_id: Task ID for logging
        
        Returns:
            Updated workflow dictionary
        
        Raises:
            ValueError: If analysis cannot be performed
            FileNotFoundError: If FITS file not found
        """  
        # Step 1: Update status
        workflow['current_step'] = 'analysis'
        workflow['progress'] = '50%'
        await self._add_to_memory(task_id, workflow)
        
        try:
            # Step 2: Build AnalysisRequest
            logger.info(f"Building AnalysisRequest for task {task_id}")
            analysis_request = self._build_analysis_request(workflow)
            
            # Step 3: Get Analysis Agent
            analysis_agent = self.agents.get(AgentNames.ANALYSIS)
            if not analysis_agent:
                raise ValueError(ErrorMessages.AGENT_NOT_FOUND.format(AgentNames.ANALYSIS))
            
            # Step 4: Create database session and call Analysis Agent
            logger.info(f"Calling Analysis Agent for task {task_id}")
            
            async with AsyncSessionLocal() as session:
                try:
                    # Call Analysis Agent
                    analysis_result: AnalysisResult = await analysis_agent.process_request(
                        request=analysis_request,
                        session=session
                    )
                    
                    # Commit database changes
                    await session.commit()
                    
                    logger.info(
                        f"Analysis completed for task {task_id}: "
                        f"status={analysis_result.status}, "
                        f"completed={len(analysis_result.completed_analyses)}, "
                        f"failed={len(analysis_result.failed_analyses)}"
                    )
                    
                except Exception as e:
                    # Rollback on error
                    await session.rollback()
                    logger.error(f"Database error during analysis for task {task_id}: {e}")
                    raise
            
            # Step 5: Record result in workflow
            workflow['completed_steps'].append({
                'step': 'analysis',
                'analysis_result': {
                    'analysis_id': str(analysis_result.analysis_id),
                    'status': analysis_result.status,
                    'results': analysis_result.results,
                    'errors': analysis_result.errors,
                    'plots': [
                        {
                            'plot_id': str(p.plot_id),
                            'plot_type': p.plot_type,
                            'plot_url': p.plot_url,
                            'created_at': p.created_at.isoformat()
                        } for p in analysis_result.plots
                    ],
                    'execution_time': analysis_result.execution_time,
                    'completed_analyses': analysis_result.completed_analyses,
                    'failed_analyses': analysis_result.failed_analyses,
                    'skipped_analyses': analysis_result.skipped_analyses
                },
                'completed_at': datetime.now().isoformat()
            })
            
            # Step 6: Handle different result statuses
            if analysis_result.status == AnalysisStatus.FAILED:
                # All analyses failed
                logger.error(
                    f"All analyses failed for task {task_id}: "
                    f"{analysis_result.errors}"
                )
                # Don't fail the entire workflow, let Rewrite Agent handle it
                
            elif analysis_result.status == AnalysisStatus.PARTIAL:
                # Some analyses succeeded
                logger.warning(
                    f"Partial analysis success for task {task_id}: "
                    f"{len(analysis_result.completed_analyses)} succeeded, "
                    f"{len(analysis_result.failed_analyses)} failed"
                )
                # Continue workflow with partial results
                
            elif analysis_result.status == AnalysisStatus.COMPLETED:
                # All analyses succeeded
                logger.info(f"All analyses completed successfully for task {task_id}")
            
            # Update progress
            workflow['progress'] = '70%'
            await self._add_to_memory(task_id, workflow)
            
        except FileNotFoundError as e:
            # FITS file not found - critical error
            logger.error(f"FITS file not found for task {task_id}: {e}")
            workflow['completed_steps'].append({
                'step': 'analysis',
                'status': 'failed',
                'error': f"FITS file not found: {str(e)}",
                'completed_at': datetime.now().isoformat()
            })
            # Mark workflow as failed
            workflow['status'] = WorkflowStatusType.FAILED
            workflow['error'] = f"FITS file not found: {str(e)}"
            
        except ValueError as e:
            # Invalid request data
            logger.error(f"Invalid analysis request for task {task_id}: {e}")
            workflow['completed_steps'].append({
                'step': 'analysis',
                'status': 'failed',
                'error': f"Invalid request: {str(e)}",
                'completed_at': datetime.now().isoformat()
            })
            workflow['status'] = WorkflowStatusType.FAILED
            workflow['error'] = f"Invalid analysis request: {str(e)}"
            
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error in analysis for task {task_id}: {e}", exc_info=True)
            workflow['completed_steps'].append({
                'step': 'analysis',
                'status': 'failed',
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            })
            workflow['status'] = WorkflowStatusType.FAILED
            workflow['error'] = f"Analysis error: {str(e)}"
        
        return workflow
        
    async def _handle_astrosage(self, workflow: dict, task_id: str) -> dict:
        """ 
        Handle AstroSage step in the workflow.
        """  
        workflow['current_step'] = 'astrosage'
        workflow['progress'] = '70%'
        await self._add_to_memory(task_id, workflow)

        # TODO: Implement AstroSage handling logic
        logger.info(f"AstroSage step placeholder for task {task_id}")

        return workflow
