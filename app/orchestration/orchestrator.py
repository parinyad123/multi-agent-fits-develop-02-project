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

from app.agents.classification_parameter.unified_FITS_classification_parameter_agent import UnifiedFITSClassificationAgent

from app.core.constants import AgentNames

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
    status: str
    routing_strategy: str | None # astrosage, analysis, mixed
    current_step: str | None
    completed_steps: List[Dict[str, Any]]
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
    MAX_GPT_CONCURRENT = 3
    MAX_ASTROSAGE_CONCURRENT = 1
    MAX_WORKFLOW_MEMORY = 100  # Max number of workflows to keep in memory
    MAX_WORKER_CONCURRENT = 20  # Max number of concurrent workflows
    
    def __init__(self):

        # Main queue for incoming user requests
        self.main_queue = asyncio.Queue()

        #  Resource semaphores
        #  IMPORtANT: Classification and Rewrite use GPT-4, so they share the same semaphore
        self.shared_llm_semaphore = asyncio.Semaphore(self.MAX_GPT_CONCURRENT)
        self.astrosage_semaphore = asyncio.Semaphore(self.MAX_ASTROSAGE_CONCURRENT)


        # Storage for workflow statuses 
        self.workflow_results = OrderedDict() # In memory 
        self.workflow_lock = asyncio.Lock()

        # Agent
        self.agents = {} # Will be registered later

    def register_agent(self, name: str, agent: Any):
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
            'status': 'queued',
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
            if len(self.workflow_results) >= self.MAX_WORKFLOW_MEMORY:
                # Find older completed/failed workflow to remove
                for old_id in list(self.workflow_results.keys()):   # use list to avoid RuntimeError
                    if self.workflow_results[old_id]['status'] in ['completed', 'failed']:
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
        workers = []
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i+1}"))
            workers.append(worker)

        # wait for all workers to finish (they won't, as they run indefinitely)
        await asyncio.gather(*workers)

    async def stop_workers(self):
        """Stop all workers gracefully."""
        # For now, just log - workers will stop when app shuts down
        logger.info("Stopping workers is not implemented yet.")

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


    async def _process_workflow(self, task_id: str):
        """ 
        Process workflow with dynamic routing based on classification output.
        1. Classification Agent to determine routing strategy
        2. Route to AstroSage or Analysis or both based on strategy
        3. Rewrite Agent to finalize response
        4. Update workflow status in memory
        """

        workflow = await self._get_from_memory(task_id)

        if not workflow:
            logger.error(f"Workflow not found in memory: {task_id}")
            return

        start_time = datetime.now()

        try:
            # ========================================
            # STEP 1: CLASSIFICATION AGENT
            # ========================================

            # Update status to in_progress
            workflow['status'] = 'in_progress'
            workflow['current_step'] = 'classification'
            workflow['progress'] = '10%'
            await self._add_to_memory(task_id, workflow)

            # Classification Agent
            # classification_agent: UnifiedFITSClassificationParameterAgent = self.agent.get('classification_parameter_agent')
            classification_agent = self.agents.get(AgentNames.CLASSIFICATION)
            if not classification_agent:
                raise ValueError("Classification Parameter Agent not registered.")

            # Run classification with shared LLM semaphore
            async with self.shared_llm_semaphore:
                classification_result = await classification_agent.process_request(
                    user_input=workflow['user_request'].user_query,
                    context=workflow['user_request'].context
                )

            # Extract routing strategy and record result
            routing_strategy = classification_result.routing_strategy
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
            workflow['current_step'] = None
            workflow['progress'] = '30%'
            await self._add_to_memory(task_id, workflow)

            # ========================================
            # STEP 2: ROUTING BASED ON STRATEGY
            # ========================================
            if routing_strategy == 'astrosage':
                # AstroSage only
                print(f"Routing strategy: AstroSage only for task {task_id}")
                workflow = await self._handle_astrosage(workflow, task_id)
            elif routing_strategy == 'analysis':
                # Analysis only
                print(f"Routing strategy: Analysis only for task {task_id}")
                workflow = await self._handle_analysis(workflow, task_id)
            elif routing_strategy == 'mixed':
                # Both Analysis and AstroSage
                print(f"Routing strategy: Mixed (Analysis + AstroSage) for task {task_id}")
                workflow = await self._handle_analysis(workflow, task_id)
                workflow = await self._handle_astrosage(workflow, task_id)
            else:
                raise ValueError(f"Unknown routing strategy: {routing_strategy}")

                # workflow = await self._handle_unsucess_intent_classification(workflow, task_id, classification_result)
                
            # ========================================
            # STEP 3: REWRITE AGENT (FINAL RESPONSE)
            # ========================================

            # Step 3: Rewrite Agent
            workflow['current_step'] = 'rewrite'
            workflow['progress'] = '90%'
            await self._add_to_memory(task_id, workflow)

            rewrite_agent = self.agents.get(AgentNames.REWRITE)
            if not rewrite_agent:
                raise ValueError("Rewrite Agent not registered.")

            async with self.shared_llm_semaphore:
                final_response = await rewrite_agents.rewrite_response(
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
            workflow['current_step'] = None
            workflow['status'] = 'completed'
            workflow['progress'] = '100%'
            workflow['completed_at'] = datetime.now()
            await self._add_to_memory(task_id, workflow)

            logger.info(f"Workflow {task_id} completed in {(workflow['completed_at'] - start_time).total_seconds()} seconds.")

        except Exception as e:
            logger.error(f"Error processing workflow {task_id}: {e}")
            workflow['status'] = 'failed'
            workflow['error'] = str(e)
            workflow['completed_at'] = datetime.now()
            await self._add_to_memory(task_id, workflow)

    async def _handle_analysis(self, workflow: dict, task_id: str) -> dict:
        """ 
        Handle Analysis step in the workflow.
        """  
        workflow['current_step'] = 'analysis'
        workflow['progress'] = '50%'
        await self._add_to_memory(task_id, workflow)

        # TODO: Implement analysis handling logic
        logger.info(f"Analysis step placeholder for task {task_id}")

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
