"""
multi-agent-fits-dev-02/app/services/astrosage/client.py

Main AstroSage service client
"""

import asyncio
import logging
import time
from typing import Dict, Any
import httpx

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.astrosage.models import (
    AstroSageRequest,
    AstroSageResponse,
    AstroSageConnectionError,
    AstroSageTimeoutError,
    AstroSageInvalidResponseError
)
from app.services.astrosage.prompt_builder import PromptBuilder
from app.services.astrosage.conversation_manager import ConversationManager
from app.services.astrosage.expertise_adapter import ExpertiseAdapter

logger = logging.getLogger(__name__)

class AstroSageClient:
    """
    Client for interacting with AstroSage LLM service
    """

    def __init__(self):
        self.base_url = settings.astrosage_base_url
        self.model = settings.astrosage_model
        self.timeout = settings.astrosage_timeout
        self.max_retries = settings.astrosage_max_retries
        self.retry_delay = settings.astrosage_retry_delay

        logger.info(
            f"AstroSage client initialized: {self.base_url}, "
            f"timeout={self.timeout}s, max_retries={self.max_retries}"
        )

    async def query(
        self,
        request: AstroSageRequest,
        db_session: AsyncSession
    ) -> AstroSageResponse:
        """
        Main entry point: Query AstroSage LLM
        
        Args:
            request: AstroSageRequest object
            db_session: Database session for retrieving/saving conversations
        
        Returns:
            AstroSageResponse object
        
        Raises:
            AstroSageConnectionError: Cannot connect to service
            AstroSageTimeoutError: Request timed out
            AstroSageInvalidResponseError: Invalid response from service
        """

        start_time = time.time()
        
        logger.info(
            f"Processing AstroSage query: session={request.session_id}, "
            f"expertise={request.expertise_level.value}"
        )

        try:
            # ========================================
            # STEP 1: Get user expertise level if not provided
            # ========================================
            if request.expertise_level is None:
                request.expertise_level = await ExpertiseAdapter.get_user_expertise(
                    request.user_id,
                    db_session
                )

            # ========================================
            # STEP 2: Get conversation history if not provided
            # ========================================
            if request.conversation_history is None:
                request.conversation_history = await ConversationManager.get_last_conversations(
                    request.session_id,
                    db_session,
                    limit=settings.conversation_history_limit
                )

            # ========================================
            # STEP 3: Build prompt
            # ========================================
            messages = PromptBuilder.build_full_prompt(request)
            
            # ========================================
            # STEP 4: Get LLM config for expertise level
            # ========================================
            llm_config = ExpertiseAdapter.get_llm_config(request.expertise_level)

            # Override with request parameters if provided
            if request.temperature is not None:
                llm_config.temperature = request.temperature
            if request.max_tokens is not None:
                llm_config.max_tokens = request.max_tokens
            if request.top_p is not None:
                llm_config.top_p = request.top_p

            # ========================================
            # STEP 5: Call LLM API with retry
            # ========================================
            response_data = await self._call_llm_with_retry(messages, llm_config)

            # ========================================
            # STEP 6: Parse response
            # ========================================
            response_content = self._extract_response_content(response_data)
            tokens_used = self._extract_tokens_used(response_data)

            # ========================================
            # STEP 7: Save conversation to database
            # ========================================
            await ConversationManager.save_conversation(
                session_id=request.session_id,
                user_message=request.user_query,
                assistant_message=response_content,
                db_session=db_session
            )
            
            # ========================================
            # STEP 8: Build and return response
            # ========================================
            response_time = time.time() - start_time
            
            astrosage_response = AstroSageResponse(
                content=response_content,
                model_used=self.model,
                tokens_used=tokens_used,
                response_time=response_time,
                success=True
            )
            
            logger.info(
                f"AstroSage query completed: "
                f"session={request.session_id}, "
                f"time={response_time:.2f}s, "
                f"tokens={tokens_used}"
            )
            
            return astrosage_response

        except (AstroSageConnectionError, AstroSageTimeoutError, AstroSageInvalidResponseError) as e:
            # Known errors - re-raise
            logger.error(f"AstroSage query failed: {e}")
            raise
            
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error in AstroSage query: {e}", exc_info=True)
            
            response_time = time.time() - start_time
            
            return AstroSageResponse(
                content="I apologize, but I encountered an error processing your request. Please try again.",
                model_used=self.model,
                response_time=response_time,
                success=False,
                error=str(e)
            )
        
    async def _call_llm_with_retry(
            self,
            messages: list,
            llm_config
    ) -> dict:
        """
        Call LLM API with retry mechanism
        
        Args:
            messages: List of message dictionaries
            llm_config: LLMConfig object
        
        Returns:
            API response dictionary
        
        Raises:
            AstroSageConnectionError: Cannot connect after retries
            AstroSageTimeoutError: Request timed out
            AstroSageInvalidResponseError: Invalid response
        """

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"LLM API call attempt {attempt + 1}/{self.max_retries + 1}")
                
                response = await self._call_llm_api(messages, llm_config)
                
                # Validate response
                if self._validate_response(response):
                    return response
                else:
                    raise AstroSageInvalidResponseError("Invalid response structure")
                
            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"LLM API timeout on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise AstroSageTimeoutError(
                        f"Request timed out after {self.max_retries + 1} attempts"
                    )
            
            except httpx.ConnectError as e:
                last_error = e
                logger.error(f"Cannot connect to AstroSage service: {e}")
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise AstroSageConnectionError(
                        f"Cannot connect to AstroSage service at {self.base_url}"
                    )
            
            except AstroSageInvalidResponseError as e:
                last_error = e
                logger.error(f"Invalid response from LLM: {e}")
                
                if attempt < self.max_retries:
                    logger.info("Retrying with same prompt...")
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
            
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error calling LLM: {e}", exc_info=True)
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise
        
        # Should not reach here, but just in case
        raise Exception(f"Failed after {self.max_retries + 1} attempts: {last_error}")
    
    async def _call_llm_api(
        self,
        messages: list,
        llm_config
    ) -> dict:
        """
        Make actual HTTP call to LLM API
        
        Args:
            messages: List of message dictionaries
            llm_config: LLMConfig object
        
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            **llm_config.to_api_dict()
        }
        
        logger.debug(f"Calling LLM API: {url}")
        logger.debug(f"Payload: model={self.model}, messages={len(messages)}, config={llm_config.to_api_dict()}")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            return response.json()
    
    def _validate_response(self, response: dict) -> bool:
        """
        Validate LLM API response structure
        
        Args:
            response: API response dictionary
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check basic structure
            if 'choices' not in response:
                logger.error("Response missing 'choices' field")
                return False
            
            if not response['choices']:
                logger.error("Response 'choices' is empty")
                return False
            
            choice = response['choices'][0]
            
            if 'message' not in choice:
                logger.error("Response choice missing 'message' field")
                return False
            
            if 'content' not in choice['message']:
                logger.error("Response message missing 'content' field")
                return False
            
            content = choice['message']['content']
            
            if not content or not isinstance(content, str):
                logger.error("Response content is empty or not a string")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return False
    
    def _extract_response_content(self, response: dict) -> str:
        """
        Extract content from API response
        
        Args:
            response: API response dictionary
        
        Returns:
            Response content string
        """
        try:
            return response['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting content: {e}")
            return "Error: Could not extract response content"
    
    def _extract_tokens_used(self, response: dict) -> int:
        """
        Extract token usage from API response
        
        Args:
            response: API response dictionary
        
        Returns:
            Number of tokens used (or 0 if not available)
        """
        try:
            usage = response.get('usage', {})
            return usage.get('total_tokens', 0)
        except Exception as e:
            logger.debug(f"Could not extract token count: {e}")
            return 0
    