"""
Enhanced Chat Service with Advanced LLM Features
Handles conversation management, context, streaming, and model switching
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, Generator, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from enum import Enum

from services.ollama_service import ollama_service, ChatResponse
from models import ChatSession, ChatMessage, User, db

logger = logging.getLogger(__name__)


class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ConversationContext:
    """Conversation context management"""
    session_id: int
    messages: List[Dict]
    max_context_length: int = 4000
    context_window: int = 10  # Number of recent messages to keep
    system_prompt: str = ""
    model_name: str = ""
    parameters: Dict = None


@dataclass
class StreamingResponse:
    """Streaming response structure"""
    content: str
    is_complete: bool
    metadata: Dict
    error: Optional[str] = None


class ChatService:
    """Enhanced chat service with advanced LLM features"""
    
    def __init__(self):
        self.active_streams = {}  # Track active streaming sessions
        self.conversation_contexts = {}  # Cache conversation contexts
        
        # Default model parameters
        self.default_parameters = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2048,
            "repeat_penalty": 1.1,
            "stop": ["User:", "Human:", "\n\nUser:", "\n\nHuman:"]
        }
        
        # System prompts for different conversation types
        self.system_prompts = {
            "general": "You are a helpful, harmless, and honest AI assistant. Provide clear, accurate, and helpful responses.",
            "creative": "You are a creative AI assistant. Help users with creative writing, brainstorming, and imaginative tasks.",
            "technical": "You are a technical AI assistant. Provide accurate, detailed technical information and help with programming and technical problems.",
            "casual": "You are a friendly AI assistant. Have casual, engaging conversations while being helpful and informative.",
            "professional": "You are a professional AI assistant. Provide formal, well-structured responses suitable for business contexts."
        }
    
    def get_conversation_context(self, session_id: int, user_id: int) -> ConversationContext:
        """Get or create conversation context for a session"""
        
        if session_id in self.conversation_contexts:
            return self.conversation_contexts[session_id]
        
        # Load conversation history from database
        session = ChatSession.query.filter_by(id=session_id, user_id=user_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Get recent messages
        messages = ChatMessage.query.filter_by(session_id=session_id)\
                                  .order_by(ChatMessage.timestamp.desc())\
                                  .limit(20).all()
        
        # Convert to conversation format
        conversation_messages = []
        for msg in reversed(messages):  # Reverse to get chronological order
            conversation_messages.append({
                "role": msg.message_type,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            })
        
        # Create context
        context = ConversationContext(
            session_id=session_id,
            messages=conversation_messages,
            system_prompt=self.system_prompts.get("general", ""),
            model_name=session.model_name if hasattr(session, 'model_name') else "llama2"
        )
        
        self.conversation_contexts[session_id] = context
        return context
    
    def update_conversation_context(
        self, 
        session_id: int, 
        message: str, 
        role: str,
        metadata: Dict = None
    ):
        """Update conversation context with new message"""
        
        if session_id not in self.conversation_contexts:
            return
        
        context = self.conversation_contexts[session_id]
        
        # Add new message
        new_message = {
            "role": role,
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            new_message["metadata"] = metadata
        
        context.messages.append(new_message)
        
        # Trim context if too long
        if len(context.messages) > context.context_window:
            # Keep system message and recent messages
            system_messages = [msg for msg in context.messages if msg["role"] == "system"]
            recent_messages = context.messages[-(context.context_window-len(system_messages)):]
            context.messages = system_messages + recent_messages
    
    def generate_response(
        self,
        session_id: int,
        user_id: int,
        message: str,
        model: str = None,
        parameters: Dict = None,
        stream: bool = False
    ) -> Generator[StreamingResponse, None, None]:
        """Generate response with advanced context management"""
        
        try:
            # Get conversation context
            context = self.get_conversation_context(session_id, user_id)
            
            # Update context with user message
            self.update_conversation_context(session_id, message, "user")
            
            # Use specified model or default
            model_name = model or context.model_name or "llama2"
            
            # Merge parameters
            model_params = {**self.default_parameters}
            if parameters:
                model_params.update(parameters)
            if context.parameters:
                model_params.update(context.parameters)
            
            # Prepare messages for Ollama
            ollama_messages = self._prepare_ollama_messages(context)
            
            # Generate response
            response_content = ""
            start_time = time.time()
            
            for chunk in ollama_service.chat(
                model=model_name,
                messages=ollama_messages,
                options=model_params,
                stream=stream
            ):
                response_content += chunk.content
                
                # Create streaming response
                streaming_response = StreamingResponse(
                    content=chunk.content,
                    is_complete=chunk.done,
                    metadata={
                        "model": model_name,
                        "session_id": session_id,
                        "timestamp": chunk.created_at.isoformat(),
                        "total_duration": chunk.total_duration,
                        "eval_count": chunk.eval_count,
                        "eval_duration": chunk.eval_duration
                    }
                )
                
                yield streaming_response
                
                if chunk.done:
                    break
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Save messages to database
            self._save_messages_to_db(
                session_id=session_id,
                user_id=user_id,
                user_message=message,
                bot_response=response_content,
                model_name=model_name,
                response_time=response_time,
                metadata={
                    "parameters": model_params,
                    "eval_count": chunk.eval_count if 'chunk' in locals() else None,
                    "eval_duration": chunk.eval_duration if 'chunk' in locals() else None
                }
            )
            
            # Update context with bot response
            self.update_conversation_context(
                session_id, 
                response_content, 
                "assistant",
                {"model": model_name, "response_time": response_time}
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield StreamingResponse(
                content="",
                is_complete=True,
                metadata={"session_id": session_id},
                error=str(e)
            )
    
    def _prepare_ollama_messages(self, context: ConversationContext) -> List[Dict]:
        """Prepare messages for Ollama chat API"""
        messages = []
        
        # Add system prompt if available
        if context.system_prompt:
            messages.append({
                "role": "system",
                "content": context.system_prompt
            })
        
        # Add conversation history
        for msg in context.messages:
            # Map message types
            role = msg["role"]
            if role == "user":
                role = "user"
            elif role in ["bot", "assistant"]:
                role = "assistant"
            elif role == "system":
                role = "system"
            
            messages.append({
                "role": role,
                "content": msg["content"]
            })
        
        return messages
    
    def _save_messages_to_db(
        self,
        session_id: int,
        user_id: int,
        user_message: str,
        bot_response: str,
        model_name: str,
        response_time: float,
        metadata: Dict = None
    ):
        """Save conversation messages to database"""
        
        try:
            # Save user message
            user_msg = ChatMessage(
                session_id=session_id,
                user_id=user_id,
                message_type='user',
                content=user_message,
                timestamp=datetime.now()
            )
            db.session.add(user_msg)
            
            # Save bot response
            bot_msg = ChatMessage(
                session_id=session_id,
                user_id=user_id,
                message_type='bot',
                content=bot_response,
                timestamp=datetime.now(),
                model_used=model_name,
                response_time=response_time,
                token_count=metadata.get('eval_count') if metadata else None
            )
            db.session.add(bot_msg)
            
            # Update session timestamp
            session = ChatSession.query.get(session_id)
            if session:
                session.updated_at = datetime.now()
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Failed to save messages to database: {e}")
            db.session.rollback()
    
    def switch_model(self, session_id: int, new_model: str) -> bool:
        """Switch model for a conversation session"""
        try:
            # Update context
            if session_id in self.conversation_contexts:
                self.conversation_contexts[session_id].model_name = new_model
            
            # Update database session
            session = ChatSession.query.get(session_id)
            if session:
                session.model_name = new_model
                db.session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False
    
    def update_system_prompt(self, session_id: int, prompt_type: str, custom_prompt: str = None) -> bool:
        """Update system prompt for a session"""
        try:
            if session_id not in self.conversation_contexts:
                return False
            
            context = self.conversation_contexts[session_id]
            
            if custom_prompt:
                context.system_prompt = custom_prompt
            elif prompt_type in self.system_prompts:
                context.system_prompt = self.system_prompts[prompt_type]
            else:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update system prompt: {e}")
            return False
    
    def update_model_parameters(self, session_id: int, parameters: Dict) -> bool:
        """Update model parameters for a session"""
        try:
            if session_id not in self.conversation_contexts:
                return False
            
            context = self.conversation_contexts[session_id]
            context.parameters = {**self.default_parameters, **parameters}
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model parameters: {e}")
            return False
    
    def get_conversation_summary(self, session_id: int) -> Dict:
        """Get conversation summary and statistics"""
        try:
            if session_id not in self.conversation_contexts:
                return {}
            
            context = self.conversation_contexts[session_id]
            
            # Calculate statistics
            total_messages = len(context.messages)
            user_messages = len([msg for msg in context.messages if msg["role"] == "user"])
            assistant_messages = len([msg for msg in context.messages if msg["role"] == "assistant"])
            
            # Get recent activity
            recent_messages = context.messages[-5:] if context.messages else []
            
            return {
                "session_id": session_id,
                "total_messages": total_messages,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "current_model": context.model_name,
                "system_prompt_type": self._get_prompt_type(context.system_prompt),
                "recent_activity": recent_messages,
                "context_length": len(str(context.messages)),
                "parameters": context.parameters or self.default_parameters
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation summary: {e}")
            return {}
    
    def _get_prompt_type(self, prompt: str) -> str:
        """Determine prompt type from system prompt"""
        for prompt_type, system_prompt in self.system_prompts.items():
            if prompt == system_prompt:
                return prompt_type
        return "custom"
    
    def clear_conversation_context(self, session_id: int):
        """Clear conversation context from memory"""
        if session_id in self.conversation_contexts:
            del self.conversation_contexts[session_id]
    
    def get_available_system_prompts(self) -> Dict[str, str]:
        """Get available system prompt types"""
        return self.system_prompts.copy()
    
    def get_default_parameters(self) -> Dict:
        """Get default model parameters"""
        return self.default_parameters.copy()


# Global chat service instance
chat_service = ChatService()