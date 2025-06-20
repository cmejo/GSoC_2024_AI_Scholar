"""
Ollama Service for Local LLM Management
Handles Ollama integration, model management, and LLM interactions
"""

import os
import json
import time
import asyncio
import requests
import subprocess
from typing import Dict, List, Optional, Generator, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information structure"""
    name: str
    size: str
    modified: datetime
    digest: str
    details: Dict
    parameters: Dict = None
    template: str = None
    system: str = None


@dataclass
class ChatResponse:
    """Chat response structure"""
    content: str
    model: str
    created_at: datetime
    done: bool
    total_duration: int = None
    load_duration: int = None
    prompt_eval_count: int = None
    prompt_eval_duration: int = None
    eval_count: int = None
    eval_duration: int = None


class OllamaService:
    """Enhanced Ollama service for local LLM management"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.session = requests.Session()
        self.session.timeout = 30
        self._models_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes
        
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama service not available: {e}")
            return False
    
    def get_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """Get list of available models"""
        try:
            # Check cache
            if not force_refresh and self._is_cache_valid():
                return list(self._models_cache.values())
            
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get('models', []):
                model_info = ModelInfo(
                    name=model_data['name'],
                    size=model_data.get('size', 'Unknown'),
                    modified=datetime.fromisoformat(model_data['modified_at'].replace('Z', '+00:00')),
                    digest=model_data.get('digest', ''),
                    details=model_data.get('details', {}),
                    parameters=model_data.get('parameters', {}),
                    template=model_data.get('template'),
                    system=model_data.get('system')
                )
                models.append(model_info)
                self._models_cache[model_info.name] = model_info
            
            self._cache_timestamp = time.time()
            return models
            
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": model_name}
            )
            response.raise_for_status()
            
            data = response.json()
            return ModelInfo(
                name=model_name,
                size=data.get('size', 'Unknown'),
                modified=datetime.now(),
                digest=data.get('digest', ''),
                details=data.get('details', {}),
                parameters=data.get('parameters', {}),
                template=data.get('template'),
                system=data.get('system')
            )
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
    
    def pull_model(self, model_name: str) -> Generator[Dict, None, None]:
        """Pull a model from Ollama registry with progress tracking"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        yield data
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            yield {"error": str(e)}
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/delete",
                json={"name": model_name}
            )
            response.raise_for_status()
            
            # Remove from cache
            if model_name in self._models_cache:
                del self._models_cache[model_name]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def generate_response(
        self,
        model: str,
        prompt: str,
        system: str = None,
        context: List[Dict] = None,
        stream: bool = False,
        **kwargs
    ) -> Generator[ChatResponse, None, None] if stream else ChatResponse:
        """Generate response from model"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                **kwargs
            }
            
            if system:
                payload["system"] = system
            
            if context:
                payload["context"] = context
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._stream_response(response)
            else:
                data = response.json()
                return self._parse_response(data)
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            if stream:
                yield ChatResponse(
                    content=f"Error: {str(e)}",
                    model=model,
                    created_at=datetime.now(),
                    done=True
                )
            else:
                return ChatResponse(
                    content=f"Error: {str(e)}",
                    model=model,
                    created_at=datetime.now(),
                    done=True
                )
    
    def chat(
        self,
        model: str,
        messages: List[Dict],
        stream: bool = False,
        **kwargs
    ) -> Generator[ChatResponse, None, None] if stream else ChatResponse:
        """Chat with model using conversation format"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                **kwargs
            }
            
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._stream_response(response)
            else:
                data = response.json()
                return self._parse_response(data)
                
        except Exception as e:
            logger.error(f"Failed to chat: {e}")
            if stream:
                yield ChatResponse(
                    content=f"Error: {str(e)}",
                    model=model,
                    created_at=datetime.now(),
                    done=True
                )
            else:
                return ChatResponse(
                    content=f"Error: {str(e)}",
                    model=model,
                    created_at=datetime.now(),
                    done=True
                )
    
    def _stream_response(self, response) -> Generator[ChatResponse, None, None]:
        """Parse streaming response"""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    yield self._parse_response(data)
                except json.JSONDecodeError:
                    continue
    
    def _parse_response(self, data: Dict) -> ChatResponse:
        """Parse response data into ChatResponse object"""
        return ChatResponse(
            content=data.get('response', data.get('message', {}).get('content', '')),
            model=data.get('model', ''),
            created_at=datetime.now(),
            done=data.get('done', False),
            total_duration=data.get('total_duration'),
            load_duration=data.get('load_duration'),
            prompt_eval_count=data.get('prompt_eval_count'),
            prompt_eval_duration=data.get('prompt_eval_duration'),
            eval_count=data.get('eval_count'),
            eval_duration=data.get('eval_duration')
        )
    
    def _is_cache_valid(self) -> bool:
        """Check if models cache is still valid"""
        if not self._cache_timestamp:
            return False
        return time.time() - self._cache_timestamp < self._cache_ttl
    
    def create_model(self, name: str, modelfile: str) -> bool:
        """Create a custom model from Modelfile"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/create",
                json={
                    "name": name,
                    "modelfile": modelfile
                }
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create model {name}: {e}")
            return False
    
    def copy_model(self, source: str, destination: str) -> bool:
        """Copy a model"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/copy",
                json={
                    "source": source,
                    "destination": destination
                }
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy model {source} to {destination}: {e}")
            return False
    
    def get_embeddings(self, model: str, prompt: str) -> Optional[List[float]]:
        """Get embeddings for text"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": prompt
                }
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('embedding')
            
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return None
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.session = requests.Session()
        self.session.timeout = 30
        self.models_cache = {}
        self.last_cache_update = None
        self.cache_ttl = 300  # 5 minutes
        
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama service not available: {e}")
            return False
    
    def get_version(self) -> Dict:
        """Get Ollama version information"""
        try:
            response = self.session.get(f"{self.base_url}/api/version")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get Ollama version: {e}")
            return {}
    
    def list_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """List available models with caching"""
        now = time.time()
        
        # Use cache if available and not expired
        if (not force_refresh and 
            self.last_cache_update and 
            now - self.last_cache_update < self.cache_ttl and 
            self.models_cache):
            return list(self.models_cache.values())
        
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model_data in data.get('models', []):
                model = ModelInfo(
                    name=model_data['name'],
                    size=model_data.get('size', 'Unknown'),
                    modified=datetime.fromisoformat(model_data['modified_at'].replace('Z', '+00:00')),
                    digest=model_data.get('digest', ''),
                    details=model_data.get('details', {}),
                    parameters=model_data.get('parameters', {}),
                    template=model_data.get('template', ''),
                    system=model_data.get('system', '')
                )
                models.append(model)
                self.models_cache[model.name] = model
            
            self.last_cache_update = now
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": model_name}
            )
            response.raise_for_status()
            data = response.json()
            
            return ModelInfo(
                name=model_name,
                size=data.get('size', 'Unknown'),
                modified=datetime.fromisoformat(data['modified_at'].replace('Z', '+00:00')),
                digest=data.get('digest', ''),
                details=data.get('details', {}),
                parameters=data.get('parameters', {}),
                template=data.get('template', ''),
                system=data.get('system', '')
            )
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
    
    def pull_model(self, model_name: str) -> Generator[Dict, None, None]:
        """Pull/download a model with progress tracking"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        yield data
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            yield {"error": str(e)}
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/delete",
                json={"name": model_name}
            )
            response.raise_for_status()
            
            # Remove from cache
            if model_name in self.models_cache:
                del self.models_cache[model_name]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def generate_response(
        self, 
        model: str, 
        prompt: str, 
        system: str = None,
        context: List[Dict] = None,
        options: Dict = None,
        stream: bool = False
    ) -> Generator[ChatResponse, None, None]:
        """Generate response from model with streaming support"""
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options or {}
        }
        
        if system:
            payload["system"] = system
        
        if context:
            # Convert context to Ollama format
            payload["context"] = self._format_context(context)
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            yield self._parse_response(data)
                        except json.JSONDecodeError:
                            continue
            else:
                data = response.json()
                yield self._parse_response(data)
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            yield ChatResponse(
                content=f"Error: {str(e)}",
                model=model,
                created_at=datetime.now(),
                done=True
            )
    
    def chat(
        self,
        model: str,
        messages: List[Dict],
        options: Dict = None,
        stream: bool = False
    ) -> Generator[ChatResponse, None, None]:
        """Chat with model using conversation format"""
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options or {}
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            yield self._parse_chat_response(data)
                        except json.JSONDecodeError:
                            continue
            else:
                data = response.json()
                yield self._parse_chat_response(data)
                
        except Exception as e:
            logger.error(f"Failed to chat: {e}")
            yield ChatResponse(
                content=f"Error: {str(e)}",
                model=model,
                created_at=datetime.now(),
                done=True
            )
    
    def _format_context(self, context: List[Dict]) -> List[int]:
        """Format conversation context for Ollama"""
        # This would need to be implemented based on the specific model's context format
        # For now, return empty list
        return []
    
    def _parse_response(self, data: Dict) -> ChatResponse:
        """Parse Ollama response into ChatResponse object"""
        return ChatResponse(
            content=data.get('response', ''),
            model=data.get('model', ''),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            done=data.get('done', False),
            total_duration=data.get('total_duration'),
            load_duration=data.get('load_duration'),
            prompt_eval_count=data.get('prompt_eval_count'),
            prompt_eval_duration=data.get('prompt_eval_duration'),
            eval_count=data.get('eval_count'),
            eval_duration=data.get('eval_duration')
        )
    
    def _parse_chat_response(self, data: Dict) -> ChatResponse:
        """Parse Ollama chat response into ChatResponse object"""
        message = data.get('message', {})
        return ChatResponse(
            content=message.get('content', ''),
            model=data.get('model', ''),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            done=data.get('done', False),
            total_duration=data.get('total_duration'),
            load_duration=data.get('load_duration'),
            prompt_eval_count=data.get('prompt_eval_count'),
            prompt_eval_duration=data.get('prompt_eval_duration'),
            eval_count=data.get('eval_count'),
            eval_duration=data.get('eval_duration')
        )
    
    def get_model_parameters(self, model_name: str) -> Dict:
        """Get model parameters and configuration"""
        model_info = self.get_model_info(model_name)
        if model_info:
            return model_info.parameters
        return {}
    
    def update_model_parameters(self, model_name: str, parameters: Dict) -> bool:
        """Update model parameters (create modelfile)"""
        try:
            # This would create a custom modelfile with updated parameters
            # Implementation depends on specific requirements
            logger.info(f"Updating parameters for {model_name}: {parameters}")
            return True
        except Exception as e:
            logger.error(f"Failed to update model parameters: {e}")
            return False


# Global Ollama service instance
ollama_service = OllamaService()