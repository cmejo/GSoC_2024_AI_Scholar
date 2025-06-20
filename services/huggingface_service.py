"""
Hugging Face Model Management Service
Handles model discovery, downloading, and integration with Ollama
"""

import os
import json
import requests
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HFModelInfo:
    """Hugging Face model information"""
    model_id: str
    author: str
    model_name: str
    downloads: int
    likes: int
    tags: List[str]
    pipeline_tag: str
    library_name: str
    created_at: datetime
    last_modified: datetime
    description: str = ""
    size: str = "Unknown"
    license: str = "Unknown"
    compatible_with_ollama: bool = False


@dataclass
class ModelDownloadProgress:
    """Model download progress tracking"""
    model_id: str
    status: str  # 'downloading', 'converting', 'importing', 'completed', 'error'
    progress: float  # 0.0 to 1.0
    message: str
    current_file: str = ""
    total_files: int = 0
    completed_files: int = 0


class HuggingFaceService:
    """Service for managing Hugging Face models and Ollama integration"""
    
    def __init__(self):
        self.hf_api_url = "https://huggingface.co/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Chatbot/1.0'
        })
        
        # Popular LLM models that work well with Ollama
        self.recommended_models = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "facebook/blenderbot-400M-distill",
            "facebook/blenderbot-1B-distill",
            "microsoft/GODEL-v1_1-base-seq2seq",
            "microsoft/GODEL-v1_1-large-seq2seq",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
            "EleutherAI/gpt-j-6b",
            "bigscience/bloom-560m",
            "bigscience/bloom-1b1",
            "bigscience/bloom-3b",
            "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
            "togethercomputer/RedPajama-INCITE-Chat-7B-v1",
            "NousResearch/Llama-2-7b-chat-hf",
            "NousResearch/Llama-2-13b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "codellama/CodeLlama-7b-Instruct-hf",
            "codellama/CodeLlama-13b-Instruct-hf"
        ]
    
    def search_models(
        self, 
        query: str = "", 
        task: str = "text-generation",
        sort: str = "downloads",
        limit: int = 20,
        filter_compatible: bool = True
    ) -> List[HFModelInfo]:
        """Search for models on Hugging Face"""
        
        params = {
            "search": query,
            "filter": f"pipeline_tag:{task}" if task else "",
            "sort": sort,
            "limit": limit
        }
        
        try:
            response = self.session.get(f"{self.hf_api_url}/models", params=params)
            response.raise_for_status()
            models_data = response.json()
            
            models = []
            for model_data in models_data:
                model_info = self._parse_model_info(model_data)
                
                # Filter for Ollama compatibility if requested
                if filter_compatible and not self._is_ollama_compatible(model_info):
                    continue
                
                models.append(model_info)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Optional[HFModelInfo]:
        """Get detailed information about a specific model"""
        try:
            response = self.session.get(f"{self.hf_api_url}/models/{model_id}")
            response.raise_for_status()
            model_data = response.json()
            
            return self._parse_model_info(model_data)
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_id}: {e}")
            return None
    
    def get_recommended_models(self) -> List[HFModelInfo]:
        """Get list of recommended models for chatbot use"""
        models = []
        
        for model_id in self.recommended_models:
            model_info = self.get_model_info(model_id)
            if model_info:
                model_info.compatible_with_ollama = True
                models.append(model_info)
        
        return models
    
    def download_model_for_ollama(
        self, 
        model_id: str, 
        progress_callback=None
    ) -> ModelDownloadProgress:
        """Download and convert HF model for Ollama use"""
        
        progress = ModelDownloadProgress(
            model_id=model_id,
            status="downloading",
            progress=0.0,
            message="Starting download..."
        )
        
        if progress_callback:
            progress_callback(progress)
        
        try:
            # Step 1: Download model from Hugging Face
            progress.message = "Downloading model from Hugging Face..."
            progress.progress = 0.1
            if progress_callback:
                progress_callback(progress)
            
            download_success = self._download_hf_model(model_id, progress_callback)
            if not download_success:
                progress.status = "error"
                progress.message = "Failed to download model"
                return progress
            
            # Step 2: Convert to GGUF format (if needed)
            progress.status = "converting"
            progress.message = "Converting model to Ollama format..."
            progress.progress = 0.5
            if progress_callback:
                progress_callback(progress)
            
            convert_success = self._convert_to_ollama_format(model_id, progress_callback)
            if not convert_success:
                progress.status = "error"
                progress.message = "Failed to convert model"
                return progress
            
            # Step 3: Import into Ollama
            progress.status = "importing"
            progress.message = "Importing model into Ollama..."
            progress.progress = 0.8
            if progress_callback:
                progress_callback(progress)
            
            import_success = self._import_to_ollama(model_id, progress_callback)
            if not import_success:
                progress.status = "error"
                progress.message = "Failed to import model to Ollama"
                return progress
            
            # Step 4: Complete
            progress.status = "completed"
            progress.message = "Model successfully imported to Ollama"
            progress.progress = 1.0
            if progress_callback:
                progress_callback(progress)
            
            return progress
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            progress.status = "error"
            progress.message = f"Error: {str(e)}"
            return progress
    
    def _parse_model_info(self, model_data: Dict) -> HFModelInfo:
        """Parse HF API response into HFModelInfo object"""
        model_id = model_data.get('id', '')
        author, model_name = model_id.split('/', 1) if '/' in model_id else ('', model_id)
        
        return HFModelInfo(
            model_id=model_id,
            author=author,
            model_name=model_name,
            downloads=model_data.get('downloads', 0),
            likes=model_data.get('likes', 0),
            tags=model_data.get('tags', []),
            pipeline_tag=model_data.get('pipeline_tag', ''),
            library_name=model_data.get('library_name', ''),
            created_at=datetime.fromisoformat(
                model_data.get('createdAt', datetime.now().isoformat()).replace('Z', '+00:00')
            ),
            last_modified=datetime.fromisoformat(
                model_data.get('lastModified', datetime.now().isoformat()).replace('Z', '+00:00')
            ),
            description=model_data.get('description', ''),
            license=model_data.get('license', 'Unknown')
        )
    
    def _is_ollama_compatible(self, model_info: HFModelInfo) -> bool:
        """Check if model is compatible with Ollama"""
        # Check if it's a text generation model
        if model_info.pipeline_tag not in ['text-generation', 'conversational']:
            return False
        
        # Check if it's a supported library
        supported_libraries = ['transformers', 'pytorch', 'safetensors']
        if model_info.library_name and model_info.library_name not in supported_libraries:
            return False
        
        # Check model size (avoid very large models)
        size_tags = [tag for tag in model_info.tags if 'size:' in tag.lower()]
        if size_tags:
            # Simple heuristic: avoid models larger than 20B parameters
            for tag in size_tags:
                if any(large in tag.lower() for large in ['30b', '40b', '50b', '70b', '100b']):
                    return False
        
        return True
    
    def _download_hf_model(self, model_id: str, progress_callback=None) -> bool:
        """Download model from Hugging Face using git or huggingface_hub"""
        try:
            # Use huggingface_hub to download
            from huggingface_hub import snapshot_download
            
            cache_dir = os.path.join(os.getcwd(), 'models', 'hf_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Download model
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_dir=os.path.join(cache_dir, model_id.replace('/', '_'))
            )
            
            return True
            
        except ImportError:
            logger.warning("huggingface_hub not installed, using git clone")
            return self._download_with_git(model_id)
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            return False
    
    def _download_with_git(self, model_id: str) -> bool:
        """Download model using git clone"""
        try:
            cache_dir = os.path.join(os.getcwd(), 'models', 'hf_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            model_dir = os.path.join(cache_dir, model_id.replace('/', '_'))
            
            if os.path.exists(model_dir):
                logger.info(f"Model {model_id} already downloaded")
                return True
            
            # Clone repository
            cmd = [
                'git', 'clone', 
                f'https://huggingface.co/{model_id}',
                model_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to git clone model {model_id}: {e}")
            return False
    
    def _convert_to_ollama_format(self, model_id: str, progress_callback=None) -> bool:
        """Convert HF model to Ollama-compatible format"""
        try:
            # This is a simplified conversion process
            # In practice, you might need to use tools like:
            # - llama.cpp for GGUF conversion
            # - Custom conversion scripts
            # - Model-specific conversion tools
            
            logger.info(f"Converting {model_id} to Ollama format")
            
            # For now, we'll assume the model is already in a compatible format
            # or create a simple Modelfile for Ollama
            
            cache_dir = os.path.join(os.getcwd(), 'models', 'hf_cache')
            model_dir = os.path.join(cache_dir, model_id.replace('/', '_'))
            
            # Create Modelfile for Ollama
            modelfile_content = f"""
FROM {model_dir}

TEMPLATE \"\"\"{{{{ if .System }}}}{{{{ .System }}}}{{{{ end }}}}{{{{ if .Prompt }}}}User: {{{{ .Prompt }}}}{{{{ end }}}}
Assistant: \"\"\"

PARAMETER stop "User:"
PARAMETER stop "Assistant:"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
            
            modelfile_path = os.path.join(model_dir, 'Modelfile')
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert model {model_id}: {e}")
            return False
    
    def _import_to_ollama(self, model_id: str, progress_callback=None) -> bool:
        """Import converted model into Ollama"""
        try:
            cache_dir = os.path.join(os.getcwd(), 'models', 'hf_cache')
            model_dir = os.path.join(cache_dir, model_id.replace('/', '_'))
            modelfile_path = os.path.join(model_dir, 'Modelfile')
            
            if not os.path.exists(modelfile_path):
                logger.error(f"Modelfile not found for {model_id}")
                return False
            
            # Create Ollama model name
            ollama_model_name = model_id.replace('/', '_').lower()
            
            # Import model to Ollama
            cmd = ['ollama', 'create', ollama_model_name, '-f', modelfile_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully imported {model_id} as {ollama_model_name}")
                return True
            else:
                logger.error(f"Failed to import model: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to import model {model_id} to Ollama: {e}")
            return False
    
    def get_available_models_for_download(self) -> List[Dict]:
        """Get curated list of models available for download"""
        return [
            {
                "id": "microsoft/DialoGPT-medium",
                "name": "DialoGPT Medium",
                "description": "Conversational AI model, good for general chat",
                "size": "~1.5GB",
                "difficulty": "Easy",
                "recommended": True
            },
            {
                "id": "facebook/blenderbot-400M-distill",
                "name": "BlenderBot 400M",
                "description": "Facebook's conversational AI, lightweight",
                "size": "~800MB",
                "difficulty": "Easy",
                "recommended": True
            },
            {
                "id": "mistralai/Mistral-7B-Instruct-v0.2",
                "name": "Mistral 7B Instruct",
                "description": "High-quality instruction-following model",
                "size": "~14GB",
                "difficulty": "Medium",
                "recommended": True
            },
            {
                "id": "codellama/CodeLlama-7b-Instruct-hf",
                "name": "Code Llama 7B",
                "description": "Specialized for code generation and assistance",
                "size": "~14GB",
                "difficulty": "Medium",
                "recommended": False
            }
        ]


# Global HuggingFace service instance
hf_service = HuggingFaceService()