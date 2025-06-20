"""
Fine-tuning Service
Handles model fine-tuning, training data management, and model customization
"""

import os
import json
import time
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import tempfile

from services.ollama_service import ollama_service

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers datasets")


@dataclass
class TrainingExample:
    """Training example structure"""
    input_text: str
    output_text: str
    metadata: Optional[Dict] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class TrainingDataset:
    """Training dataset structure"""
    name: str
    examples: List[TrainingExample]
    description: str = ""
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class FineTuningConfig:
    """Fine-tuning configuration"""
    base_model: str
    dataset_name: str
    output_model_name: str
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    temperature: float = 0.7
    system_prompt: str = ""
    custom_parameters: Dict = None


@dataclass
class FineTuningJob:
    """Fine-tuning job tracking"""
    job_id: str
    config: FineTuningConfig
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    eval_loss: float = 0.0
    learning_rate: float = 0.0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    logs: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.logs is None:
            self.logs = []


class TrainingDataManager:
    """Manage training datasets"""
    
    def __init__(self, data_dir: str = "./training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.datasets = {}
        self._load_datasets()
    
    def _load_datasets(self):
        """Load existing datasets from disk"""
        try:
            for dataset_file in self.data_dir.glob("*.json"):
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    examples = [
                        TrainingExample(
                            input_text=ex['input_text'],
                            output_text=ex['output_text'],
                            metadata=ex.get('metadata'),
                            created_at=datetime.fromisoformat(ex['created_at'])
                        )
                        for ex in data['examples']
                    ]
                    
                    dataset = TrainingDataset(
                        name=data['name'],
                        examples=examples,
                        description=data.get('description', ''),
                        created_at=datetime.fromisoformat(data['created_at']),
                        updated_at=datetime.fromisoformat(data['updated_at'])
                    )
                    
                    self.datasets[dataset.name] = dataset
            
            logger.info(f"Loaded {len(self.datasets)} training datasets")
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
    
    def create_dataset(self, name: str, description: str = "") -> bool:
        """Create a new training dataset"""
        try:
            if name in self.datasets:
                logger.warning(f"Dataset '{name}' already exists")
                return False
            
            dataset = TrainingDataset(
                name=name,
                examples=[],
                description=description
            )
            
            self.datasets[name] = dataset
            self._save_dataset(dataset)
            
            logger.info(f"Created dataset '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            return False
    
    def add_examples(self, dataset_name: str, examples: List[TrainingExample]) -> bool:
        """Add training examples to a dataset"""
        try:
            if dataset_name not in self.datasets:
                logger.error(f"Dataset '{dataset_name}' not found")
                return False
            
            dataset = self.datasets[dataset_name]
            dataset.examples.extend(examples)
            dataset.updated_at = datetime.now()
            
            self._save_dataset(dataset)
            
            logger.info(f"Added {len(examples)} examples to dataset '{dataset_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding examples: {e}")
            return False
    
    def import_from_file(self, dataset_name: str, file_path: str, format_type: str = "jsonl") -> bool:
        """Import training data from file"""
        try:
            if dataset_name not in self.datasets:
                if not self.create_dataset(dataset_name):
                    return False
            
            examples = []
            
            if format_type == "jsonl":
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        example = TrainingExample(
                            input_text=data['input'],
                            output_text=data['output'],
                            metadata=data.get('metadata')
                        )
                        examples.append(example)
            
            elif format_type == "csv":
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        example = TrainingExample(
                            input_text=row['input'],
                            output_text=row['output'],
                            metadata={k: v for k, v in row.items() if k not in ['input', 'output']}
                        )
                        examples.append(example)
            
            elif format_type == "conversational":
                # Format: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for conversation in data.get('conversations', []):
                        if len(conversation) >= 2:
                            human_msg = next((msg['value'] for msg in conversation if msg['from'] == 'human'), '')
                            gpt_msg = next((msg['value'] for msg in conversation if msg['from'] == 'gpt'), '')
                            
                            if human_msg and gpt_msg:
                                example = TrainingExample(
                                    input_text=human_msg,
                                    output_text=gpt_msg
                                )
                                examples.append(example)
            
            return self.add_examples(dataset_name, examples)
            
        except Exception as e:
            logger.error(f"Error importing from file: {e}")
            return False
    
    def export_dataset(self, dataset_name: str, output_path: str, format_type: str = "jsonl") -> bool:
        """Export dataset to file"""
        try:
            if dataset_name not in self.datasets:
                logger.error(f"Dataset '{dataset_name}' not found")
                return False
            
            dataset = self.datasets[dataset_name]
            
            if format_type == "jsonl":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for example in dataset.examples:
                        data = {
                            'input': example.input_text,
                            'output': example.output_text,
                            'metadata': example.metadata
                        }
                        f.write(json.dumps(data) + '\n')
            
            elif format_type == "alpaca":
                # Alpaca format for training
                alpaca_data = []
                for example in dataset.examples:
                    alpaca_data.append({
                        'instruction': example.input_text,
                        'input': '',
                        'output': example.output_text
                    })
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported dataset '{dataset_name}' to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting dataset: {e}")
            return False
    
    def _save_dataset(self, dataset: TrainingDataset):
        """Save dataset to disk"""
        try:
            dataset_file = self.data_dir / f"{dataset.name}.json"
            
            data = {
                'name': dataset.name,
                'description': dataset.description,
                'created_at': dataset.created_at.isoformat(),
                'updated_at': dataset.updated_at.isoformat(),
                'examples': [
                    {
                        'input_text': ex.input_text,
                        'output_text': ex.output_text,
                        'metadata': ex.metadata,
                        'created_at': ex.created_at.isoformat()
                    }
                    for ex in dataset.examples
                ]
            }
            
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
    
    def get_dataset(self, name: str) -> Optional[TrainingDataset]:
        """Get a training dataset"""
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[Dict]:
        """List all datasets"""
        return [
            {
                'name': dataset.name,
                'description': dataset.description,
                'example_count': len(dataset.examples),
                'created_at': dataset.created_at.isoformat(),
                'updated_at': dataset.updated_at.isoformat()
            }
            for dataset in self.datasets.values()
        ]
    
    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset"""
        try:
            if name not in self.datasets:
                return False
            
            # Remove from memory
            del self.datasets[name]
            
            # Remove file
            dataset_file = self.data_dir / f"{name}.json"
            if dataset_file.exists():
                dataset_file.unlink()
            
            logger.info(f"Deleted dataset '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting dataset: {e}")
            return False


class ModelFileGenerator:
    """Generate Ollama Modelfiles for fine-tuned models"""
    
    @staticmethod
    def generate_modelfile(config: FineTuningConfig, model_path: str) -> str:
        """Generate Modelfile content"""
        
        # Base template
        modelfile_content = f"""FROM {model_path}

# Fine-tuned model: {config.output_model_name}
# Base model: {config.base_model}
# Dataset: {config.dataset_name}

"""
        
        # Add system prompt if provided
        if config.system_prompt:
            modelfile_content += f'SYSTEM """{config.system_prompt}"""\n\n'
        
        # Add parameters
        modelfile_content += f"""PARAMETER temperature {config.temperature}
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

"""
        
        # Add custom parameters
        if config.custom_parameters:
            for param, value in config.custom_parameters.items():
                modelfile_content += f"PARAMETER {param} {value}\n"
        
        # Add template
        modelfile_content += '''
TEMPLATE """{{ if .System }}{{ .System }}{{ end }}{{ if .Prompt }}User: {{ .Prompt }}{{ end }}
Assistant: """

'''
        
        return modelfile_content


class FineTuningEngine:
    """Handle model fine-tuning operations"""
    
    def __init__(self, output_dir: str = "./fine_tuned_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.active_jobs = {}
        self.completed_jobs = {}
        self.job_counter = 0
    
    def start_fine_tuning(self, config: FineTuningConfig, dataset: TrainingDataset) -> str:
        """Start a fine-tuning job"""
        try:
            # Generate job ID
            self.job_counter += 1
            job_id = f"ft_{int(time.time())}_{self.job_counter}"
            
            # Create job
            job = FineTuningJob(
                job_id=job_id,
                config=config,
                status='pending',
                total_steps=self._estimate_steps(dataset, config)
            )
            
            self.active_jobs[job_id] = job
            
            # Start training in background (simplified)
            self._run_fine_tuning(job, dataset)
            
            logger.info(f"Started fine-tuning job {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error starting fine-tuning: {e}")
            return ""
    
    def _estimate_steps(self, dataset: TrainingDataset, config: FineTuningConfig) -> int:
        """Estimate total training steps"""
        examples_per_epoch = len(dataset.examples)
        steps_per_epoch = examples_per_epoch // config.batch_size
        return steps_per_epoch * config.num_epochs
    
    def _run_fine_tuning(self, job: FineTuningJob, dataset: TrainingDataset):
        """Run the fine-tuning process (simplified implementation)"""
        try:
            job.status = 'running'
            job.started_at = datetime.now()
            
            # Create output directory for this job
            job_output_dir = self.output_dir / job.job_id
            job_output_dir.mkdir(exist_ok=True)
            
            # For demonstration, we'll create a simple Ollama model
            # In a real implementation, this would involve actual training
            
            # Simulate training progress
            for step in range(job.total_steps):
                time.sleep(0.1)  # Simulate training time
                
                job.current_step = step + 1
                job.progress = (step + 1) / job.total_steps
                job.loss = 2.5 - (step / job.total_steps) * 1.5  # Simulated decreasing loss
                
                if step % 10 == 0:
                    job.logs.append(f"Step {step + 1}/{job.total_steps}, Loss: {job.loss:.4f}")
            
            # Create Modelfile
            modelfile_content = ModelFileGenerator.generate_modelfile(
                job.config, 
                job.config.base_model
            )
            
            modelfile_path = job_output_dir / "Modelfile"
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
            
            # Create model in Ollama
            success = self._create_ollama_model(job.config.output_model_name, modelfile_path)
            
            if success:
                job.status = 'completed'
                job.output_path = str(modelfile_path)
                job.logs.append("Model successfully created in Ollama")
            else:
                job.status = 'failed'
                job.error_message = "Failed to create model in Ollama"
            
            job.completed_at = datetime.now()
            
            # Move to completed jobs
            self.completed_jobs[job.job_id] = job
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Fine-tuning job {job.job_id} failed: {e}")
    
    def _create_ollama_model(self, model_name: str, modelfile_path: Path) -> bool:
        """Create model in Ollama"""
        try:
            cmd = ['ollama', 'create', model_name, '-f', str(modelfile_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Successfully created Ollama model: {model_name}")
                return True
            else:
                logger.error(f"Failed to create Ollama model: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating Ollama model: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[FineTuningJob]:
        """Get status of a fine-tuning job"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        return None
    
    def list_jobs(self) -> Dict:
        """List all fine-tuning jobs"""
        return {
            'active': list(self.active_jobs.keys()),
            'completed': list(self.completed_jobs.keys())
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running fine-tuning job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = 'cancelled'
            job.completed_at = datetime.now()
            
            # Move to completed jobs
            self.completed_jobs[job_id] = job
            del self.active_jobs[job_id]
            
            logger.info(f"Cancelled fine-tuning job {job_id}")
            return True
        
        return False


class FineTuningService:
    """Main fine-tuning service"""
    
    def __init__(self):
        self.data_manager = TrainingDataManager()
        self.fine_tuning_engine = FineTuningEngine()
        self.stats = {
            'jobs_started': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'datasets_created': 0,
            'examples_added': 0
        }
    
    def create_dataset(self, name: str, description: str = "") -> bool:
        """Create a new training dataset"""
        success = self.data_manager.create_dataset(name, description)
        if success:
            self.stats['datasets_created'] += 1
        return success
    
    def add_training_examples(self, dataset_name: str, examples: List[Dict]) -> bool:
        """Add training examples to a dataset"""
        training_examples = [
            TrainingExample(
                input_text=ex['input'],
                output_text=ex['output'],
                metadata=ex.get('metadata')
            )
            for ex in examples
        ]
        
        success = self.data_manager.add_examples(dataset_name, training_examples)
        if success:
            self.stats['examples_added'] += len(examples)
        return success
    
    def import_training_data(self, dataset_name: str, file_path: str, format_type: str = "jsonl") -> bool:
        """Import training data from file"""
        return self.data_manager.import_from_file(dataset_name, file_path, format_type)
    
    def start_fine_tuning(self, config_dict: Dict) -> str:
        """Start a fine-tuning job"""
        try:
            # Create config object
            config = FineTuningConfig(**config_dict)
            
            # Get dataset
            dataset = self.data_manager.get_dataset(config.dataset_name)
            if not dataset:
                raise ValueError(f"Dataset '{config.dataset_name}' not found")
            
            if len(dataset.examples) == 0:
                raise ValueError(f"Dataset '{config.dataset_name}' is empty")
            
            # Start fine-tuning
            job_id = self.fine_tuning_engine.start_fine_tuning(config, dataset)
            
            if job_id:
                self.stats['jobs_started'] += 1
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error starting fine-tuning: {e}")
            return ""
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get fine-tuning job status"""
        job = self.fine_tuning_engine.get_job_status(job_id)
        if job:
            return asdict(job)
        return None
    
    def list_datasets(self) -> List[Dict]:
        """List all training datasets"""
        return self.data_manager.list_datasets()
    
    def list_jobs(self) -> Dict:
        """List all fine-tuning jobs"""
        return self.fine_tuning_engine.list_jobs()
    
    def get_dataset_details(self, name: str) -> Optional[Dict]:
        """Get detailed information about a dataset"""
        dataset = self.data_manager.get_dataset(name)
        if dataset:
            return {
                'name': dataset.name,
                'description': dataset.description,
                'example_count': len(dataset.examples),
                'created_at': dataset.created_at.isoformat(),
                'updated_at': dataset.updated_at.isoformat(),
                'sample_examples': [
                    {
                        'input': ex.input_text[:200] + '...' if len(ex.input_text) > 200 else ex.input_text,
                        'output': ex.output_text[:200] + '...' if len(ex.output_text) > 200 else ex.output_text
                    }
                    for ex in dataset.examples[:5]  # Show first 5 examples
                ]
            }
        return None
    
    def export_dataset(self, dataset_name: str, output_path: str, format_type: str = "jsonl") -> bool:
        """Export dataset to file"""
        return self.data_manager.export_dataset(dataset_name, output_path, format_type)
    
    def delete_dataset(self, name: str) -> bool:
        """Delete a training dataset"""
        return self.data_manager.delete_dataset(name)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a fine-tuning job"""
        success = self.fine_tuning_engine.cancel_job(job_id)
        if success:
            self.stats['jobs_failed'] += 1
        return success
    
    def get_service_stats(self) -> Dict:
        """Get service statistics"""
        return {
            'stats': self.stats,
            'datasets': len(self.data_manager.datasets),
            'active_jobs': len(self.fine_tuning_engine.active_jobs),
            'completed_jobs': len(self.fine_tuning_engine.completed_jobs)
        }
    
    def get_recommended_config(self, dataset_name: str, use_case: str = "general") -> Dict:
        """Get recommended fine-tuning configuration"""
        dataset = self.data_manager.get_dataset(dataset_name)
        if not dataset:
            return {}
        
        example_count = len(dataset.examples)
        
        # Base configuration
        config = {
            'dataset_name': dataset_name,
            'learning_rate': 2e-5,
            'batch_size': 4,
            'num_epochs': 3,
            'max_length': 512,
            'use_lora': True,
            'lora_r': 16,
            'lora_alpha': 32
        }
        
        # Adjust based on dataset size
        if example_count < 100:
            config['num_epochs'] = 5
            config['learning_rate'] = 1e-5
        elif example_count > 1000:
            config['num_epochs'] = 2
            config['batch_size'] = 8
        
        # Adjust based on use case
        use_case_configs = {
            'conversational': {
                'system_prompt': 'You are a helpful, harmless, and honest AI assistant.',
                'temperature': 0.7
            },
            'code': {
                'system_prompt': 'You are a coding assistant that helps with programming tasks.',
                'temperature': 0.1,
                'max_length': 1024
            },
            'creative': {
                'system_prompt': 'You are a creative writing assistant.',
                'temperature': 0.9
            },
            'factual': {
                'system_prompt': 'You are a factual assistant that provides accurate information.',
                'temperature': 0.3
            }
        }
        
        if use_case in use_case_configs:
            config.update(use_case_configs[use_case])
        
        return config


# Global fine-tuning service instance
fine_tuning_service = FineTuningService()