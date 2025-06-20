"""
Model Management Service
Handles model lifecycle, monitoring, and optimization
"""

import os
import json
import time
import psutil
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from services.ollama_service import ollama_service, ModelInfo
from services.huggingface_service import hf_service, ModelDownloadProgress

logger = logging.getLogger(__name__)


@dataclass
class ModelUsageStats:
    """Model usage statistics"""
    model_name: str
    total_requests: int
    total_tokens: int
    average_response_time: float
    last_used: datetime
    error_count: int
    success_rate: float
    memory_usage: float = 0.0
    gpu_usage: float = 0.0


@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics"""
    model_name: str
    avg_tokens_per_second: float
    avg_memory_usage: float
    avg_gpu_usage: float
    avg_load_time: float
    uptime: float
    requests_per_hour: float


@dataclass
class SystemResources:
    """System resource information"""
    cpu_percent: float
    memory_percent: float
    memory_available: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_usage: float = 0.0
    temperature: float = 0.0


class ModelManager:
    """Comprehensive model management and monitoring"""
    
    def __init__(self):
        self.usage_stats = defaultdict(lambda: ModelUsageStats(
            model_name="",
            total_requests=0,
            total_tokens=0,
            average_response_time=0.0,
            last_used=datetime.now(),
            error_count=0,
            success_rate=100.0
        ))
        
        self.performance_metrics = {}
        self.active_models = set()
        self.model_load_times = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Model recommendations based on use case
        self.model_recommendations = {
            "general_chat": [
                "llama2:7b-chat",
                "mistral:7b-instruct",
                "neural-chat:7b"
            ],
            "code_assistance": [
                "codellama:7b-instruct",
                "codellama:13b-instruct",
                "deepseek-coder:6.7b"
            ],
            "creative_writing": [
                "llama2:13b-chat",
                "mistral:7b-instruct",
                "neural-chat:7b"
            ],
            "technical_support": [
                "mistral:7b-instruct",
                "llama2:13b-chat",
                "openchat:7b"
            ],
            "lightweight": [
                "tinyllama:1.1b",
                "phi:2.7b",
                "gemma:2b"
            ]
        }
    
    def start_monitoring(self):
        """Start system and model monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Model monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Model monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_system_metrics()
                self._update_model_metrics()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # GPU metrics (if available)
            gpu_percent = 0.0
            gpu_memory_percent = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_percent = gpu.load * 100
                    gpu_memory_percent = gpu.memoryUtil * 100
            except ImportError:
                pass
            
            # Temperature (if available)
            temperature = 0.0
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get CPU temperature
                    for name, entries in temps.items():
                        if entries:
                            temperature = entries[0].current
                            break
            except (AttributeError, OSError):
                pass
            
            self.system_resources = SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available=memory.available / (1024**3),  # GB
                gpu_percent=gpu_percent,
                gpu_memory_percent=gpu_memory_percent,
                disk_usage=disk_percent,
                temperature=temperature
            )
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def _update_model_metrics(self):
        """Update model performance metrics"""
        try:
            # Get current models
            models = ollama_service.list_models()
            
            for model in models:
                model_name = model.name
                
                if model_name in self.usage_stats:
                    stats = self.usage_stats[model_name]
                    
                    # Calculate performance metrics
                    if stats.total_requests > 0:
                        tokens_per_second = stats.total_tokens / (stats.total_requests * stats.average_response_time) if stats.average_response_time > 0 else 0
                        
                        self.performance_metrics[model_name] = ModelPerformanceMetrics(
                            model_name=model_name,
                            avg_tokens_per_second=tokens_per_second,
                            avg_memory_usage=stats.memory_usage,
                            avg_gpu_usage=stats.gpu_usage,
                            avg_load_time=self.model_load_times.get(model_name, 0.0),
                            uptime=(datetime.now() - stats.last_used).total_seconds() / 3600,  # hours
                            requests_per_hour=stats.total_requests / max(1, (datetime.now() - stats.last_used).total_seconds() / 3600)
                        )
            
        except Exception as e:
            logger.error(f"Failed to update model metrics: {e}")
    
    def record_model_usage(
        self,
        model_name: str,
        response_time: float,
        token_count: int = 0,
        success: bool = True,
        memory_usage: float = 0.0,
        gpu_usage: float = 0.0
    ):
        """Record model usage statistics"""
        
        stats = self.usage_stats[model_name]
        stats.model_name = model_name
        
        # Update counters
        stats.total_requests += 1
        stats.total_tokens += token_count
        stats.last_used = datetime.now()
        
        if not success:
            stats.error_count += 1
        
        # Update averages
        if stats.total_requests > 1:
            stats.average_response_time = (
                (stats.average_response_time * (stats.total_requests - 1) + response_time) / 
                stats.total_requests
            )
        else:
            stats.average_response_time = response_time
        
        # Update success rate
        stats.success_rate = ((stats.total_requests - stats.error_count) / stats.total_requests) * 100
        
        # Update resource usage
        if memory_usage > 0:
            stats.memory_usage = memory_usage
        if gpu_usage > 0:
            stats.gpu_usage = gpu_usage
        
        # Add to active models
        self.active_models.add(model_name)
    
    def get_model_recommendations(self, use_case: str = "general_chat") -> List[str]:
        """Get model recommendations for specific use case"""
        return self.model_recommendations.get(use_case, self.model_recommendations["general_chat"])
    
    def get_optimal_model(self, use_case: str = "general_chat", max_memory_gb: float = 8.0) -> Optional[str]:
        """Get optimal model based on use case and system resources"""
        
        recommendations = self.get_model_recommendations(use_case)
        available_models = [model.name for model in ollama_service.list_models()]
        
        # Filter by available models
        available_recommendations = [model for model in recommendations if model in available_models]
        
        if not available_recommendations:
            return None
        
        # Consider system resources
        if hasattr(self, 'system_resources'):
            available_memory = self.system_resources.memory_available
            
            # Filter by memory requirements (rough estimates)
            memory_requirements = {
                "tinyllama:1.1b": 1.0,
                "phi:2.7b": 2.0,
                "gemma:2b": 2.0,
                "llama2:7b": 4.0,
                "mistral:7b": 4.0,
                "codellama:7b": 4.0,
                "llama2:13b": 8.0,
                "codellama:13b": 8.0
            }
            
            suitable_models = []
            for model in available_recommendations:
                required_memory = memory_requirements.get(model, 4.0)  # Default 4GB
                if required_memory <= min(available_memory, max_memory_gb):
                    suitable_models.append(model)
            
            if suitable_models:
                available_recommendations = suitable_models
        
        # Return best performing model if we have stats
        if self.usage_stats:
            best_model = None
            best_score = 0
            
            for model in available_recommendations:
                if model in self.usage_stats:
                    stats = self.usage_stats[model]
                    # Score based on success rate and response time
                    score = stats.success_rate / max(1, stats.average_response_time)
                    if score > best_score:
                        best_score = score
                        best_model = model
            
            if best_model:
                return best_model
        
        # Return first available recommendation
        return available_recommendations[0]
    
    def get_model_usage_stats(self, model_name: str = None) -> Dict:
        """Get usage statistics for specific model or all models"""
        
        if model_name:
            if model_name in self.usage_stats:
                return asdict(self.usage_stats[model_name])
            return {}
        
        # Return all stats
        return {name: asdict(stats) for name, stats in self.usage_stats.items()}
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        
        status = {
            "ollama_available": ollama_service.is_available(),
            "ollama_version": ollama_service.get_version(),
            "total_models": len(ollama_service.list_models()),
            "active_models": len(self.active_models),
            "monitoring_active": self.monitoring_active
        }
        
        if hasattr(self, 'system_resources'):
            status["system_resources"] = asdict(self.system_resources)
        
        return status
    
    def get_model_performance_report(self) -> Dict:
        """Get comprehensive model performance report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_status": self.get_system_status(),
            "model_usage_stats": self.get_model_usage_stats(),
            "performance_metrics": {name: asdict(metrics) for name, metrics in self.performance_metrics.items()},
            "recommendations": {}
        }
        
        # Add recommendations for each use case
        for use_case in self.model_recommendations:
            optimal_model = self.get_optimal_model(use_case)
            report["recommendations"][use_case] = optimal_model
        
        return report
    
    def cleanup_unused_models(self, days_unused: int = 7) -> List[str]:
        """Clean up models that haven't been used for specified days"""
        
        cutoff_date = datetime.now() - timedelta(days=days_unused)
        unused_models = []
        
        for model_name, stats in self.usage_stats.items():
            if stats.last_used < cutoff_date and stats.total_requests == 0:
                unused_models.append(model_name)
        
        # Remove unused models
        removed_models = []
        for model_name in unused_models:
            if ollama_service.delete_model(model_name):
                removed_models.append(model_name)
                del self.usage_stats[model_name]
                self.active_models.discard(model_name)
        
        return removed_models
    
    def optimize_model_parameters(self, model_name: str, use_case: str = "general_chat") -> Dict:
        """Get optimized parameters for a model based on use case and performance"""
        
        base_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1
        }
        
        # Adjust based on use case
        use_case_params = {
            "general_chat": {"temperature": 0.7, "top_p": 0.9},
            "code_assistance": {"temperature": 0.1, "top_p": 0.95, "repeat_penalty": 1.05},
            "creative_writing": {"temperature": 0.9, "top_p": 0.95, "top_k": 50},
            "technical_support": {"temperature": 0.3, "top_p": 0.9, "repeat_penalty": 1.1},
            "lightweight": {"temperature": 0.7, "top_p": 0.8, "top_k": 30}
        }
        
        if use_case in use_case_params:
            base_params.update(use_case_params[use_case])
        
        # Adjust based on model performance
        if model_name in self.usage_stats:
            stats = self.usage_stats[model_name]
            
            # If model is slow, reduce complexity
            if stats.average_response_time > 5.0:
                base_params["top_k"] = min(base_params["top_k"], 30)
                base_params["max_tokens"] = 1024
            
            # If model has high error rate, be more conservative
            if stats.success_rate < 90:
                base_params["temperature"] = min(base_params["temperature"], 0.5)
                base_params["repeat_penalty"] = max(base_params["repeat_penalty"], 1.1)
        
        return base_params
    
    def get_model_health_check(self) -> Dict:
        """Perform health check on all models"""
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "overall_health": "good"
        }
        
        models = ollama_service.list_models()
        issues_found = 0
        
        for model in models:
            model_name = model.name
            model_health = {
                "status": "healthy",
                "issues": [],
                "recommendations": []
            }
            
            # Check if model has been used
            if model_name in self.usage_stats:
                stats = self.usage_stats[model_name]
                
                # Check success rate
                if stats.success_rate < 80:
                    model_health["status"] = "warning"
                    model_health["issues"].append(f"Low success rate: {stats.success_rate:.1f}%")
                    model_health["recommendations"].append("Consider model retraining or parameter adjustment")
                    issues_found += 1
                
                # Check response time
                if stats.average_response_time > 10.0:
                    model_health["status"] = "warning"
                    model_health["issues"].append(f"Slow response time: {stats.average_response_time:.1f}s")
                    model_health["recommendations"].append("Consider using a smaller model or optimizing parameters")
                    issues_found += 1
                
                # Check recent usage
                days_since_use = (datetime.now() - stats.last_used).days
                if days_since_use > 30:
                    model_health["issues"].append(f"Not used for {days_since_use} days")
                    model_health["recommendations"].append("Consider removing if no longer needed")
            
            health_report["models"][model_name] = model_health
        
        # Overall health assessment
        if issues_found > len(models) * 0.5:
            health_report["overall_health"] = "poor"
        elif issues_found > 0:
            health_report["overall_health"] = "warning"
        
        return health_report


# Global model manager instance
model_manager = ModelManager()