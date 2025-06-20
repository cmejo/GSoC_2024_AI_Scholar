"""
Embeddings Service
Advanced embedding generation, storage, and similarity search capabilities
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import numpy as np
from collections import defaultdict
import hashlib

from services.ollama_service import ollama_service

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")


@dataclass
class EmbeddingModel:
    """Embedding model configuration"""
    name: str
    dimension: int
    max_sequence_length: int
    model_type: str  # 'sentence_transformer', 'ollama', 'openai'
    model_path: Optional[str] = None
    is_loaded: bool = False
    load_time: Optional[float] = None


@dataclass
class EmbeddingRequest:
    """Embedding generation request"""
    texts: List[str]
    model_name: str
    normalize: bool = True
    batch_size: int = 32
    metadata: Optional[Dict] = None


@dataclass
class EmbeddingResult:
    """Embedding generation result"""
    embeddings: List[List[float]]
    model_used: str
    dimension: int
    processing_time: float
    token_count: int
    metadata: Optional[Dict] = None


@dataclass
class SimilarityResult:
    """Similarity search result"""
    text: str
    embedding: List[float]
    score: float
    rank: int
    metadata: Optional[Dict] = None


class EmbeddingModelManager:
    """Manage multiple embedding models"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {
            # Sentence Transformer models
            'all-MiniLM-L6-v2': EmbeddingModel(
                name='all-MiniLM-L6-v2',
                dimension=384,
                max_sequence_length=256,
                model_type='sentence_transformer'
            ),
            'all-mpnet-base-v2': EmbeddingModel(
                name='all-mpnet-base-v2',
                dimension=768,
                max_sequence_length=384,
                model_type='sentence_transformer'
            ),
            'multi-qa-MiniLM-L6-cos-v1': EmbeddingModel(
                name='multi-qa-MiniLM-L6-cos-v1',
                dimension=384,
                max_sequence_length=512,
                model_type='sentence_transformer'
            ),
            'paraphrase-multilingual-MiniLM-L12-v2': EmbeddingModel(
                name='paraphrase-multilingual-MiniLM-L12-v2',
                dimension=384,
                max_sequence_length=128,
                model_type='sentence_transformer'
            ),
            # Ollama models (for embedding-capable models)
            'nomic-embed-text': EmbeddingModel(
                name='nomic-embed-text',
                dimension=768,
                max_sequence_length=2048,
                model_type='ollama'
            ),
            'mxbai-embed-large': EmbeddingModel(
                name='mxbai-embed-large',
                dimension=1024,
                max_sequence_length=512,
                model_type='ollama'
            )
        }
        
        self.default_model = 'all-MiniLM-L6-v2'
        self.stats = defaultdict(int)
    
    def get_available_models(self) -> List[EmbeddingModel]:
        """Get list of available embedding models"""
        available = []
        
        for name, config in self.model_configs.items():
            if config.model_type == 'sentence_transformer' and SENTENCE_TRANSFORMERS_AVAILABLE:
                available.append(config)
            elif config.model_type == 'ollama' and ollama_service.is_available():
                # Check if model is available in Ollama
                ollama_models = [m.name for m in ollama_service.list_models()]
                if name in ollama_models:
                    available.append(config)
        
        return available
    
    def load_model(self, model_name: str) -> bool:
        """Load an embedding model"""
        if model_name in self.models:
            return True
        
        if model_name not in self.model_configs:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        config = self.model_configs[model_name]
        start_time = time.time()
        
        try:
            if config.model_type == 'sentence_transformer':
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    logger.error("SentenceTransformers not available")
                    return False
                
                model = SentenceTransformer(model_name)
                self.models[model_name] = model
                
            elif config.model_type == 'ollama':
                # For Ollama models, we don't need to load them explicitly
                # Just verify they're available
                if not ollama_service.is_available():
                    logger.error("Ollama service not available")
                    return False
                
                ollama_models = [m.name for m in ollama_service.list_models()]
                if model_name not in ollama_models:
                    logger.error(f"Model {model_name} not available in Ollama")
                    return False
                
                self.models[model_name] = "ollama_model"  # Placeholder
            
            load_time = time.time() - start_time
            config.is_loaded = True
            config.load_time = load_time
            
            logger.info(f"Loaded embedding model {model_name} in {load_time:.2f}s")
            self.stats['models_loaded'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Unload an embedding model"""
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.model_configs:
                self.model_configs[model_name].is_loaded = False
            logger.info(f"Unloaded model {model_name}")
            self.stats['models_unloaded'] += 1
            return True
        return False
    
    def get_model_info(self, model_name: str) -> Optional[EmbeddingModel]:
        """Get information about a model"""
        return self.model_configs.get(model_name)


class EmbeddingGenerator:
    """Generate embeddings using various models"""
    
    def __init__(self, model_manager: EmbeddingModelManager):
        self.model_manager = model_manager
        self.cache = {}  # Simple embedding cache
        self.cache_hits = 0
        self.cache_misses = 0
    
    def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate embeddings for texts"""
        start_time = time.time()
        
        # Load model if not already loaded
        if not self.model_manager.load_model(request.model_name):
            raise ValueError(f"Failed to load model {request.model_name}")
        
        model_config = self.model_manager.get_model_info(request.model_name)
        if not model_config:
            raise ValueError(f"Model configuration not found: {request.model_name}")
        
        # Check cache for embeddings
        cached_embeddings = []
        texts_to_process = []
        cache_keys = []
        
        for text in request.texts:
            cache_key = self._get_cache_key(text, request.model_name)
            cache_keys.append(cache_key)
            
            if cache_key in self.cache:
                cached_embeddings.append(self.cache[cache_key])
                self.cache_hits += 1
            else:
                cached_embeddings.append(None)
                texts_to_process.append(text)
                self.cache_misses += 1
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if texts_to_process:
            if model_config.model_type == 'sentence_transformer':
                new_embeddings = self._generate_sentence_transformer_embeddings(
                    texts_to_process, request.model_name, request
                )
            elif model_config.model_type == 'ollama':
                new_embeddings = self._generate_ollama_embeddings(
                    texts_to_process, request.model_name, request
                )
        
        # Combine cached and new embeddings
        final_embeddings = []
        new_idx = 0
        
        for i, cached_emb in enumerate(cached_embeddings):
            if cached_emb is not None:
                final_embeddings.append(cached_emb)
            else:
                if new_idx < len(new_embeddings):
                    embedding = new_embeddings[new_idx]
                    final_embeddings.append(embedding)
                    # Cache the new embedding
                    self.cache[cache_keys[i]] = embedding
                    new_idx += 1
                else:
                    # Fallback embedding
                    final_embeddings.append([0.0] * model_config.dimension)
        
        # Normalize embeddings if requested
        if request.normalize:
            final_embeddings = self._normalize_embeddings(final_embeddings)
        
        processing_time = time.time() - start_time
        token_count = sum(len(text.split()) for text in request.texts)
        
        return EmbeddingResult(
            embeddings=final_embeddings,
            model_used=request.model_name,
            dimension=model_config.dimension,
            processing_time=processing_time,
            token_count=token_count,
            metadata=request.metadata
        )
    
    def _generate_sentence_transformer_embeddings(self, 
                                                texts: List[str], 
                                                model_name: str, 
                                                request: EmbeddingRequest) -> List[List[float]]:
        """Generate embeddings using SentenceTransformers"""
        model = self.model_manager.models[model_name]
        
        try:
            # Process in batches
            all_embeddings = []
            batch_size = request.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = model.encode(
                    batch_texts,
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
                all_embeddings.extend(batch_embeddings.tolist())
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating SentenceTransformer embeddings: {e}")
            # Return zero embeddings as fallback
            model_config = self.model_manager.get_model_info(model_name)
            return [[0.0] * model_config.dimension for _ in texts]
    
    def _generate_ollama_embeddings(self, 
                                  texts: List[str], 
                                  model_name: str, 
                                  request: EmbeddingRequest) -> List[List[float]]:
        """Generate embeddings using Ollama"""
        embeddings = []
        
        try:
            for text in texts:
                embedding = ollama_service.get_embeddings(model_name, text)
                if embedding:
                    embeddings.append(embedding)
                else:
                    # Fallback embedding
                    model_config = self.model_manager.get_model_info(model_name)
                    embeddings.append([0.0] * model_config.dimension)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating Ollama embeddings: {e}")
            # Return zero embeddings as fallback
            model_config = self.model_manager.get_model_info(model_name)
            return [[0.0] * model_config.dimension for _ in texts]
    
    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize embeddings to unit length"""
        normalized = []
        for embedding in embeddings:
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                normalized_embedding = (embedding_array / norm).tolist()
            else:
                normalized_embedding = embedding
            normalized.append(normalized_embedding)
        return normalized
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model"""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }


class SimilaritySearchEngine:
    """Efficient similarity search using FAISS or fallback methods"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.texts = []
        self.embeddings = []
        self.metadata = []
        self.use_faiss = FAISS_AVAILABLE
        
        if self.use_faiss:
            self._initialize_faiss_index()
        else:
            logger.warning("FAISS not available, using numpy fallback for similarity search")
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index"""
        try:
            # Use cosine similarity index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.use_faiss = False
    
    def add_embeddings(self, 
                      texts: List[str], 
                      embeddings: List[List[float]], 
                      metadata: List[Dict] = None) -> bool:
        """Add embeddings to the search index"""
        try:
            if len(texts) != len(embeddings):
                raise ValueError("Number of texts and embeddings must match")
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = []
            for embedding in embeddings:
                embedding_array = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(embedding_array)
                if norm > 0:
                    normalized_embeddings.append(embedding_array / norm)
                else:
                    normalized_embeddings.append(embedding_array)
            
            # Store texts and metadata
            self.texts.extend(texts)
            self.embeddings.extend(normalized_embeddings)
            self.metadata.extend(metadata or [{}] * len(texts))
            
            if self.use_faiss and self.index is not None:
                # Add to FAISS index
                embeddings_array = np.array(normalized_embeddings, dtype=np.float32)
                self.index.add(embeddings_array)
            
            logger.info(f"Added {len(texts)} embeddings to search index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings to search index: {e}")
            return False
    
    def search(self, 
               query_embedding: List[float], 
               k: int = 5, 
               threshold: float = 0.0) -> List[SimilarityResult]:
        """Search for similar embeddings"""
        try:
            if not self.embeddings:
                return []
            
            # Normalize query embedding
            query_array = np.array(query_embedding, dtype=np.float32)
            norm = np.linalg.norm(query_array)
            if norm > 0:
                query_array = query_array / norm
            
            if self.use_faiss and self.index is not None:
                return self._search_faiss(query_array, k, threshold)
            else:
                return self._search_numpy(query_array, k, threshold)
                
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def _search_faiss(self, 
                     query_array: np.ndarray, 
                     k: int, 
                     threshold: float) -> List[SimilarityResult]:
        """Search using FAISS index"""
        # Search the index
        scores, indices = self.index.search(query_array.reshape(1, -1), k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and score >= threshold:
                results.append(SimilarityResult(
                    text=self.texts[idx],
                    embedding=self.embeddings[idx].tolist(),
                    score=float(score),
                    rank=rank + 1,
                    metadata=self.metadata[idx]
                ))
        
        return results
    
    def _search_numpy(self, 
                     query_array: np.ndarray, 
                     k: int, 
                     threshold: float) -> List[SimilarityResult]:
        """Search using numpy (fallback method)"""
        similarities = []
        
        for i, embedding in enumerate(self.embeddings):
            # Cosine similarity (dot product of normalized vectors)
            similarity = np.dot(query_array, embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:k]
        
        results = []
        for rank, (score, idx) in enumerate(top_results):
            if score >= threshold:
                results.append(SimilarityResult(
                    text=self.texts[idx],
                    embedding=self.embeddings[idx].tolist(),
                    score=float(score),
                    rank=rank + 1,
                    metadata=self.metadata[idx]
                ))
        
        return results
    
    def get_stats(self) -> Dict:
        """Get search engine statistics"""
        return {
            "total_embeddings": len(self.embeddings),
            "dimension": self.dimension,
            "using_faiss": self.use_faiss,
            "index_type": "FAISS IndexFlatIP" if self.use_faiss else "Numpy fallback"
        }
    
    def clear(self):
        """Clear all embeddings from the index"""
        self.texts.clear()
        self.embeddings.clear()
        self.metadata.clear()
        
        if self.use_faiss:
            self._initialize_faiss_index()


class EmbeddingsService:
    """Main embeddings service"""
    
    def __init__(self):
        self.model_manager = EmbeddingModelManager()
        self.embedding_generator = EmbeddingGenerator(self.model_manager)
        self.search_engines = {}  # Multiple search engines for different collections
        self.stats = defaultdict(int)
    
    def get_available_models(self) -> List[Dict]:
        """Get available embedding models"""
        models = self.model_manager.get_available_models()
        return [asdict(model) for model in models]
    
    def load_model(self, model_name: str) -> bool:
        """Load an embedding model"""
        return self.model_manager.load_model(model_name)
    
    def generate_embeddings(self, 
                          texts: List[str], 
                          model_name: str = None,
                          normalize: bool = True,
                          batch_size: int = 32) -> EmbeddingResult:
        """Generate embeddings for texts"""
        model_name = model_name or self.model_manager.default_model
        
        request = EmbeddingRequest(
            texts=texts,
            model_name=model_name,
            normalize=normalize,
            batch_size=batch_size
        )
        
        result = self.embedding_generator.generate_embeddings(request)
        self.stats['embeddings_generated'] += len(texts)
        
        return result
    
    def create_search_index(self, 
                          collection_name: str, 
                          dimension: int = None) -> bool:
        """Create a new search index"""
        try:
            if dimension is None:
                # Use default model dimension
                default_model = self.model_manager.get_model_info(self.model_manager.default_model)
                dimension = default_model.dimension if default_model else 384
            
            self.search_engines[collection_name] = SimilaritySearchEngine(dimension)
            logger.info(f"Created search index '{collection_name}' with dimension {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating search index: {e}")
            return False
    
    def add_to_search_index(self, 
                          collection_name: str,
                          texts: List[str],
                          embeddings: List[List[float]] = None,
                          metadata: List[Dict] = None,
                          model_name: str = None) -> bool:
        """Add texts and embeddings to search index"""
        try:
            # Create index if it doesn't exist
            if collection_name not in self.search_engines:
                if not self.create_search_index(collection_name):
                    return False
            
            # Generate embeddings if not provided
            if embeddings is None:
                result = self.generate_embeddings(texts, model_name)
                embeddings = result.embeddings
            
            # Add to search index
            success = self.search_engines[collection_name].add_embeddings(
                texts, embeddings, metadata
            )
            
            if success:
                self.stats['texts_indexed'] += len(texts)
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding to search index: {e}")
            return False
    
    def search_similar(self, 
                      collection_name: str,
                      query: str,
                      k: int = 5,
                      threshold: float = 0.0,
                      model_name: str = None) -> List[SimilarityResult]:
        """Search for similar texts"""
        try:
            if collection_name not in self.search_engines:
                logger.error(f"Search index '{collection_name}' not found")
                return []
            
            # Generate query embedding
            result = self.generate_embeddings([query], model_name)
            query_embedding = result.embeddings[0]
            
            # Search
            results = self.search_engines[collection_name].search(
                query_embedding, k, threshold
            )
            
            self.stats['searches_performed'] += 1
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_collection_stats(self, collection_name: str = None) -> Dict:
        """Get statistics for a collection or all collections"""
        if collection_name:
            if collection_name in self.search_engines:
                return self.search_engines[collection_name].get_stats()
            else:
                return {"error": f"Collection '{collection_name}' not found"}
        else:
            # Return stats for all collections
            collections = {}
            for name, engine in self.search_engines.items():
                collections[name] = engine.get_stats()
            return collections
    
    def get_service_stats(self) -> Dict:
        """Get overall service statistics"""
        cache_stats = self.embedding_generator.get_cache_stats()
        
        return {
            "service_stats": dict(self.stats),
            "cache_stats": cache_stats,
            "model_stats": {
                "loaded_models": len(self.model_manager.models),
                "available_models": len(self.model_manager.get_available_models())
            },
            "collections": list(self.search_engines.keys()),
            "collection_count": len(self.search_engines)
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_generator.clear_cache()
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a search collection"""
        if collection_name in self.search_engines:
            del self.search_engines[collection_name]
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        return False


# Global embeddings service instance
embeddings_service = EmbeddingsService()