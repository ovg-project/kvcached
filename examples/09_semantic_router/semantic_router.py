#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Semantic Router for Multi-Model LLM Serving

Routes requests to specialized models based on query content analysis.
Demonstrates intelligent routing with kvcached multi-model support.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import uvicorn

# Optional: Use sentence-transformers for better semantic matching
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Using keyword matching.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Router", description="Routes to specialized models")


@dataclass
class ModelConfig:
    """Configuration for a specialized model."""
    name: str
    endpoint: str
    categories: List[str]
    keywords: List[str]
    priority: int = 0


# Default model configurations
DEFAULT_MODELS = [
    ModelConfig(
        name="code",
        endpoint="http://localhost:12346",
        categories=["programming", "coding", "software"],
        keywords=[
            "code", "program", "function", "class", "algorithm", "python",
            "javascript", "rust", "debug", "compile", "syntax", "api",
            "implement", "script", "library", "framework", "git", "docker"
        ],
        priority=1,
    ),
    ModelConfig(
        name="math",
        endpoint="http://localhost:12347",
        categories=["mathematics", "calculation", "equations"],
        keywords=[
            "math", "calculate", "solve", "equation", "integral", "derivative",
            "algebra", "geometry", "statistics", "probability", "matrix",
            "vector", "calculus", "theorem", "proof", "formula", "sum"
        ],
        priority=1,
    ),
    ModelConfig(
        name="general",
        endpoint="http://localhost:12348",
        categories=["general", "conversation", "knowledge"],
        keywords=[],  # Fallback for everything else
        priority=0,
    ),
]


class SemanticRouter:
    """Routes queries to specialized models based on content."""

    def __init__(self, models: List[ModelConfig], embedding_model: str = "all-MiniLM-L6-v2"):
        self.models = {m.name: m for m in models}
        self.stats: Dict[str, int] = {m.name: 0 for m in models}
        self.total_requests = 0

        # Initialize embeddings if available
        self.embedder = None
        self.category_embeddings: Dict[str, any] = {}
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer(embedding_model)
                self._precompute_embeddings()
                logger.info("Initialized semantic embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")

    def _precompute_embeddings(self):
        """Pre-compute embeddings for category keywords."""
        if not self.embedder:
            return
        for model in self.models.values():
            if model.keywords:
                text = " ".join(model.keywords + model.categories)
                self.category_embeddings[model.name] = self.embedder.encode(text)

    def classify_query(self, query: str) -> str:
        """Classify a query and return the best model name."""
        query_lower = query.lower()

        # Try semantic matching first
        if self.embedder and self.category_embeddings:
            query_embedding = self.embedder.encode(query)
            best_score = -1
            best_model = "general"

            for name, cat_embedding in self.category_embeddings.items():
                # Cosine similarity
                score = float(query_embedding @ cat_embedding) / (
                    (query_embedding @ query_embedding) ** 0.5 *
                    (cat_embedding @ cat_embedding) ** 0.5
                )
                if score > best_score and score > 0.3:  # Threshold
                    best_score = score
                    best_model = name

            if best_score > 0.3:
                logger.debug(f"Semantic match: {best_model} (score={best_score:.3f})")
                return best_model

        # Fallback to keyword matching
        best_match = "general"
        best_count = 0
        for model in self.models.values():
            if model.name == "general":
                continue
            count = sum(1 for kw in model.keywords if kw in query_lower)
            if count > best_count:
                best_count = count
                best_match = model.name

        logger.debug(f"Keyword match: {best_match} (count={best_count})")
        return best_match

    def route(self, query: str) -> ModelConfig:
        """Route a query to the appropriate model."""
        model_name = self.classify_query(query)
        self.stats[model_name] = self.stats.get(model_name, 0) + 1
        self.total_requests += 1
        return self.models[model_name]

    def get_stats(self) -> dict:
        """Return routing statistics."""
        return {
            "total_requests": self.total_requests,
            "routes": self.stats.copy(),
        }


# Global router instance
router: Optional[SemanticRouter] = None


def extract_query(request_body: dict) -> str:
    """Extract the user query from various request formats."""
    # Chat completions format
    if "messages" in request_body:
        messages = request_body["messages"]
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")

    # Completions format
    if "prompt" in request_body:
        return request_body["prompt"]

    return ""


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Route chat completion requests to appropriate model."""
    body = await request.json()
    query = extract_query(body)

    if not query:
        raise HTTPException(status_code=400, detail="No query found in request")

    model_config = router.route(query)
    logger.info(f"Routing to {model_config.name}: {query[:50]}...")

    # Forward request to selected model
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{model_config.endpoint}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"},
        )

        if body.get("stream", False):
            return StreamingResponse(
                response.aiter_bytes(),
                media_type="text/event-stream",
            )
        return response.json()


@app.post("/v1/completions")
async def completions(request: Request):
    """Route completion requests to appropriate model."""
    body = await request.json()
    query = extract_query(body)

    if not query:
        raise HTTPException(status_code=400, detail="No query found in request")

    model_config = router.route(query)
    logger.info(f"Routing to {model_config.name}: {query[:50]}...")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{model_config.endpoint}/v1/completions",
            json=body,
            headers={"Content-Type": "application/json"},
        )

        if body.get("stream", False):
            return StreamingResponse(
                response.aiter_bytes(),
                media_type="text/event-stream",
            )
        return response.json()


@app.get("/stats")
async def get_stats():
    """Return routing statistics."""
    return router.get_stats()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "models": list(router.models.keys())}


def main():
    parser = argparse.ArgumentParser(description="Semantic Router for LLM serving")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--config", help="Path to config file")
    args = parser.parse_args()

    global router

    # Load custom config or use defaults
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        models = [ModelConfig(**m) for m in config["models"]]
    else:
        models = DEFAULT_MODELS

    router = SemanticRouter(models)
    logger.info(f"Starting semantic router on {args.host}:{args.port}")
    logger.info(f"Models: {list(router.models.keys())}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
