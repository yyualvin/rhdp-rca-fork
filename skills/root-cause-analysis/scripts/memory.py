"""mem0-backed memory layer for RCA analyses.

Replaces the custom pgvector embedding pipeline with mem0's automatic
deduplication and consolidation. Each RCA is stored as a memory keyed
by catalog_item (user_id) and job_id (run_id).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

_mem0_client = None


def _compact_failed_tasks(failed_tasks: list[dict] | None) -> list[dict[str, str]]:
    """Keep only the fields useful for retrieval/display."""
    result: list[dict[str, str]] = []
    for task in failed_tasks or []:
        if not isinstance(task, dict):
            continue
        entry = {
            "task": str(task.get("task") or ""),
            "error_message": str(task.get("error_message") or ""),
            "task_action": str(task.get("task_action") or ""),
        }
        if any(entry.values()):
            result.append(entry)
    return result


def _collect_github_paths(summary: dict[str, Any]) -> list[str]:
    """Collect unique github_path values from evidence and recommendations."""
    paths: list[str] = []
    seen: set[str] = set()
    for section in ("evidence", "recommendations"):
        for item in summary.get(section) or []:
            if not isinstance(item, dict):
                continue
            path = item.get("github_path")
            if path and path not in seen:
                seen.add(path)
                paths.append(str(path))
    return paths


def build_rca_metadata(
    job_context: dict[str, Any],
    github_context: dict[str, Any] | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one embedding row's metadata from step1/step4/step5 artifacts."""
    job_meta = (github_context or {}).get("job_metadata") or {}
    root_cause = summary.get("root_cause") or {}
    time_window = job_context.get("time_window") or {}

    return {
        "job_id": str(
            summary.get("job_id") or job_context.get("job_id") or ""
        ),
        "guid": job_context.get("guid") or "",
        "namespace": job_context.get("namespace") or "",
        "cluster": job_context.get("cluster") or "",
        "status": job_context.get("status") or "",
        "action": job_context.get("action") or "",
        "cloud_provider": job_context.get("cloud_provider") or "",
        "env_type": job_context.get("env_type") or "",
        "platform": job_meta.get("platform") or "",
        "catalog_item": job_meta.get("catalog_item") or "",
        "env": job_meta.get("env") or "",
        "job_duration_seconds": time_window.get("duration_seconds"),
        "root_cause_category": root_cause.get("category") or "",
        "root_cause_summary": root_cause.get("summary") or "",
        "confidence": root_cause.get("confidence") or "",
        "contributing_factors": list(summary.get("contributing_factors") or []),
        "failed_tasks": _compact_failed_tasks(job_context.get("failed_tasks")),
        "github_paths": _collect_github_paths(summary),
        "recommendation_count": len(summary.get("recommendations") or []),
        "analyzed_at": summary.get("analyzed_at") or "",
    }


def build_embedding_text(metadata: dict[str, Any]) -> str:
    """Build retrieval-friendly text weighted toward semantic failure content."""
    parts: list[str] = []

    category = metadata.get("root_cause_category") or ""
    summary = metadata.get("root_cause_summary") or ""
    if category or summary:
        parts.append(f"Root cause ({category}): {summary}".strip())

    catalog = metadata.get("catalog_item") or ""
    platform = metadata.get("platform") or ""
    env = metadata.get("env") or ""
    env_bits = [b for b in (platform, catalog, env) if b]
    if env_bits:
        parts.append(f"Catalog: {' / '.join(env_bits)}")

    cloud = metadata.get("cloud_provider") or ""
    env_type = metadata.get("env_type") or ""
    action = metadata.get("action") or ""
    ctx_bits = [b for b in (cloud, env_type, action) if b]
    if ctx_bits:
        parts.append(f"Context: {' '.join(ctx_bits)}")

    factors = metadata.get("contributing_factors") or []
    if factors:
        parts.append("Contributing factors: " + "; ".join(str(f) for f in factors))

    failed = metadata.get("failed_tasks") or []
    for task in failed:
        if not isinstance(task, dict):
            continue
        name = task.get("task") or "unknown task"
        err = task.get("error_message") or ""
        action_name = task.get("task_action") or ""
        line = f"Failed task: {name}"
        if action_name:
            line += f" ({action_name})"
        if err:
            line += f" — {err}"
        parts.append(line)

    return "\n".join(parts).strip()


def _build_mem0_config(config: Any) -> dict[str, Any]:
    """Build mem0 config dict from Config's pgvector settings.

    Uses Vertex AI for both LLM (Claude via litellm) and embeddings.
    Requires ANTHROPIC_VERTEX_PROJECT_ID and CLOUD_ML_REGION env vars.
    """
    vertex_project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") or ""
    vertex_region = os.environ.get("CLOUD_ML_REGION") or "us-east5"

    if vertex_project:
        os.environ.setdefault("VERTEXAI_PROJECT", vertex_project)
    os.environ.setdefault("VERTEXAI_LOCATION", vertex_region)

    return {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "host": config.pgvector_host,
                "port": config.pgvector_port,
                "dbname": config.pgvector_db_name,
                "user": config.pgvector_db_user,
                "password": config.pgvector_db_password,
                "collection_name": "rca_memories",
            },
        },
        "llm": {
            "provider": "litellm",
            "config": {
                "model": f"vertex_ai/{config.mem0_llm_model}",
            },
        },
        "embedder": {
            "provider": "vertexai",
            "config": {
                "model": "text-embedding-004",
                "embedding_dims": 768,
            },
        },
        "version": "v1.1",
    }


def get_memory_client(config: Any):
    """Return a (cached) mem0 Memory client configured from Config."""
    global _mem0_client
    if _mem0_client is None:
        from mem0 import Memory

        mem0_config = _build_mem0_config(config)
        _mem0_client = Memory.from_config(config_dict=mem0_config)
    return _mem0_client


def _build_rca_message(metadata: dict[str, Any]) -> str:
    """Build a narrative message for mem0 add() from RCA metadata."""
    parts = [build_embedding_text(metadata)]

    recommendations = metadata.get("recommendation_count") or 0
    if recommendations:
        parts.append(f"Recommendations: {recommendations}")

    github_paths = metadata.get("github_paths") or []
    if github_paths:
        parts.append(f"Affected files: {', '.join(github_paths)}")

    return "\n".join(parts)


def _rca_metadata_for_mem0(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract the metadata fields to attach to a mem0 memory."""
    return {
        "root_cause_category": metadata.get("root_cause_category") or "",
        "platform": metadata.get("platform") or "",
        "cloud_provider": metadata.get("cloud_provider") or "",
        "env_type": metadata.get("env_type") or "",
        "action": metadata.get("action") or "",
        "confidence": metadata.get("confidence") or "",
        "analyzed_at": metadata.get("analyzed_at") or "",
        "cluster": metadata.get("cluster") or "",
        "namespace": metadata.get("namespace") or "",
    }


def store_rca_memory(
    config: Any,
    job_id: str,
    job_context: dict[str, Any],
    github_context: dict[str, Any] | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Store a completed RCA in mem0. Returns mem0's add() response."""
    metadata = build_rca_metadata(job_context, github_context, summary)
    message = _build_rca_message(metadata)

    if not message.strip():
        raise ValueError("RCA produced empty message text; nothing to store")

    catalog_item = metadata.get("catalog_item") or "uncategorized"
    mem0_metadata = _rca_metadata_for_mem0(metadata)

    client = get_memory_client(config)
    result = client.add(
        messages=[{"role": "user", "content": message}],
        user_id=catalog_item,
        agent_id="rca",
        run_id=str(job_id),
        metadata=mem0_metadata,
    )

    return result


def search_similar(
    config: Any,
    query: str,
    limit: int = 5,
    category: str | None = None,
    catalog_item: str | None = None,
) -> list[dict[str, Any]]:
    """Search mem0 for similar RCA memories.

    Maps catalog_item to user_id and category to metadata filter.
    Returns results normalized to a consistent format.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query text must be a non-empty string.")

    client = get_memory_client(config)

    kwargs: dict[str, Any] = {
        "query": query,
        "top_k": limit,
        "agent_id": "rca",
    }
    if catalog_item:
        kwargs["user_id"] = catalog_item
    if category:
        kwargs["filters"] = {"root_cause_category": category}

    raw_results = client.search(**kwargs)

    results: list[dict[str, Any]] = []
    for r in raw_results.get("results", raw_results) if isinstance(raw_results, dict) else raw_results:
        mem = r if isinstance(r, dict) else {}
        mem_metadata = mem.get("metadata") or {}
        results.append({
            "id": mem.get("id", ""),
            "memory": mem.get("memory", ""),
            "root_cause_category": mem_metadata.get("root_cause_category", ""),
            "confidence": mem_metadata.get("confidence", ""),
            "analyzed_at": mem_metadata.get("analyzed_at", ""),
            "platform": mem_metadata.get("platform", ""),
            "cloud_provider": mem_metadata.get("cloud_provider", ""),
            "action": mem_metadata.get("action", ""),
            "score": mem.get("score", 0.0),
        })

    return results


def store_batch_rca_memory(config: Any, job: dict[str, Any]) -> dict[str, Any]:
    """Store a batch RCA result in mem0. Simplified interface for store_report.py."""
    job_id = str(job.get("job_id", ""))
    catalog_item = job.get("catalog_item") or "uncategorized"

    parts = []
    category = job.get("root_cause_category") or ""
    summary = job.get("root_cause_summary") or ""
    if category or summary:
        parts.append(f"Root cause ({category}): {summary}".strip())
    if catalog_item and catalog_item != "uncategorized":
        parts.append(f"Catalog: {catalog_item}")

    message = "\n".join(parts)
    if not message.strip():
        raise ValueError(f"Batch job {job_id} has no root cause content to store")

    mem0_metadata = {
        "root_cause_category": category,
        "confidence": job.get("confidence") or "",
        "platform": job.get("platform") or "",
        "cloud_provider": job.get("cloud_provider") or "",
    }

    client = get_memory_client(config)
    return client.add(
        messages=[{"role": "user", "content": message}],
        user_id=catalog_item,
        agent_id="rca",
        run_id=job_id,
        metadata=mem0_metadata,
    )


def reset_client() -> None:
    """Reset the cached mem0 client (for testing)."""
    global _mem0_client
    _mem0_client = None
