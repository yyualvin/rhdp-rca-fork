"""Unit tests for the mem0 memory layer (mocked, no live DB required)."""

from unittest.mock import MagicMock, patch

from scripts.memory import (
    _build_rca_message,
    _rca_metadata_for_mem0,
    build_rca_metadata,
    search_similar,
    store_batch_rca_memory,
    store_rca_memory,
    reset_client,
)


def _sample_job_context():
    return {
        "job_id": "12345",
        "status": "failed",
        "guid": "x1234",
        "namespace": "sandbox-x1234-xx-ansiblebu",
        "cluster": "test-cluster",
        "cloud_provider": "aws",
        "env_type": "sandbox",
        "action": "provision",
        "time_window": {"duration_seconds": 300},
        "failed_tasks": [
            {
                "task": "Install dependencies",
                "error_message": "Package not found",
                "task_action": "yum",
            },
        ],
    }


def _sample_github_context():
    return {
        "job_id": "12345",
        "job_metadata": {
            "platform": "openshift_cnv",
            "catalog_item": "ocp4-cluster",
            "env": "test",
        },
    }


def _sample_summary():
    return {
        "job_id": "12345",
        "analyzed_at": "2025-01-15T10:30:45Z",
        "root_cause": {
            "summary": "Missing aws_access_key_id in config",
            "category": "configuration",
            "confidence": "high",
        },
        "evidence": [],
        "recommendations": [{"action": "Add missing variable"}],
        "contributing_factors": ["Missing variable"],
    }


def _mock_config():
    config = MagicMock()
    config.pgvector_host = "localhost"
    config.pgvector_port = 5432
    config.pgvector_db_name = "testdb"
    config.pgvector_db_user = "testuser"
    config.pgvector_db_password = "testpass"
    config.mem0_llm_model = "claude-sonnet-4-20250514"
    config.has_memory_store.return_value = True
    return config


def test_build_rca_message_includes_key_fields():
    metadata = build_rca_metadata(
        _sample_job_context(), _sample_github_context(), _sample_summary()
    )
    message = _build_rca_message(metadata)

    assert "configuration" in message
    assert "Missing aws_access_key_id" in message
    assert "ocp4-cluster" in message
    assert "Recommendations: 1" in message


def test_rca_metadata_for_mem0_extracts_filterable_fields():
    metadata = build_rca_metadata(
        _sample_job_context(), _sample_github_context(), _sample_summary()
    )
    mem0_meta = _rca_metadata_for_mem0(metadata)

    assert mem0_meta["root_cause_category"] == "configuration"
    assert mem0_meta["platform"] == "openshift_cnv"
    assert mem0_meta["cloud_provider"] == "aws"
    assert mem0_meta["confidence"] == "high"
    assert mem0_meta["action"] == "provision"


@patch("scripts.memory.get_memory_client")
def test_store_rca_memory_calls_add_with_correct_params(mock_get_client):
    mock_client = MagicMock()
    mock_client.add.return_value = {"results": [{"id": "mem_123"}]}
    mock_get_client.return_value = mock_client

    config = _mock_config()
    result = store_rca_memory(
        config, "12345", _sample_job_context(), _sample_github_context(), _sample_summary()
    )

    mock_client.add.assert_called_once()
    call_kwargs = mock_client.add.call_args[1]
    assert call_kwargs["user_id"] == "ocp4-cluster"
    assert call_kwargs["agent_id"] == "rca"
    assert call_kwargs["run_id"] == "12345"
    assert call_kwargs["metadata"]["root_cause_category"] == "configuration"


@patch("scripts.memory.get_memory_client")
def test_store_rca_memory_uses_uncategorized_without_catalog_item(mock_get_client):
    mock_client = MagicMock()
    mock_client.add.return_value = {"results": []}
    mock_get_client.return_value = mock_client

    config = _mock_config()
    store_rca_memory(config, "12345", _sample_job_context(), None, _sample_summary())

    call_kwargs = mock_client.add.call_args[1]
    assert call_kwargs["user_id"] == "uncategorized"


@patch("scripts.memory.get_memory_client")
def test_search_similar_maps_catalog_item_to_user_id(mock_get_client):
    mock_client = MagicMock()
    mock_client.search.return_value = []
    mock_get_client.return_value = mock_client

    config = _mock_config()
    search_similar(config, "DNS failure", catalog_item="ocp4-cluster")

    call_kwargs = mock_client.search.call_args[1]
    assert call_kwargs["user_id"] == "ocp4-cluster"
    assert call_kwargs["agent_id"] == "rca"


@patch("scripts.memory.get_memory_client")
def test_search_similar_maps_category_to_filter(mock_get_client):
    mock_client = MagicMock()
    mock_client.search.return_value = []
    mock_get_client.return_value = mock_client

    config = _mock_config()
    search_similar(config, "credential issue", category="credential")

    call_kwargs = mock_client.search.call_args[1]
    assert call_kwargs["filters"] == {"root_cause_category": "credential"}


@patch("scripts.memory.get_memory_client")
def test_search_similar_normalizes_results(mock_get_client):
    mock_client = MagicMock()
    mock_client.search.return_value = [
        {
            "id": "mem_1",
            "memory": "Root cause (configuration): Missing key",
            "score": 0.85,
            "metadata": {
                "root_cause_category": "configuration",
                "confidence": "high",
                "platform": "openshift_cnv",
                "cloud_provider": "aws",
                "action": "provision",
                "analyzed_at": "2025-01-15",
            },
        }
    ]
    mock_get_client.return_value = mock_client

    config = _mock_config()
    results = search_similar(config, "missing key")

    assert len(results) == 1
    assert results[0]["id"] == "mem_1"
    assert results[0]["root_cause_category"] == "configuration"
    assert results[0]["score"] == 0.85


@patch("scripts.memory.get_memory_client")
def test_store_batch_rca_memory(mock_get_client):
    mock_client = MagicMock()
    mock_client.add.return_value = {"results": [{"id": "mem_456"}]}
    mock_get_client.return_value = mock_client

    config = _mock_config()
    job = {
        "job_id": "99999",
        "root_cause_category": "infrastructure",
        "root_cause_summary": "Node pool exhausted",
        "catalog_item": "ocp4-cluster",
        "confidence": "medium",
    }
    store_batch_rca_memory(config, job)

    call_kwargs = mock_client.add.call_args[1]
    assert call_kwargs["user_id"] == "ocp4-cluster"
    assert call_kwargs["run_id"] == "99999"
    assert "Node pool exhausted" in call_kwargs["messages"][0]["content"]
