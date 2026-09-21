"""
Monitoring & Observability — LangSmith integration + metrics.

Features:
- LangSmith tracing (production-grade observability)
- Metrics collection (latency, error rate, token usage)
- Health metrics endpoint
- Prometheus metrics (coming soon)
"""
import time
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict

from config.settings import settings


@dataclass
class Metrics:
    """System metrics."""
    total_requests: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0
    requests_per_endpoint: Dict[str, int] = field(default_factory=dict)
    errors_per_endpoint: Dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """
    Collect and expose system metrics.
    
    In production, use Prometheus + Grafana.
    This is a lightweight in-memory collector for development.
    """

    def __init__(self):
        self.metrics = Metrics()
        self.request_latencies: Dict[str, List[float]] = defaultdict(list)

    def record_request(self, endpoint: str, latency_ms: float, tokens: int = 0, error: bool = False):
        """Record a request."""
        self.metrics.total_requests += 1
        self.metrics.total_latency_ms += latency_ms
        self.metrics.total_tokens += tokens

        # Per-endpoint stats
        if endpoint not in self.metrics.requests_per_endpoint:
            self.metrics.requests_per_endpoint[endpoint] = 0
        self.metrics.requests_per_endpoint[endpoint] += 1

        # Track latency
        self.request_latencies[endpoint].append(latency_ms)

        # Track errors
        if error:
            self.metrics.total_errors += 1
            if endpoint not in self.metrics.errors_per_endpoint:
                self.metrics.errors_per_endpoint[endpoint] = 0
            self.metrics.errors_per_endpoint[endpoint] += 1

    def get_stats(self) -> Dict:
        """Get current metrics."""
        avg_latency = (
            self.metrics.total_latency_ms / self.metrics.total_requests
            if self.metrics.total_requests > 0
            else 0
        )

        error_rate = (
            self.metrics.total_errors / self.metrics.total_requests * 100
            if self.metrics.total_requests > 0
            else 0
        )

        return {
            "total_requests": self.metrics.total_requests,
            "total_errors": self.metrics.total_errors,
            "error_rate_pct": round(error_rate, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "total_tokens": self.metrics.total_tokens,
            "requests_per_endpoint": self.metrics.requests_per_endpoint,
            "errors_per_endpoint": self.metrics.errors_per_endpoint,
        }


class LangSmithTracer:
    """
    LangSmith tracing integration.
    
    In production, set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY.
    This enables full trace of every RAG step.
    """

    def __init__(self):
        self.enabled = False
        self._check_langsmith()

    def _check_langsmith(self):
        """Check if LangSmith is configured."""
        import os
        if os.getenv("LANGCHAIN_TRACING_V2") == "true":
            self.enabled = True
            print("✅ LangSmith tracing enabled")

    def trace_rag_step(self, step_name: str, inputs: dict, outputs: dict):
        """
        Trace a RAG step.
        
        In production, this sends data to LangSmith.
        """
        if not self.enabled:
            return

        # In production, LangChain automatically traces when
        # LANGCHAIN_TRACING_V2=true is set.
        # This is just a placeholder for custom tracing.
        pass


# Global instances
metrics_collector = MetricsCollector()
langsmith_tracer = LangSmithTracer()
