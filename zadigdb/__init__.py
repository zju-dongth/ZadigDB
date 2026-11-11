"""
ZadigDB: Optimizing Semantic Queries over Multi-Camera Video.
"""

from .schema import (
    CameraTopology,
    CompetitiveProvenanceView,
    OperatorType,
    PhysicalOperatorNode,
    PhysicalPlanDAG,
    ProvenanceEdge,
    TokenResolutionTier,
    VisualDetectionSlice,
)
from .admission_gate import (
    AdmissionAuditReport,
    ClauseAdmissionGate,
    theoretical_coupled_provenance_recall,
    theoretical_recall_collapse_top1,
)
from .buffer_pool import BufferPoolTelemetry, BufferSlotEntry, ProgressiveVisualBufferPool
from .algebra import SpatioTemporalSemanticAlgebra, estimate_empirical_tier0_hubs
from .synthesizer import DeclarativeVideoQuery, TopologyAwarePlanSynthesizer
from .pipeline import QueryExecutionResult, ZadigDBEngine
from .vlm_backend import (
    DEFAULT_CALIBRATED_TIMINGS,
    DEFAULT_PREDICATE_PROMPTS,
    Qwen25VLHardwareBackend,
    VisualKVSlot,
    build_tier_image,
)

__all__ = [
    "CameraTopology",
    "CompetitiveProvenanceView",
    "OperatorType",
    "PhysicalOperatorNode",
    "PhysicalPlanDAG",
    "ProvenanceEdge",
    "TokenResolutionTier",
    "VisualDetectionSlice",
    "AdmissionAuditReport",
    "ClauseAdmissionGate",
    "theoretical_coupled_provenance_recall",
    "theoretical_recall_collapse_top1",
    "BufferPoolTelemetry",
    "BufferSlotEntry",
    "ProgressiveVisualBufferPool",
    "SpatioTemporalSemanticAlgebra",
    "estimate_empirical_tier0_hubs",
    "DeclarativeVideoQuery",
    "TopologyAwarePlanSynthesizer",
    "QueryExecutionResult",
    "ZadigDBEngine",
    "DEFAULT_CALIBRATED_TIMINGS",
    "DEFAULT_PREDICATE_PROMPTS",
    "Qwen25VLHardwareBackend",
    "VisualKVSlot",
    "build_tier_image",
]
