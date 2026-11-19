"""可选评测结果存储，保留原有评测器与评分定义。"""

from .core import ClosureError, EvaluationClosure, NotFound, input_hash

__all__ = ["EvaluationClosure", "ClosureError", "NotFound", "input_hash"]
