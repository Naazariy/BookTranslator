"""
Stopping criteria for cooperative inference cancellation in PyTorch/Transformers generation loops.
"""
from typing import Optional, Any
import logging

try:
    import torch
    from transformers import StoppingCriteria
except ImportError:
    # Graceful fallback if transformers/torch is imported in lightweight testing
    class StoppingCriteria:  # type: ignore
        pass


logger = logging.getLogger(__name__)


class CancellationTokenStoppingCriteria(StoppingCriteria):
    """
    Halts PyTorch autoregressive token generation immediately if cancellation is requested
    via a CancellationToken or threading.Event.
    """
    def __init__(self, cancel_token: Optional[Any] = None):
        self.cancel_token = cancel_token

    def _is_cancelled(self) -> bool:
        if self.cancel_token is None:
            return False
        # Support CancellationToken with is_cancelled() method or property
        if hasattr(self.cancel_token, "is_cancelled"):
            checker = getattr(self.cancel_token, "is_cancelled")
            return checker() if callable(checker) else bool(checker)
        # Support threading.Event with is_set() method
        if hasattr(self.cancel_token, "is_set"):
            return self.cancel_token.is_set()
        # Support direct boolean attribute or flag
        if hasattr(self.cancel_token, "cancelled"):
            return bool(self.cancel_token.cancelled)
        return False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        if self._is_cancelled():
            logger.info("CancellationTokenStoppingCriteria: Cancellation requested. Halting generation loop.")
            return True
        return False
