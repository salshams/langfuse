import os
import logging
from langchain.callbacks.base import BaseCallbackHandler
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

try:
    from opentelemetry.trace import get_current_span
except Exception:
    get_current_span = None

logger = logging.getLogger(__name__)

_ALLOWED = set(os.getenv(
    "LANGFUSE_ALLOW_FULL_LLM_OUTPUT_NODES",
    "azure_llm_node,azure_llm_overview_node"
).split(","))


def _short(s, n=120):
    if s is None:
        return None
    text = s.content if hasattr(s, "content") else str(s)
    return text if len(text) <= n else text[:n] + "..."


def _allow_full():
    try:
        span = get_current_span()
        return span and span.name in _ALLOWED
    except Exception:
        return False


class LangfuseFilteredCallback(BaseCallbackHandler):
    def __init__(self):
        self.inner = LangfuseCallbackHandler()

    def on_llm_start(self, serialized, prompts, **kwargs):
        return self.inner.on_llm_start(serialized, [_short(p, 80) for p in prompts])

    def on_llm_end(self, response, **kwargs):
        content = getattr(response, "content", response)
        if not _allow_full():
            content = _short(content)
        return self.inner.on_llm_end({"content": content})

    def on_chain_start(self, serialized, inputs, **kwargs):
        return self.inner.on_chain_start(serialized, _short(inputs, 200))

    def on_chain_end(self, outputs, **kwargs):
        return self.inner.on_chain_end({}, _short(outputs, 200))
