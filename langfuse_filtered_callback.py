from typing import Any, Tuple, Dict
import logging
import os

from langchain.callbacks.base import BaseCallbackHandler
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

# optional imports: be defensive
try:
    from langchain.schema import Generation, LLMResult, AIMessage
except Exception:
    Generation = None
    LLMResult = None
    AIMessage = None

# opentelemetry to inspect current span (best effort)
try:
    from opentelemetry.trace import get_current_span
except Exception:
    get_current_span = None  # fallback if not available

logger = logging.getLogger(__name__)

# Environment variable controls which span names (node names) should allow full LLM outputs
_ALLOWED_FULL_OUTPUT_NODES_ENV = os.getenv("LANGFUSE_ALLOW_FULL_LLM_OUTPUT_NODES", "azure_llm_node,azure_llm_overview_node")
_ALLOWED_FULL_OUTPUT_NODES = set([n.strip() for n in _ALLOWED_FULL_OUTPUT_NODES_ENV.split(",") if n.strip()])

def _short_preview(s: Any, max_chars: int = 80) -> Any:
    if s is None:
        return None
    try:
        # AIMessage-like
        if hasattr(s, "content"):
            text = getattr(s, "content")
            if text is None:
                return "[no-content]"
            return text if len(text) <= max_chars else text[:max_chars] + "... [truncated]"
        # plain string
        if isinstance(s, str):
            return s if len(s) <= max_chars else s[:max_chars] + "... [truncated]"
        # Pydantic/BaseModel -> try to model_dump (v2)
        if hasattr(s, "model_dump"):
            try:
                dd = s.model_dump(exclude_unset=True)
                return _short_preview(str(dd), max_chars=max_chars)
            except Exception:
                pass
        # Fallback to str()
        text = str(s)
        return text if len(text) <= max_chars else text[:max_chars] + "... [truncated]"
    except Exception:
        return "[unserializable]"

def _allowed_full_output_current_span() -> bool:
    """
    Best-effort: return True if the current OpenTelemetry span name is in the allowlist.
    If current span cannot be retrieved, return False.
    """
    try:
        if get_current_span is None:
            return False
        span = get_current_span()
        if span is None:
            return False
        # Newer OTEL APIs expose name on span.get_span_context() or span.name
        name = getattr(span, "name", None)
        if name is None:
            # some SDKs store attributes
            attrs = getattr(span, "attributes", {}) or {}
            name = attrs.get("span.name")
        if not name:
            return False
        # simple match against allowlist
        return name in _ALLOWED_FULL_OUTPUT_NODES
    except Exception:
        logger.debug("Could not determine current span for full-output check", exc_info=True)
        return False

class LangfuseFilteredCallback(BaseCallbackHandler):
    """
    Wrapper that sanitizes callback events before forwarding to langfuse CallbackHandler.
    If the current OpenTelemetry span is a node in the allowlist (env LANGFUSE_ALLOW_FULL_LLM_OUTPUT_NODES),
    then LLM outputs for that call are forwarded *without* aggressive truncation (useful for create_prompt_node and azure_llm_node).
    """

    def __init__(self):
        super().__init__()
        self._inner = LangfuseCallbackHandler()

    def on_llm_start(self, *args, **kwargs) -> Any:
        try:
            # THESE TWO ARE TEMPORARY
            #logger.info(f"Received args: {args}")
            #logger.info(f"Received kwargs: {kwargs}")

            prompts = kwargs.pop("prompts", None)
            # THIS ONE IS AS WELL TEMPORARY
            #logger.info(f"Extracted prompts from kwargs: {prompts}")

            if prompts is None:
                if len(args) >= 2:
                    prompts = args[1]
                elif len(args) == 1 and isinstance(args[0], (list, tuple)):
                    prompts = args[0]
                else:
                    prompts = []

            sanitized_prompts = []
            for p in (prompts or []):
                sanitized_prompts.append(_short_preview(p, max_chars=80) or "[REDACTED PROMPT]")

            serialized = kwargs.pop("serialized", args[0] if len(args) > 0 else {})
            try:
                return self._inner.on_llm_start(serialized, sanitized_prompts, **kwargs)
            except Exception:
                try:
                    return self._inner.on_llm_start(serialized=serialized, inputs=sanitized_prompts, **kwargs)
                except Exception:
                    logger.exception("LangfuseFilteredCallback.on_llm_start inner failed")
                    return None
        except Exception:
            logger.exception("LangfuseFilteredCallback.on_llm_start failed")
            return None

    def on_llm_new_token(self, token: str, **kwargs):
        try:
            return self._inner.on_llm_new_token(token, **kwargs)
        except Exception:
            logger.debug("LangfuseFilteredCallback.on_llm_new_token inner failed", exc_info=True)
            return None

    def on_llm_end(self, *args, **kwargs) -> Any:
        """
        Normalize the LLM response to a minimal LLMResult-like object unless the
        current span is in the allowlist (then allow full content if present).
        """
        try:
            # Extract response: positional or kw
            response = None
            if "response" in kwargs:
                response = kwargs.pop("response")
            elif len(args) >= 1:
                response = args[0]

            allow_full = _allowed_full_output_current_span()

            # If response is LLMResult, try to extract best text
            if LLMResult is not None and isinstance(response, LLMResult):
                try:
                    gen_text = None
                    if getattr(response, "generations", None):
                        last = response.generations[-1]
                        if last and len(last) > 0:
                            g = last[-1]
                            gen_text = getattr(g, "text", None) or getattr(g, "content", None)
                    # Truncate or pass full based on allowlist
                    if allow_full:
                        final_text = gen_text
                    else:
                        final_text = _short_preview(gen_text, max_chars=120)
                    if Generation is not None:
                        gen = Generation(text=final_text)
                        sanitized = LLMResult(generations=[[gen]])
                        return self._inner.on_llm_end(sanitized, **kwargs)
                except Exception:
                    try:
                        return self._inner.on_llm_end(response, **kwargs)
                    except Exception:
                        logger.exception("Inner on_llm_end failed with LLMResult.")
                        return None

            # If it's a dict with 'raw'
            preview_text = None
            if isinstance(response, dict):
                raw = response.get("raw") if response is not None else None
                if isinstance(raw, dict):
                    preview_text = raw.get("content")
                elif hasattr(raw, "content"):
                    preview_text = getattr(raw, "content")
                if preview_text is None:
                    preview_text = response.get("content")
            elif AIMessage is not None and isinstance(response, AIMessage):
                preview_text = response.content
            elif hasattr(response, "content"):
                preview_text = getattr(response, "content")
            else:
                preview_text = _short_preview(response, max_chars=120)

            # If allowed: pass full preview_text
            if allow_full and preview_text is not None:
                full_text = preview_text
            else:
                full_text = _short_preview(preview_text, max_chars=120)

            if Generation is not None and LLMResult is not None:
                try:
                    gen = Generation(text=full_text)
                    llm_result = LLMResult(generations=[[gen]])
                    return self._inner.on_llm_end(llm_result, **kwargs)
                except Exception:
                    logger.debug("Could not build LLMResult; falling back.")
            try:
                return self._inner.on_llm_end({"content": full_text}, **kwargs)
            except Exception:
                logger.exception("LangfuseFilteredCallback.on_llm_end inner failed to accept fallback dict")
                return None

        except Exception:
            logger.exception("LangfuseFilteredCallback.on_llm_end failed to sanitize/forward")
            try:
                return self._inner.on_llm_end({"content": "[REDACTED]"}, **kwargs)
            except Exception:
                return None

    def on_tool_end(self, *args, **kwargs):
        try:
            output = kwargs.pop("output", args[0] if len(args) >= 1 else None)
            if output is not None:
                output_preview = _short_preview(output, max_chars=120)
                return self._inner.on_tool_end(output_preview, **kwargs)
            return self._inner.on_tool_end(*args, **kwargs)
        except Exception:
            logger.debug("LangfuseFilteredCallback.on_tool_end inner failed", exc_info=True)
            return None

    def on_chain_start(self, *args, **kwargs):
        try:
            # normalize like before but minimal
            serialized = kwargs.get("serialized", args[0] if len(args) > 0 else {})
            inputs = kwargs.get("inputs", args[1] if len(args) > 1 else (args[0] if len(args) == 1 else None))
            # sanitize inputs
            try:
                sanitized_inputs = _short_preview(inputs, max_chars=200)
            except Exception:
                sanitized_inputs = {"_error": "on_chain_start_sanitization_failed"}
            try:
                return self._inner.on_chain_start(serialized, sanitized_inputs, **kwargs)
            except Exception:
                return self._inner.on_chain_start(serialized=serialized, inputs=sanitized_inputs, **kwargs)
        except Exception:
            logger.exception("LangfuseFilteredCallback.on_chain_start failed")
            try:
                return self._inner.on_chain_start({}, {"_error": "on_chain_start_failed"})
            except Exception:
                return None

    def on_chain_end(self, *args, **kwargs):
        try:
            output = kwargs.pop("outputs", args[1] if len(args) >= 2 else (args[0] if len(args)==1 else None))
            out_sanitized = _short_preview(output, max_chars=200) if output is not None else {}
            serialized = kwargs.pop("serialized", args[0] if len(args) > 0 else {})
            try:
                return self._inner.on_chain_end(serialized, out_sanitized, **kwargs)
            except Exception:
                try:
                    return self._inner.on_chain_end(serialized=serialized, outputs=out_sanitized, **kwargs)
                except Exception:
                    logger.exception("LangfuseFilteredCallback.on_chain_end inner failed")
                    return None
        except Exception:
            logger.exception("LangfuseFilteredCallback.on_chain_end failed")
            return None
