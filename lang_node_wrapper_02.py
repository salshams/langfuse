from functools import wraps
import langfuse
import os
import yaml
import asyncio
import logging

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'langfuse_nodes.yaml')

try:
    with open(os.path.normpath(_CONFIG_PATH), 'r', encoding='utf-8') as f:
        NODE_CONFIG = yaml.safe_load(f) or {}
        if not isinstance(NODE_CONFIG, dict):
            NODE_CONFIG = {}
except FileNotFoundError:
    NODE_CONFIG = {}

CREATE_PROMPT_NODE_PREVIEW_CHARS = None


def _summarize_value(v, max_str_preview=200):
    try:
        if v is None:
            return None
        if hasattr(v, 'shape'):
            return {
                'type': type(v).__name__,
                'shape': getattr(v, 'shape', None)
            }
        if isinstance(v, (list, tuple, set)):
            return {'type': type(v).__name__, 'len': len(v)}
        if isinstance(v, dict):
            return {'type': 'dict', 'keys_sample': list(v.keys())[:20]}
        if isinstance(v, str):
            preview = v if max_str_preview is None or len(v) <= max_str_preview else v[:max_str_preview] + '...'
            return {'type': 'str', 'len': len(v), 'preview': preview}
        if hasattr(v, 'model_dump'):
            return {'type': type(v).__name__, 'keys': list(v.model_dump().keys())[:20]}
        return {'type': type(v).__name__, 'repr': repr(v)[:200]}
    except Exception as e:
        return {'error': str(e)}


def _get_attr_from_obj(obj, path):
    cur = obj
    for part in path.split('.'):
        if cur is None:
            return None
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
    return cur


def _filter_snapshot(source_obj, paths, node_name, truncate=True):
    out = {}
    for p in paths:
        val = _get_attr_from_obj(source_obj, p)
        max_preview = None if (node_name == 'create_prompt_node' and not truncate) else 200
        out[p] = _summarize_value(val, max_preview)
    return out


def wrap_node(node_callable, node_name=None):
    node_name = node_name or getattr(node_callable, '__name__', 'node')
    cfg = NODE_CONFIG.get(node_name, NODE_CONFIG.get('_default', {}))

    input_paths = cfg.get('input', [])
    output_paths = cfg.get('output', [])
    truncate_input = cfg.get('truncate_input', True)
    truncate_output = cfg.get('truncate_output', True)

    is_coro = asyncio.iscoroutinefunction(node_callable)

    async def _run_async(*args, **kwargs):
        client = langfuse.get_client()
        with client.start_as_current_span(name=node_name) as span:
            span.update(metadata={
                "node": node_name,
                "truncate_input": truncate_input,
                "truncate_output": truncate_output
            })

            state = args[0] if args else kwargs.get('state')
            span.update(input=_filter_snapshot(state, input_paths, node_name, truncate_input))

            try:
                result = await node_callable(*args, **kwargs)
            except Exception as e:
                span.update(output={"error": str(e)})
                raise

            if isinstance(result, dict):
                span.update(output=_filter_snapshot(result, output_paths, node_name, truncate_output))
            else:
                span.update(output={"result_type": type(result).__name__})

            return result

    def _run_sync(*args, **kwargs):
        client = langfuse.get_client()
        with client.start_as_current_span(name=node_name) as span:
            span.update(metadata={
                "node": node_name,
                "truncate_input": truncate_input,
                "truncate_output": truncate_output
            })

            state = args[0] if args else kwargs.get('state')
            span.update(input=_filter_snapshot(state, input_paths, node_name, truncate_input))

            try:
                result = node_callable(*args, **kwargs)
            except Exception as e:
                span.update(output={"error": str(e)})
                raise

            if isinstance(result, dict):
                span.update(output=_filter_snapshot(result, output_paths, node_name, truncate_output))
            else:
                span.update(output={"result_type": type(result).__name__})

            return result

    return wraps(node_callable)(_run_async if is_coro else _run_sync)
