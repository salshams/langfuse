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
            logger.warning("langfuse_nodes.yaml loaded but top-level is not a mapping; treating as empty.")
            NODE_CONFIG = {}
        else:
            logger.info(f"Loaded langfuse node config from {_CONFIG_PATH}. Keys: {list(NODE_CONFIG.keys())[:200]}")
except FileNotFoundError:
    NODE_CONFIG = {}
    logger.warning(f"langfuse_nodes.yaml not found at {_CONFIG_PATH}; NODE_CONFIG empty. Please check file location.")

# None means "no truncation" for prompt length (create_prompt_node)
CREATE_PROMPT_NODE_PREVIEW_CHARS = None

def _summarize_value(v, max_str_preview=200):
    try:
        if v is None:
            return None
        # pandas DataFrame-ish
        if hasattr(v, 'shape'):
            try:
                nrows = int(v.shape[0])
            except Exception:
                nrows = None
            try:
                ncols = int(v.shape[1]) if len(getattr(v, 'shape')) > 1 else None
            except Exception:
                ncols = None
            cols = None
            try:
                if hasattr(v, 'columns'):
                    cols = list(map(str, list(getattr(v, 'columns')[:10])))
            except Exception:
                cols = None
            return {'type': type(v).__name__, 'n_rows': nrows, 'n_cols': ncols, 'columns_sample': cols}
        if isinstance(v, (list, tuple, set)):
            return {'type': type(v).__name__, 'len': len(v)}
        if isinstance(v, dict):
            keys_sample = list(v.keys())[:20]
            return {'type': 'dict', 'keys_sample': keys_sample}
        if isinstance(v, str):
            if max_str_preview is None:
                preview = v  # no truncation
            else:
                preview = v if len(v) <= max_str_preview else (v[:max_str_preview] + '... [truncated]')
            return {'type': 'str', 'len': len(v), 'preview': preview}
        if isinstance(v, (int, float, bool)):
            return {'type': type(v).__name__, 'value': v}
        if hasattr(v, 'model_dump'):
            try:
                d = v.model_dump()
                return {'type': type(v).__name__, 'keys_sample': list(d.keys())[:20]}
            except Exception:
                pass
        r = repr(v)
        return {'type': type(v).__name__, 'repr': r[:300] + ('...' if len(r) > 300 else '')}
    except Exception:
        try:
            return {'type': type(v).__name__, 'repr': str(v)[:300]}
        except Exception:
            return {'type': type(v).__name__}

def _get_attr_from_obj(obj, path):
    if obj is None:
        return None
    cur = obj
    for part in path.split('.'):
        if cur is None:
            return None
        try:
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                cur = getattr(cur, part, None)
        except Exception:
            return None
    return cur

def _filter_snapshot(source_obj, paths, node_name=None, truncate=True):
    """
    Build a limited snapshot from the source_obj for the given dotted paths.

    - paths: list[str] dotted paths to extract
    - node_name: for special-case logic (e.g., create_prompt_node)
    - truncate: if False => do not truncate string values (max_str_preview=None)
    """

    out = {}
    if not paths:
        return out
    for p in paths:
        try:
            # REDACTION: never capture full prompt/prompt_deck except in create_prompt_node
            if any(keyword in p.lower() for keyword in ('prompt', 'prompt_deck')):
                if node_name != 'create_prompt_node' and truncate:
                    out[p] = {'type': 'str', 'note': 'redacted-prompt'}
                    continue

            val = _get_attr_from_obj(source_obj, p)

            # path-based preview lengths
            max_preview = 120
            if any(keyword in p.lower() for keyword in ('mer_markdown', 'markdown_output', 'llm_response')):
                max_preview = 200

            # Special-casing: create_prompt_node gets a larger or unlimited preview
            if node_name == 'create_prompt_node' and any(k in p.lower() for k in ('mer_markdown', 'prompt')):
                # if CREATE_PROMPT_NODE_PREVIEW_CHARS is None -> no truncation
                max_preview = CREATE_PROMPT_NODE_PREVIEW_CHARS

            # handle global truncate flag
            if not truncate:
                max_preview = None

            out[p] = _summarize_value(val, max_str_preview=max_preview)
        except Exception as e:
            out[p] = {'error': str(e)}
    return out

def wrap_node(node_callable, node_name=None):

    # If node_name will not be provided in the future addings
    if node_name is None:
        node_name = getattr(node_callable, '__name__', 'node')

    # Loading config from langfuse_nodes.YAML
    # logging for Default and No Config
    cfg = NODE_CONFIG.get(node_name)
    if cfg is None:
        cfg = NODE_CONFIG.get('_default', {})
        if NODE_CONFIG:
            logger.debug(f"No explicit langfuse config for node '{node_name}', using _default.")
        else:
            logger.debug(f"NODE_CONFIG empty; no langfuse snapshot configured for node '{node_name}'.")

    input_paths = cfg.get('input', []) if isinstance(cfg, dict) else []
    output_paths = cfg.get('output', []) if isinstance(cfg, dict) else []

    # Added to control the Truncation (default should be true in langfuse_nodes.YAML)
    # new per-node flags (default True)
    truncate_input = cfg.get('truncate_input', True) if isinstance(cfg, dict) else True
    truncate_output = cfg.get('truncate_output', True) if isinstance(cfg, dict) else True

    # wrap any async and sync nodes correctly, without manual intervention.
    # Just keep it for future changes
    is_coro = asyncio.iscoroutinefunction(node_callable)

    if is_coro:
        @wraps(node_callable)
        async def wrapped(*args, **kwargs):
            try:
                client = langfuse.get_client()
            except Exception:
                client = None
            if client is None:
                logger.debug(f"langfuse client not available - running node '{node_name}' without span.")
                return await node_callable(*args, **kwargs)

            try:
                with client.start_as_current_span(name=node_name) as span:
                    state_obj = args[0] if len(args) > 0 else kwargs.get('state')
                    logger.info(f"Extracted objs from kwargs for \n######################### \n######################### \n node: {node_name} \n args: {args} \n kwarg: {kwargs}: \n state_obj: {state_obj} \n######################### \n######################### \n")
                    input_snapshot = _filter_snapshot(state_obj, input_paths, node_name=node_name, truncate=truncate_input)
                    try:
                        span.update_trace(input=input_snapshot)
                    except Exception:
                        logger.exception("Failed to update span input for node: %s", node_name)

                    try:
                        result = await node_callable(*args, **kwargs)
                    except Exception as e:
                        try:
                            span.update_trace(output={'error': str(e)})
                        except Exception:
                            pass
                        raise

                    output_snapshot = {}
                    if isinstance(result, dict):
                        for p in output_paths:
                            try:
                                if '.' in p:
                                    top, rest = p.split('.', 1)
                                    v = result.get(top)
                                    if v is None:
                                        v = _get_attr_from_obj(state_obj, p)
                                    else:
                                        v = _get_attr_from_obj(v, rest)
                                    # if truncate_output False => pass max_str_preview=None inside _summarize_value path
                                    output_snapshot[p] = _summarize_value(v, max_str_preview=(None if not truncate_output else 200))
                                else:
                                    if p in result:
                                        output_snapshot[p] = _summarize_value(result[p], max_str_preview=(None if not truncate_output else 200))
                                    else:
                                        v = _get_attr_from_obj(state_obj, p)
                                        output_snapshot[p] = _summarize_value(v, max_str_preview=(None if not truncate_output else 200))
                            except Exception as e:
                                output_snapshot[p] = {'error': str(e)}
                    else:
                        for p in output_paths:
                            try:
                                if p.startswith('result.'):
                                    v = _get_attr_from_obj(result, p[len('result.'):])
                                else:
                                    v = _get_attr_from_obj(result, p) or _get_attr_from_obj(state_obj, p)
                                output_snapshot[p] = _summarize_value(v, max_str_preview=(None if not truncate_output else 200))
                            except Exception as e:
                                output_snapshot[p] = {'error': str(e)}

                    try:
                        span.update_trace(output=output_snapshot)
                    except Exception:
                        logger.exception("Failed to update span output for node: %s", node_name)
                    return result
            except Exception:
                logger.exception("Wrapper failing for node '%s' — running node without wrapper as fallback.", node_name)
                return await node_callable(*args, **kwargs)
        return wrapped
    else:
        @wraps(node_callable)
        def wrapped_sync(*args, **kwargs):
            try:
                client = langfuse.get_client()
            except Exception:
                client = None
            if client is None:
                logger.debug(f"langfuse client not available - running node '{node_name}' without span.")
                return node_callable(*args, **kwargs)
            try:
                with client.start_as_current_span(name=node_name) as span:
                    state_obj = args[0] if len(args) > 0 else kwargs.get('state')
                    input_snapshot = _filter_snapshot(state_obj, input_paths, node_name=node_name, truncate=truncate_input)
                    try:
                        span.update_trace(input=input_snapshot)
                    except Exception:
                        logger.exception("Failed to update span input for node: %s", node_name)

                    try:
                        result = node_callable(*args, **kwargs)
                    except Exception as e:
                        try:
                            span.update_trace(output={'error': str(e)})
                        except Exception:
                            pass
                        raise

                    output_snapshot = {}
                    if isinstance(result, dict):
                        for p in output_paths:
                            try:
                                if '.' in p:
                                    top, rest = p.split('.', 1)
                                    v = result.get(top)
                                    if v is None:
                                        v = _get_attr_from_obj(state_obj, p)
                                    else:
                                        v = _get_attr_from_obj(v, rest)
                                    output_snapshot[p] = _summarize_value(v, max_str_preview=(None if not truncate_output else 200))
                                else:
                                    if p in result:
                                        output_snapshot[p] = _summarize_value(result[p], max_str_preview=(None if not truncate_output else 200))
                                    else:
                                        v = _get_attr_from_obj(state_obj, p)
                                        output_snapshot[p] = _summarize_value(v, max_str_preview=(None if not truncate_output else 200))
                            except Exception as e:
                                output_snapshot[p] = {'error': str(e)}
                    else:
                        for p in output_paths:
                            try:
                                if p.startswith('result.'):
                                    v = _get_attr_from_obj(result, p[len('result.'):])
                                else:
                                    v = _get_attr_from_obj(result, p) or _get_attr_from_obj(state_obj, p)
                                output_snapshot[p] = _summarize_value(v, max_str_preview=(None if not truncate_output else 200))
                            except Exception as e:
                                output_snapshot[p] = {'error': str(e)}

                    try:
                        span.update_trace(output=output_snapshot)
                    except Exception:
                        logger.exception("Failed to update span output for node: %s", node_name)
                    return result
            except Exception:
                logger.exception("Wrapper failing for node '%s' — running node without wrapper as fallback.", node_name)
                return node_callable(*args, **kwargs)
        return wrapped_sync
