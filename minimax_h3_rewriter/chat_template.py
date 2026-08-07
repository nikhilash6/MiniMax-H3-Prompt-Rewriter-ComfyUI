"""Rendering a GGUF's own chat template, the way Transformers would.

The upstream reference passes ``enable_thinking=False`` to
``apply_chat_template``; without it Qwen3.6 spends hundreds of tokens reasoning
before it starts the rewrite. llama-cpp-python's chat formatter has no way to
forward that flag, so the template is taken out of the GGUF metadata and
rendered here, in a sandbox set up like the one Transformers uses.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

TEMPLATE_KEY = "tokenizer.chat_template"
BOS_KEY = "tokenizer.ggml.bos_token_id"
EOS_KEY = "tokenizer.ggml.eos_token_id"


class TemplateError(RuntimeError):
    pass


def _environment():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    def tojson(value, indent=None, **_kwargs):
        return json.dumps(value, ensure_ascii=False, indent=indent)

    def strftime_now(fmt):
        # Templates that stamp a date are rare here; a fixed value keeps the
        # rendered prompt deterministic, which the seed contract depends on.
        return ""

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    env.filters["tojson"] = tojson
    return env


def render(
    template: str,
    messages: list[dict[str, str]],
    bos_token: str = "",
    eos_token: str = "",
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    if not template:
        raise TemplateError("the model file carries no chat template")

    rendered = _environment().from_string(template).render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        bos_token=bos_token,
        eos_token=eos_token,
        tools=None,
    )
    return rendered


def from_metadata(metadata: dict, messages: list[dict[str, str]], **kwargs) -> str:
    """Render using the ``tokenizer.chat_template`` entry of a GGUF's metadata."""
    template = metadata.get(TEMPLATE_KEY) or metadata.get("chat_template")
    if not template:
        raise TemplateError(
            "this GGUF has no embedded chat template, so the rewriter cannot build its prompt"
        )
    return render(template, messages, **kwargs)
