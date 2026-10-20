"""Resolve thinking options without mutating the caller's template dictionary."""


def resolve_thinking(enable_thinking, chat_template_args, think_end_token):
    template = dict(chat_template_args or {})
    nested = template.pop("enable_thinking", None)
    resolved = enable_thinking if enable_thinking is not None else nested
    if resolved is not None and type(resolved) is not bool:
        raise ValueError("enable_thinking must be a boolean or None")
    if resolved and think_end_token is None:
        raise ValueError("enable_thinking=True requires think_end_token")
    return resolved, template
