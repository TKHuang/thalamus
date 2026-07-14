from __future__ import annotations
"""
Claude Code Tool Detection and Registry.

Canonical registry of Claude Code tool names with alias-based normalization,
hallucination filtering, and request detection.
"""

import json

CLAUDE_CODE_TOOLS: dict[str, dict] = {
    'Bash':             {'category': 'core', 'aliases': ['bash', 'shell', 'terminal', 'exec', 'run_command', 'run', 'run_terminal_command']},
    'Read':             {'category': 'core', 'aliases': ['read', 'read_file', 'ReadFile', 'readfile', 'read_file_v2']},
    'Write':            {'category': 'core', 'aliases': ['write', 'write_file', 'WriteFile', 'writefile', 'create_file', 'edit_file_v2']},
    'Edit':             {'category': 'core', 'aliases': ['edit', 'file_editor', 'EditFile', 'editfile', 'MultiEdit', 'multiedit', 'edit_file', 'apply_diff', 'apply_agent_diff']},
    'MultiEdit':        {'category': 'core', 'aliases': ['multiedit', 'multi_edit']},
    'Glob':             {'category': 'core', 'aliases': ['glob', 'find_files', 'file_search', 'findfiles', 'glob_file_search']},
    'Grep':             {'category': 'core', 'aliases': ['grep', 'search', 'codebase_search', 'code_search', 'ripgrep_raw_search']},
    'NotebookEdit':     {'category': 'core', 'aliases': ['notebookedit', 'notebook_edit', 'notebook']},
    'TodoWrite':        {'category': 'task', 'aliases': ['todowrite', 'todo_write', 'todo', 'task_tracker']},
    'TodoRead':         {'category': 'task', 'aliases': ['todoread', 'todo_read']},
    'Agent':            {'category': 'task', 'aliases': ['agent', 'delegate', 'subagent']},
    'Task':             {'category': 'task', 'aliases': ['task']},
    'TaskOutput':       {'category': 'task', 'aliases': ['taskoutput', 'task_output']},
    'KillShell':        {'category': 'task', 'aliases': ['killshell', 'kill_shell', 'kill']},
    'TaskStop':         {'category': 'task', 'aliases': ['taskstop', 'task_stop']},
    'ListDir':          {'category': 'core', 'aliases': ['listdir', 'list_dir', 'list_dir_v2', 'ls']},
    'DeleteFile':       {'category': 'core', 'aliases': ['deletefile', 'delete_file']},
    'WebSearch':        {'category': 'web',  'aliases': ['websearch', 'web_search', 'search_web']},
    'WebFetch':         {'category': 'web',  'aliases': ['webfetch', 'web_fetch', 'fetch_url', 'fetch']},
    'AskUserQuestion':  {'category': 'ui',   'aliases': ['askuserquestion', 'ask_user_question', 'question', 'ask']},
    'EnterPlanMode':    {'category': 'ui',   'aliases': ['enterplanmode', 'enter_plan_mode', 'plan_mode']},
    'ExitPlanMode':     {'category': 'ui',   'aliases': ['exitplanmode', 'exit_plan_mode']},
    'Sleep':            {'category': 'misc', 'aliases': ['sleep', 'wait']},
    'Computer':         {'category': 'misc', 'aliases': ['computer', 'browser']},
    'LSP':              {'category': 'misc', 'aliases': ['lsp', 'language_server']},
    'Skill':            {'category': 'misc', 'aliases': ['skill']},
    'Config':           {'category': 'misc', 'aliases': ['config', 'configuration']},
    'ListMcpResources': {'category': 'mcp',  'aliases': ['listmcpresources', 'list_mcp_resources']},
    'ReadMcpResource':  {'category': 'mcp',  'aliases': ['readmcpresource', 'read_mcp_resource']},
    'ToolSearch':       {'category': 'mcp',  'aliases': ['toolsearch', 'tool_search']},
    'Mcp':              {'category': 'mcp',  'aliases': ['mcp']},
    'BatchTool':        {'category': 'misc', 'aliases': ['batchtool', 'batch_tool', 'batch']},
}

_alias_to_canonical: dict[str, str] = {}
_lower_to_canonical: dict[str, str] = {}

for _canonical, _info in CLAUDE_CODE_TOOLS.items():
    _lower_to_canonical[_canonical.lower()] = _canonical
    for _alias in _info['aliases']:
        _alias_to_canonical[_alias.lower()] = _canonical


def normalize_tool_name(raw_name: str | None) -> dict:
    if not raw_name:
        return {'normalized': raw_name, 'was_fixed': False, 'original': raw_name}
    if not isinstance(raw_name, str):
        return {'normalized': '', 'was_fixed': False, 'original': raw_name}

    if raw_name in CLAUDE_CODE_TOOLS:
        return {'normalized': raw_name, 'was_fixed': False, 'original': raw_name}

    lower = raw_name.lower()

    from_alias = _alias_to_canonical.get(lower)
    if from_alias:
        return {'normalized': from_alias, 'was_fixed': True, 'original': raw_name}

    from_lower = _lower_to_canonical.get(lower)
    if from_lower:
        return {'normalized': from_lower, 'was_fixed': True, 'original': raw_name}

    return {'normalized': raw_name, 'was_fixed': False, 'original': raw_name}


def is_claude_code_request(tool_names: list | None) -> bool:
    return bool(tool_names and isinstance(tool_names, list) and len(tool_names) > 0)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f'Non-standard JSON constant: {value}')


def normalize_tool_arguments_as_json_object(raw_arguments) -> tuple[bool, str]:
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError):
            return False, 'arguments_invalid_json'
        if isinstance(parsed, dict):
            try:
                return True, json.dumps(parsed, allow_nan=False)
            except (TypeError, ValueError):
                return False, 'arguments_invalid_json'
        return False, 'arguments_not_object'

    if isinstance(raw_arguments, dict):
        try:
            return True, json.dumps(raw_arguments, allow_nan=False)
        except (TypeError, ValueError):
            return False, 'arguments_invalid_json'

    return False, 'arguments_not_object'


def post_process_tool_calls(tool_calls: list | None, valid_names: list | None) -> dict:
    if not tool_calls:
        return {
            'processed': [],
            'stats': {'passed': 0, 'normalized': 0, 'filtered': 0, 'invalid_arguments_filtered': 0},
        }

    exact_name_map = {str(n).lower(): n for n in (valid_names or [])}
    results = []
    stats = {'passed': 0, 'normalized': 0, 'filtered': 0, 'invalid_arguments_filtered': 0}

    for tc in tool_calls:
        func = tc.get('function') or {}
        raw_name = func.get('name') or tc.get('name') or ''
        info = normalize_tool_name(raw_name)
        normalized = info['normalized']
        was_fixed = info['was_fixed']

        if was_fixed:
            stats['normalized'] += 1

        exact_name = exact_name_map.get(str(raw_name or '').lower())
        if exact_name is None:
            exact_name = exact_name_map.get(str(normalized or '').lower())
        if exact_name is None:
            stats['filtered'] += 1
            continue

        raw_arguments = func.get('arguments') if 'arguments' in func else tc.get('arguments')
        ok, value = normalize_tool_arguments_as_json_object(raw_arguments)
        if not ok:
            stats['invalid_arguments_filtered'] += 1
            continue

        merged_func = {}
        if isinstance(func, dict):
            merged_func.update(func)
        merged_func['name'] = exact_name
        merged_func['arguments'] = value

        result = dict(tc)
        result['function'] = merged_func
        results.append(result)
        stats['passed'] += 1

    return {'processed': results, 'stats': stats}
