"""
Shell completion support for YinshML CLI.

Provides bash and zsh completion for commands, options, and experiment IDs.
"""

import click
import os
from typing import List, Dict, Any
from .utils import get_experiment_tracker
from .config import get_config


def get_experiment_ids() -> List[str]:
    """Get list of experiment IDs for completion."""
    try:
        tracker = get_experiment_tracker()
        experiments = tracker.query_experiments(limit=100)
        return [str(exp['id']) for exp in experiments]
    except Exception:
        return []


def get_experiment_statuses() -> List[str]:
    """Get list of valid experiment statuses."""
    return ['running', 'done', 'failed', 'paused', 'cancelled', 'pending']


def get_output_formats() -> List[str]:
    """Get list of valid output formats."""
    return ['table', 'json', 'csv']


def experiment_id_completion(ctx, param, incomplete):
    """Completion function for experiment IDs."""
    experiment_ids = get_experiment_ids()
    return [id for id in experiment_ids if id.startswith(incomplete)]


def status_completion(ctx, param, incomplete):
    """Completion function for status values."""
    statuses = get_experiment_statuses()
    return [status for status in statuses if status.startswith(incomplete)]


def format_completion(ctx, param, incomplete):
    """Completion function for output formats."""
    formats = get_output_formats()
    return [fmt for fmt in formats if fmt.startswith(incomplete)]


def config_file_completion(ctx, param, incomplete):
    """Completion function for configuration files."""
    # Complete JSON files in current directory and common config directories
    import glob
    patterns = [
        f"{incomplete}*.json",
        f"config/{incomplete}*.json",
        f"configs/{incomplete}*.json",
        f"./{incomplete}*.json"
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    
    return [f for f in files if os.path.isfile(f)]


def setup_completion():
    """Set up shell completion for the CLI."""
    # This function can be called to install completion scripts
    completion_script = """
# YinshML CLI completion
_yinsh_track_completion() {
    local IFS=$'\\n'
    COMPREPLY=( $( env COMP_WORDS="${COMP_WORDS[*]}" \\
                   COMP_CWORD=$COMP_CWORD \\
                   _YINSH_TRACK_COMPLETE=complete $1 ) )
    return 0
}

complete -F _yinsh_track_completion -o default yinsh-track
"""
    
    return completion_script


def generate_completion_script(shell: str = 'bash') -> str:
    """Generate completion script for specified shell."""
    if shell == 'bash':
        return """
# Bash completion for yinsh-track
_yinsh_track_completion() {
    local cur prev words cword
    _init_completion || return

    case $prev in
        --status|-s)
            COMPREPLY=( $(compgen -W "running done failed paused cancelled pending" -- "$cur") )
            return 0
            ;;
        --format|-f)
            COMPREPLY=( $(compgen -W "table json csv" -- "$cur") )
            return 0
            ;;
        --config|--config-file)
            COMPREPLY=( $(compgen -f -X "!*.json" -- "$cur") )
            return 0
            ;;
        show|compare|reproduce)
            # Complete experiment IDs
            local ids=$(yinsh-track list --format json 2>/dev/null | grep -o '"id":[0-9]*' | cut -d: -f2 | tr '\\n' ' ')
            COMPREPLY=( $(compgen -W "$ids" -- "$cur") )
            return 0
            ;;
    esac

    case $cword in
        1)
            COMPREPLY=( $(compgen -W "start list compare reproduce search config" -- "$cur") )
            ;;
        *)
            case ${words[1]} in
                start)
                    COMPREPLY=( $(compgen -W "--description --tags --config-file --parameter --help" -- "$cur") )
                    ;;
                list)
                    COMPREPLY=( $(compgen -W "--status --tags --limit --format --sort --reverse --date-from --date-to --help" -- "$cur") )
                    ;;
                compare)
                    COMPREPLY=( $(compgen -W "--metrics --format --include-config --statistical --help" -- "$cur") )
                    ;;
                reproduce)
                    COMPREPLY=( $(compgen -W "--output-dir --dry-run --script-only --force --skip-env --skip-data --help" -- "$cur") )
                    ;;
                search)
                    COMPREPLY=( $(compgen -W "--query --status --tags --metric --metric-min --metric-max --date-from --date-to --limit --format --help" -- "$cur") )
                    ;;
            esac
            ;;
    esac
}

complete -F _yinsh_track_completion yinsh-track
"""
    
    elif shell == 'zsh':
        return """
#compdef yinsh-track

_yinsh_track() {
    local context state state_descr line
    local -A opt_args

    _arguments -C \\
        '1:command:->commands' \\
        '*::arg:->args' && return 0

    case $state in
        commands)
            _values 'yinsh-track commands' \\
                'start[Create new experiment]' \\
                'list[List experiments]' \\
                'compare[Compare experiments]' \\
                'reproduce[Reproduce experiment]' \\
                'search[Search experiments]' \\
                'config[Show configuration]'
            ;;
        args)
            case $words[1] in
                start)
                    _arguments \\
                        '--description[Description]:description:' \\
                        '*--tags[Tags]:tag:' \\
                        '--config-file[Config file]:file:_files -g "*.json"' \\
                        '*--parameter[Parameter]:parameter:' \\
                        '--help[Show help]'
                    ;;
                list)
                    _arguments \\
                        '--status[Status]:status:(running done failed paused cancelled pending)' \\
                        '--tags[Tags]:tags:' \\
                        '--limit[Limit]:limit:' \\
                        '--format[Format]:format:(table json csv)' \\
                        '--sort[Sort]:sort:(id name status created updated)' \\
                        '--reverse[Reverse order]' \\
                        '--date-from[Date from]:date:' \\
                        '--date-to[Date to]:date:' \\
                        '--help[Show help]'
                    ;;
                compare)
                    _arguments \\
                        '*:experiment_id:' \\
                        '*--metrics[Metrics]:metric:' \\
                        '--format[Format]:format:(table json csv)' \\
                        '--include-config[Include config]' \\
                        '--statistical[Statistical analysis]' \\
                        '--help[Show help]'
                    ;;
                reproduce)
                    _arguments \\
                        ':experiment_id:' \\
                        '--output-dir[Output directory]:directory:_directories' \\
                        '--dry-run[Dry run]' \\
                        '--script-only[Script only]' \\
                        '--force[Force]' \\
                        '--skip-env[Skip environment]' \\
                        '--skip-data[Skip data]' \\
                        '--help[Show help]'
                    ;;
                search)
                    _arguments \\
                        '--query[Query]:query:' \\
                        '--status[Status]:status:(running done failed paused cancelled pending)' \\
                        '--tags[Tags]:tags:' \\
                        '--metric[Metric]:metric:' \\
                        '--metric-min[Metric minimum]:value:' \\
                        '--metric-max[Metric maximum]:value:' \\
                        '--date-from[Date from]:date:' \\
                        '--date-to[Date to]:date:' \\
                        '--limit[Limit]:limit:' \\
                        '--format[Format]:format:(table json csv)' \\
                        '--help[Show help]'
                    ;;
            esac
            ;;
    esac
}

_yinsh_track "$@"
"""
    
    else:
        raise ValueError(f"Unsupported shell: {shell}")


def install_completion(shell: str = 'bash') -> str:
    """Generate instructions for installing completion."""
    script = generate_completion_script(shell)
    
    if shell == 'bash':
        instructions = f"""
To install bash completion for yinsh-track:

1. Save the completion script:
   yinsh-track --install-completion bash > ~/.yinsh-track-completion.bash

2. Add this line to your ~/.bashrc:
   source ~/.yinsh-track-completion.bash

3. Reload your shell:
   source ~/.bashrc

Or for system-wide installation:
   sudo yinsh-track --install-completion bash > /etc/bash_completion.d/yinsh-track
"""
    
    elif shell == 'zsh':
        instructions = f"""
To install zsh completion for yinsh-track:

1. Create completion directory if it doesn't exist:
   mkdir -p ~/.zsh/completions

2. Save the completion script:
   yinsh-track --install-completion zsh > ~/.zsh/completions/_yinsh-track

3. Add to your ~/.zshrc if not already present:
   fpath=(~/.zsh/completions $fpath)
   autoload -U compinit
   compinit

4. Reload your shell:
   source ~/.zshrc
"""
    
    return instructions 