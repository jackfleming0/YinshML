"""
Reproduce command for recreating experiments.
"""

import click
import os
from pathlib import Path
from typing import Optional

from ..utils import (
    get_experiment_tracker, create_reproduction_script,
    get_config, colorize_status
)


@click.command()
@click.argument('experiment_id', type=int)
@click.option('--output-dir', '-o',
              help='Directory to reproduce experiment in')
@click.option('--force', is_flag=True,
              help='Force reproduction even if environment differs')
@click.option('--dry-run', is_flag=True,
              help='Show what would be done without actually doing it')
@click.option('--skip-env', is_flag=True,
              help='Skip environment recreation')
@click.option('--skip-data', is_flag=True,
              help='Skip data validation/download')
@click.option('--script-only', is_flag=True,
              help='Only generate reproduction script without environment setup')
@click.pass_context
def reproduce(ctx, experiment_id: int, output_dir: Optional[str], 
             force: bool, dry_run: bool, skip_env: bool, skip_data: bool,
             script_only: bool):
    """
    Reproduce an experiment.
    
    Recreates the environment and configuration for EXPERIMENT_ID,
    generating reproduction scripts and validating dependencies.
    
    Examples:
        yinsh-track reproduce 123
        yinsh-track reproduce 123 --output-dir ./reproduced_exp
        yinsh-track reproduce 123 --dry-run --script-only
    """
    config = get_config()
    use_color = config.get('color_output', True)
    
    try:
        # Get tracker and experiment data
        tracker = get_experiment_tracker()
        
        with click.progressbar(length=4, label='Loading experiment data') as bar:
            # Get experiment details
            experiment = tracker.get_experiment(experiment_id)
            if not experiment:
                raise click.ClickException(f"Experiment {experiment_id} not found")
            bar.update(1)
            
            # Validate experiment status
            status = experiment.get('status', 'unknown')
            if status not in ['completed', 'done', 'failed', 'cancelled'] and not force:
                warning_msg = f"Experiment status is '{status}' (not completed)"
                if use_color:
                    warning_msg = click.style(warning_msg, fg='yellow')
                click.echo(f"⚠️  {warning_msg}")
                click.echo("Use --force to reproduce anyway")
                if not click.confirm("Continue reproduction?"):
                    return
            bar.update(1)
            
            # Set up output directory
            if output_dir:
                output_path = Path(output_dir)
            else:
                output_path = Path(f"reproduced_exp_{experiment_id}")
            
            if output_path.exists() and not force:
                if not click.confirm(f"Directory {output_path} exists. Overwrite?"):
                    return
            bar.update(1)
            
            # Get full experiment data including metrics and environment
            experiment_data = tracker.export_experiment_data(
                experiment_id, 
                format='dict', 
                include_metrics=True, 
                include_config=True
            )
            bar.update(1)
        
        # Display experiment information
        _display_experiment_info(experiment_data, use_color)
        
        if dry_run:
            click.echo(f"\n🔍 DRY RUN - Would reproduce experiment to: {output_path}")
            _display_reproduction_plan(experiment_data, output_path, skip_env, skip_data)
            return
        
        # Create output directory
        if not script_only:
            output_path.mkdir(parents=True, exist_ok=True)
            click.echo(f"📁 Created reproduction directory: {output_path}")
        
        # Generate reproduction script
        reproduction_script = create_reproduction_script(experiment_data, str(output_path))
        
        if script_only:
            click.echo("🐍 Reproduction Script:")
            click.echo("=" * 50)
            click.echo(reproduction_script)
            return
        
        # Save reproduction script
        script_path = output_path / f"reproduce_exp_{experiment_id}.py"
        with open(script_path, 'w') as f:
            f.write(reproduction_script)
        os.chmod(script_path, 0o755)  # Make executable
        click.echo(f"✅ Generated reproduction script: {script_path}")
        
        # Save experiment metadata
        metadata_path = output_path / "experiment_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
        click.echo(f"✅ Saved experiment metadata: {metadata_path}")
        
        # Environment recreation
        if not skip_env:
            _setup_environment(experiment_data, output_path, force)
        
        # Data validation (placeholder)
        if not skip_data:
            _validate_data(experiment_data, output_path)
        
        # Generate instructions
        _generate_instructions(experiment_data, output_path, script_path)
        
        click.echo(f"\n🎉 Experiment reproduction setup complete!")
        click.echo(f"📂 Location: {output_path}")
        click.echo(f"🚀 Run: {script_path}")
        
    except Exception as e:
        raise click.ClickException(f"Reproduction failed: {e}")


def _display_experiment_info(experiment_data: dict, use_color: bool = True):
    """Display experiment information."""
    exp_id = experiment_data.get('id', 'Unknown')
    name = experiment_data.get('name', 'Unknown')
    status = experiment_data.get('status', 'unknown')
    created = experiment_data.get('timestamp', 'Unknown')
    
    click.echo(f"\n📋 Experiment Information:")
    click.echo(f"   ID: {exp_id}")
    click.echo(f"   Name: {name}")
    
    status_display = colorize_status(status, use_color) if use_color else status
    click.echo(f"   Status: {status_display}")
    click.echo(f"   Created: {created}")
    
    # Show configuration summary
    config = experiment_data.get('config', {})
    user_config = config.get('user_config', {}) if isinstance(config, dict) else {}
    if user_config:
        click.echo(f"   Parameters: {len(user_config)} configured")
    
    # Show environment info
    environment = experiment_data.get('environment', {})
    if environment:
        git_info = environment.get('git', {})
        if git_info.get('commit'):
            click.echo(f"   Git Commit: {git_info['commit'][:8]}...")
        
        system_info = environment.get('system', {})
        if system_info.get('python_version'):
            click.echo(f"   Python: {system_info['python_version']}")


def _display_reproduction_plan(experiment_data: dict, output_path: Path, 
                              skip_env: bool, skip_data: bool):
    """Display what would be done in reproduction."""
    click.echo("\n📋 Reproduction Plan:")
    click.echo(f"   1. Create directory: {output_path}")
    click.echo(f"   2. Generate reproduction script")
    click.echo(f"   3. Save experiment metadata")
    
    if not skip_env:
        click.echo(f"   4. Set up Python environment")
        click.echo(f"   5. Install required packages")
    else:
        click.echo(f"   4. Skip environment setup")
    
    if not skip_data:
        click.echo(f"   6. Validate data dependencies")
    else:
        click.echo(f"   6. Skip data validation")
    
    click.echo(f"   7. Generate run instructions")


def _setup_environment(experiment_data: dict, output_path: Path, force: bool):
    """Set up reproduction environment."""
    click.echo("\n🐍 Setting up environment...")
    
    environment = experiment_data.get('environment', {})
    system_info = environment.get('system', {})
    
    # Check Python version
    current_python = f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    original_python = system_info.get('python_version', '')
    
    if original_python and current_python not in original_python:
        warning = f"Python version mismatch: current={current_python}, original={original_python}"
        click.echo(f"⚠️  {warning}")
        if not force and not click.confirm("Continue with different Python version?"):
            raise click.ClickException("Environment mismatch. Use --force to override.")
    
    # Generate requirements.txt if packages available
    packages = environment.get('environment', {}).get('installed_packages', {})
    if packages:
        requirements_path = output_path / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write("# Generated requirements for experiment reproduction\n")
            for package, version in packages.items():
                f.write(f"{package}=={version}\n")
        click.echo(f"✅ Generated requirements.txt: {requirements_path}")
        
        # Optionally install packages
        if click.confirm("Install requirements now?"):
            import subprocess
            try:
                subprocess.check_call([
                    os.sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
                ])
                click.echo("✅ Requirements installed successfully")
            except subprocess.CalledProcessError as e:
                click.echo(f"⚠️  Some packages failed to install: {e}")


def _validate_data(experiment_data: dict, output_path: Path):
    """Validate data dependencies."""
    click.echo("\n📊 Validating data dependencies...")
    
    # This is a placeholder for data validation logic
    # In a real implementation, this would:
    # 1. Check for data files referenced in the experiment
    # 2. Validate data checksums if available
    # 3. Download missing data if possible
    # 4. Create data symlinks or copies
    
    click.echo("ℹ️  Data validation not yet implemented")
    click.echo("   Future features:")
    click.echo("   - Checksum validation")
    click.echo("   - Automatic data download")
    click.echo("   - Data file discovery")


def _generate_instructions(experiment_data: dict, output_path: Path, script_path: Path):
    """Generate reproduction instructions."""
    instructions_path = output_path / "REPRODUCTION_INSTRUCTIONS.md"
    
    experiment_id = experiment_data.get('id', 'Unknown')
    name = experiment_data.get('name', 'Unknown')
    
    environment = experiment_data.get('environment', {})
    git_info = environment.get('git', {})
    
    instructions = f"""# Experiment Reproduction Instructions

## Experiment Details
- **ID**: {experiment_id}
- **Name**: {name}
- **Status**: {experiment_data.get('status', 'unknown')}
- **Created**: {experiment_data.get('timestamp', 'Unknown')}

## Reproduction Steps

### 1. Environment Setup
- Ensure you have the correct Python version
- Install requirements: `pip install -r requirements.txt`

### 2. Code Setup
"""
    
    if git_info.get('commit'):
        instructions += f"""- Checkout the original Git commit: `git checkout {git_info['commit']}`
- Or ensure you're using the same codebase version
"""
    
    instructions += f"""
### 3. Run Experiment
Execute the reproduction script:
```bash
python {script_path.name}
```

### 4. Configuration
The original experiment used these parameters:
"""
    
    config = experiment_data.get('config', {})
    user_config = config.get('user_config', {}) if isinstance(config, dict) else {}
    for key, value in user_config.items():
        instructions += f"- **{key}**: `{value}`\n"
    
    instructions += f"""
### 5. Expected Results
Review the original experiment metrics in `experiment_metadata.json` to compare results.

### Notes
- This reproduction was generated automatically
- Environment differences may affect results
- Contact the original experimenter for questions
"""
    
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    click.echo(f"✅ Generated instructions: {instructions_path}") 