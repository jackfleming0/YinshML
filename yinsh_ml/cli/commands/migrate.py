"""
CLI command for migrating historical experiment data.
"""

import logging
from pathlib import Path
from typing import Optional

import click

from yinsh_ml.tracking.migration_tool import MigrationTool

logger = logging.getLogger(__name__)


@click.command()
@click.argument('results_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option('--dry-run', is_flag=True, help='Preview what would be migrated without importing')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Set logging level')
@click.option('--report-file', type=click.Path(path_type=Path), 
              help='Save detailed migration report to file')
def migrate(results_dir: Path, dry_run: bool, log_level: str, report_file: Optional[Path]):
    """
    Migrate historical experiment data to the tracking system.
    
    Scans the specified results directory for experiment subdirectories
    and imports their data into the new experiment tracking system.
    
    Args:
        results_dir: Directory containing historical experiment results
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Starting migration from {results_dir}")
    
    if dry_run:
        click.echo("DRY RUN MODE - No data will be imported")
    
    try:
        # Initialize migration tool
        migration_tool = MigrationTool()
        
        if dry_run:
            # Just scan and report what would be migrated
            experiment_dirs = migration_tool.scan_experiment_directory(results_dir)
            
            click.echo(f"\nFound {len(experiment_dirs)} experiment directories:")
            for exp_dir in experiment_dirs:
                click.echo(f"  - {exp_dir.name}")
            
            click.echo(f"\nWould migrate {len(experiment_dirs)} experiments")
            return
        
        # Perform actual migration
        results = migration_tool.migrate_directory(results_dir)
        
        # Generate summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_metrics = sum(r.metrics_imported for r in results)
        
        # Display summary
        click.echo(f"\n{'='*50}")
        click.echo("MIGRATION SUMMARY")
        click.echo(f"{'='*50}")
        click.echo(f"Total experiments processed: {len(results)}")
        click.echo(f"Successful imports: {successful}")
        click.echo(f"Failed imports: {failed}")
        click.echo(f"Total metrics imported: {total_metrics}")
        
        if successful > 0:
            success_rate = (successful / len(results)) * 100
            click.echo(f"Success rate: {success_rate:.1f}%")
        
        # Show detailed results
        if results:
            click.echo(f"\nDETAILED RESULTS:")
            click.echo("-" * 30)
            
            for result in results:
                status = click.style("✓ SUCCESS", fg='green') if result.success else click.style("✗ FAILED", fg='red')
                click.echo(f"{status}: {result.experiment_name}")
                
                if result.experiment_id:
                    click.echo(f"  Experiment ID: {result.experiment_id}")
                
                if result.metrics_imported > 0:
                    click.echo(f"  Metrics imported: {result.metrics_imported}")
                
                if result.errors:
                    click.echo(click.style("  Errors:", fg='red'))
                    for error in result.errors:
                        click.echo(f"    - {error}")
                
                if result.warnings:
                    click.echo(click.style("  Warnings:", fg='yellow'))
                    for warning in result.warnings:
                        click.echo(f"    - {warning}")
                
                click.echo()
        
        # Save report if requested
        if report_file:
            try:
                report_content = _generate_detailed_report(results)
                report_file.parent.mkdir(parents=True, exist_ok=True)
                with open(report_file, 'w') as f:
                    f.write(report_content)
                click.echo(f"Detailed report saved to {report_file}")
            except Exception as e:
                click.echo(click.style(f"Failed to save report: {e}", fg='red'))
        
        # Return appropriate exit code
        if failed > 0:
            click.echo(click.style(f"\nMigration completed with {failed} failures", fg='yellow'))
            exit(1)
        else:
            click.echo(click.style("\nMigration completed successfully!", fg='green'))
            exit(0)
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        click.echo(click.style(f"Migration failed: {e}", fg='red'))
        exit(1)


def _generate_detailed_report(results) -> str:
    """Generate a detailed text report of migration results."""
    from datetime import datetime
    
    report_lines = [
        "=" * 60,
        "YINSHML EXPERIMENT DATA MIGRATION REPORT",
        "=" * 60,
        f"Generated: {datetime.now().isoformat()}",
        f"Total experiments: {len(results)}",
        ""
    ]
    
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_metrics = sum(r.metrics_imported for r in results)
    
    report_lines.extend([
        "SUMMARY",
        "-" * 20,
        f"Successful imports: {successful}",
        f"Failed imports: {failed}",
        f"Total metrics imported: {total_metrics}",
        ""
    ])
    
    if successful > 0:
        success_rate = (successful / len(results)) * 100
        report_lines.append(f"Success rate: {success_rate:.1f}%")
        report_lines.append("")
    
    report_lines.extend([
        "DETAILED RESULTS",
        "-" * 20
    ])
    
    for result in results:
        status = "SUCCESS" if result.success else "FAILED"
        report_lines.append(f"[{status}] {result.experiment_name}")
        report_lines.append(f"  Source: {result.source_path}")
        
        if result.experiment_id:
            report_lines.append(f"  Experiment ID: {result.experiment_id}")
        
        if result.metrics_imported > 0:
            report_lines.append(f"  Metrics imported: {result.metrics_imported}")
        
        if result.errors:
            report_lines.append("  Errors:")
            for error in result.errors:
                report_lines.append(f"    - {error}")
        
        if result.warnings:
            report_lines.append("  Warnings:")
            for warning in result.warnings:
                report_lines.append(f"    - {warning}")
        
        report_lines.append("")
    
    return "\n".join(report_lines) 