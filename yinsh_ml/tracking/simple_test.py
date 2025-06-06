"""Simple test for database utilities."""

from utils import (
    initialize_database, create_experiment, add_metric_to_experiment, 
    get_database_stats, close_all_connections, query_experiments,
    get_experiment_by_id, add_metrics_bulk
)
import tempfile
import os

def main():
    print('Testing database utilities...')
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'test.db')
    
    try:
        # Test initialization
        db = initialize_database(test_db)
        print('✓ Database initialized')
        
        # Test experiment creation
        config = {'model': 'ResNet50', 'batch_size': 32}
        environment = {'python': '3.9', 'pytorch': '1.10'}
        exp_id = create_experiment(
            'test_exp', 'abc123', 'main', 
            config, environment, 
            tags=['test', 'demo']
        )
        print(f'✓ Created experiment {exp_id}')
        
        # Test single metric
        add_metric_to_experiment(exp_id, 'accuracy', 0.95, 1)
        print('✓ Added single metric')
        
        # Test bulk metrics
        metrics = [
            {'name': 'loss', 'value': 0.05, 'iteration': 1},
            {'name': 'accuracy', 'value': 0.97, 'iteration': 2}
        ]
        add_metrics_bulk(exp_id, metrics)
        print('✓ Added bulk metrics')
        
        # Test querying
        exp = get_experiment_by_id(exp_id)
        print(f'✓ Retrieved experiment: {exp["name"]}')
        
        exps = query_experiments(tags=['test'])
        print(f'✓ Found {len(exps)} experiments with tag "test"')
        
        # Test stats
        stats = get_database_stats()
        print(f'✓ Stats: {stats["experiment_count"]} experiments, {stats["metric_count"]} metrics')
        
        print('✅ All tests passed!')
        
    finally:
        # Cleanup
        close_all_connections()
        if os.path.exists(test_db):
            os.unlink(test_db)
        os.rmdir(temp_dir)

if __name__ == '__main__':
    main() 