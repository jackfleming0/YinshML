#!/usr/bin/env python3
try:
    from utils import initialize_database, create_experiment, close_all_connections
    print('✓ Import works')
    
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, 'quick_test.db')
    
    db = initialize_database(test_db)
    print('✓ Database initialized')
    
    exp_id = create_experiment('test', 'commit', 'branch', {}, {})
    print(f'✓ Created experiment: {exp_id}')
    
    close_all_connections()
    os.unlink(test_db)
    os.rmdir(temp_dir)
    print('✅ SUCCESS: Basic functionality works')
    
except Exception as e:
    print(f'❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
    exit(1) 