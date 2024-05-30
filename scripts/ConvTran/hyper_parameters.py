params = {'data_path': 'Dataset/UEA/', 'output_dir': 'Results', 'Norm': False, 'val_ratio': 0.2,
            'print_interval': 50, 'Net_Type': ['C-T'], 'emb_size': 16, 'dim_ff': 256, 'num_heads': 8,
            'Fix_pos_encode': 'tAPE', 'Rel_pos_encode': 'eRPE', 'epochs': 100, 'batch_size': 16,
            'lr': 0.001, 'dropout': 0.01, 'val_interval': 2, 'key_metric': 'fall_recall', 'gpu': 0,
            'console': False, 'seed': 0}