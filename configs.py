class Config:
    # Data parameters
    sampling_rate = 256
    epoch_length = 2  # seconds
    overlap = 0.5
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    valid_conditions = ['EC', 'EO', 'TASK']
    
    # Electrode coordinates (10-20 system)
    coord_map = {
        'Fp1': (-0.033, 0.090, 0.015), 'F3': (-0.045, 0.050, 0.060),
        'C3': (-0.063, 0.000, 0.075), 'P3': (-0.045, -0.050, 0.060),
        'O1': (-0.033, -0.090, 0.015), 'F7': (-0.075, 0.065, 0.030),
        'T3': (-0.095, 0.000, 0.000), 'T5': (-0.075, -0.065, 0.030),
        'Fz': (0.000, 0.057, 0.038), 'Fp2': (0.033, 0.090, 0.015),
        'F4': (0.045, 0.050, 0.060), 'C4': (0.063, 0.000, 0.075),
        'P4': (0.045, -0.050, 0.060), 'O2': (0.033, -0.090, 0.015),
        'F8': (0.075, 0.065, 0.030), 'T4': (0.095, 0.000, 0.000),
        'T6': (0.075, -0.065, 0.030), 'Cz': (0.000, 0.000, 0.095),
        'Pz': (0.000, -0.057, 0.038)
    }
    
    # Model parameters
    cheb_k = 2
    hidden_dim = 128
    condition_dim = 3  # EC/EO/TASK encoding
    
    # Training parameters
    pretrain_epochs = 5
    finetune_epochs = 5
    batch_size = 32
    lr = 1e-5
    mask_ratio = 0.15
    spatial_kappa = 0.9
    functional_tau = 3

config = Config()