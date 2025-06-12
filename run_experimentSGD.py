# run_experiments.py
import subprocess
import os
import json
from datetime import datetime

from sympy.physics.units import momentum

from core.noise_regularization import NoiseSchedule

# Create experiments directory for results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiments_dir = f"experiments_{timestamp}"
os.makedirs(experiments_dir, exist_ok=True)


def run_baseline(model, sub, opti, momentum, beta1, beta2):
    """
    Function to run a baseline experiment without noise.
    """

    # Add baseline experiment
    args = [
        "--classifier", model,
        "--noise-type", "none",
        "--save-as", f"{model}_{sub}_{opti}_baseline.pth",
        "--subset", str(sub),
        "--optimizer", opti,
        "--momentum", str(momentum),
        "--beta1", str(beta1),
        "--beta2", str(beta2),
        "--lr", ("1e-3" if opti == 'adam' else "1e-3"),
    ]

    return args


def run_noise_experiment(model, sub, opti, momentum, beta1, beta2,
                         noise, noise_distrib, noise_magnitude, noise_schedule, noise_layer,
                         permanent, noise_during_stuck_only, patience):
    """
    Function to run an experiment with noise.
    """

    # Add noise experiments
    args = [
        "--classifier", model,
        "--noise-type", str(noise),
        "--noise-distribution", noise_distrib,  # Assuming normal distribution for noise
        "--noise-magnitude", str(noise_magnitude),
        "--noise-schedule", noise_schedule,
        "--save-as", f"{model}_{sub}_{opti}_{noise}_{noise_magnitude}_{noise_schedule}.pth",
        "--subset", str(sub),
        "--optimizer", opti,
        "--momentum", str(momentum),
        "--beta1", str(beta1),
        "--beta2", str(beta2),
        "--lr", ("1e-3" if opti == 'adam' else "1e-3"),
        "--patience", str(patience),
    ]

    if permanent:
        args.append("--permanent")
    if noise_during_stuck_only:
        args.append("--noise-during-stuck-only")
    if noise_layer is not None:
        args.append("--noise-layer")
        args.append(noise_layer)

    return args

experiments = []

# List of parameter combinations for experiments
models = ['baby_cnn', 'tiny_cnn', 'resnet18', 'simple_cnn']
subset = [1.0]

# Optimizer parameters
optimizers = ['sgd']
momentums = [0]

# Common arguments for all runs
common_args = [
    "--data-dir", "./data/cifar10",
    "--batch-size", "256",
    "--epochs", "300",
    "--workers", "6",
    "--wd", "1e-4",
    "--flag-min-epochs", "70",
    "--flag-window-size", "10",
    "--flag-plateau-delta", "1e-4",
    "--flag-overfit-epochs", "3",
    "--flag-grad-plateau-thr", "0.025",
    "--flag-low-weight-update-thr", "5e-4",


    "--relative-min-noise","0.01", # 1% relative noise
    "--relative-max-noise","0.05", # 10% relative noise
    "--lr-restart-period","50",    # Restart LR every 50 epochs
    "--consecutive-flag-trigger","10", # Start cooldown after 3 noisy epochs
    "--min-cooldown-epochs","3",      # Cooldown for at least 5 epochs
    "--max-cooldown-epochs","10",
]

baseline_common_args = [
    "--data-dir", "./data/cifar10",
    "--batch-size", "256",
    "--epochs", "300",
    "--workers", "6",
    "--wd", "1e-4",
    "--disable-adaptive-flags"
]


# # --- SCENARIO 1: Baseline (No Noise Ever) ---

print("Configuring Baseline Experiment...")
for model in models:
    for sub in subset:
        for opti in optimizers:
            if opti == 'sgd':
                for momentum_val in momentums:
                    args = run_baseline(model, sub, opti, momentum_val, 0, 0) + baseline_common_args
                    experiments.append({
                        "name": f"{model}_{sub}_baseline_NoNoise",
                        "args": args
                    })
            if opti == 'adam':
                for momentum_val in momentums:
                    args = run_baseline(model, sub, opti, momentum_val, 0.9, 0.999) + baseline_common_args
                    experiments.append({
                        "name": f"{model}_{sub}_baseline_NoNoise",
                        "args": args
                    })

# --- SCENARIO 2: Adaptive Noise Injection (Triggered by Flags) ---
print("Configuring Adaptive Noise Experiment...")
# Parameters for this scenario:

adaptive_noise_types = ['none']  # Choose the specific noise type for adaptive
adaptive_noise_magnitudes = [0.01]  # The initial magnitude for the adaptive noise
adaptive_noise_schedules = [NoiseSchedule.cosine]  # The schedule for the adaptive noise
adaptive_noise_dist = ['gaussian', 'uniform']
adaptive_permanent = [True]
adaptive_stuck_only = [True]  # KEY: True for adaptive
patience = 10

for model in models:
    for sub in subset:
        for opti in optimizers:
            if opti == 'sgd':
                for momentum_val in momentums:
                    for noise_type_val in adaptive_noise_types:
                        for noise_mag_val in adaptive_noise_magnitudes:
                            for noise_dist_val in adaptive_noise_dist:
                                for noise_sched_val in adaptive_noise_schedules:
                                    for perm_val in adaptive_permanent:
                                        for stuck_only_val in adaptive_stuck_only:
                                            args = (run_noise_experiment(model, sub, opti, momentum_val, 0, 0,
                                                                         noise_type_val, noise_dist_val, noise_mag_val,
                                                                         noise_sched_val, None,
                                                                         perm_val, stuck_only_val, patience,
                                                                         )
                                                    + common_args)
                                            experiments.append({
                                                "name": f"{model}_{sub}_Adaptive_{noise_type_val}_Mag{noise_mag_val}_Sched{noise_sched_val}",
                                                "args": args
                                            })
            if opti == 'adam':
                for noise_type_val in adaptive_noise_types:
                    for noise_mag_val in adaptive_noise_magnitudes:
                        for noise_dist_val in adaptive_noise_dist:
                            for noise_sched_val in adaptive_noise_schedules:
                                for perm_val in adaptive_permanent:
                                    for stuck_only_val in adaptive_stuck_only:
                                        args = (run_noise_experiment(model, sub, opti, momentum_val, 0.9, 0.999,
                                                                     noise_type_val, noise_dist_val, noise_mag_val,
                                                                     noise_sched_val, None,
                                                                     perm_val, stuck_only_val, patience,
                                                                     )
                                                + common_args)
                                        experiments.append({
                                            "name": f"{model}_{sub}_Adaptive_{noise_type_val}_Mag{noise_mag_val}_Sched{noise_sched_val}",
                                            "args": args
                                        })

# Create experiment log file
log_file = os.path.join(experiments_dir, "experiments_log.json")
with open(log_file, 'w') as f:
    json.dump(experiments, f, indent=2)

# Run each experiment
for i, experiment in enumerate(experiments):
    if isinstance(experiment, str):
        # Handle the case where experiment is a string
        name = f"experiment_{i + 1}"
        args = [experiment]  # Wrap the string in a list
    else:
        # Normal case where experiment is a dictionary
        name = experiment["name"]
        args = experiment["args"]

    print(f"\n{'=' * 80}")
    print(f"Starting experiment {i + 1}/{len(experiments)}: {name}")
    print(f"{'=' * 80}\n")

    # Log start time
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Build command
    cmd = ["python", "train.py"] + args
    print(f"Running command: {' '.join(cmd)}")

    # Run the training process
    try:
        process = subprocess.run(cmd, check=True)
        status = "Completed successfully"
    except subprocess.CalledProcessError as e:
        status = f"Failed with error code {e.returncode}"
        print(f"ERROR: {status}")

    # Log end time and duration
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nEnded at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Status: {status}")

    print(f"\n{'=' * 80}")
    print(f"Completed experiment {i + 1}/{len(experiments)}: {name}")
    print(f"{'=' * 80}\n")

print(f"\nAll experiments completed. Results saved in {experiments_dir}/")
