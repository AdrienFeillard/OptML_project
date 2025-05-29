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
        "--noise-type", noise,
        "--noise-distribution", noise_distrib,  # Assuming normal distribution for noise
        "--noise-magnitude", str(noise_magnitude),
        "--noise-schedule", noise_schedule,
        "--save-as", f"{model}_{sub}_{opti}_{noise}_{noise_magnitude}_{noise_schedule}.pth",
        "--subset", str(sub),
        "--optimizer", opti,
        "--momentum", str(momentum),
        "--beta1", str(beta1),
        "--beta2", str(beta2),
        "--lr", ("1e-3" if opti == 'adam' else "1e-2"),
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

common_args = [
    "--data-dir", "./data/cifar10",
    "--batch-size", "128",
    "--epochs", "200",
    "--workers", "4",
    "--wd", "1e-4",
]

# List of parameter combinations for experiments
models = ['baby_cnn']#['baby_cnn', 'tiny_cnn', 'simple_cnn', 'resnet18']
subset = [1.0]

# Optimizer parameters
optimizers = ['sgd', 'adam']
momentums = [0, 0.9]
betas1 = [0, 0.9]
betas2 = [0.999]

# Noise type parameters
noise_types = ['weight', 'gradient']
noise_layer_config = [None] #["layer4.", "conv1", "bias"]
noise_magnitudes = [0.01, 0.05]
noise_distribution = ['normal'] #['normal', 'uniform']
noise_schedules = [NoiseSchedule.cosine, NoiseSchedule.exponential]

# Noise application parameters

permanent = [False] # TODO Test
noise_during_stuck_only = [True, False]
patience = 5


for model in models:
    for sub in subset:
        for opti in optimizers:

            # Run experiments for sgd
            if opti == 'sgd':
                for momentum in momentums:

                    # Add baseline experiment
                    args = run_baseline(model, sub, opti, momentum, 0, 0) + common_args
                    experiments.append({
                        "name": f"{model}_{sub}_baseline",
                        "args": args
                    })

                    for noise in noise_types:
                        for noise_distrib in noise_distribution:
                            for noise_magnitude in noise_magnitudes:
                                for noise_schedule in noise_schedules:
                                    for noise_layer in noise_layer_config:
                                        # permanent is only for weight noise
                                        for perm in permanent:
                                            for stuck_only in noise_during_stuck_only:
                                                if noise!="weight":
                                                    break
                                                # Add noise experiments
                                                args = (run_noise_experiment(model, sub, opti, momentum, 0, 0,
                                                                            noise, noise_distrib, noise_magnitude, noise_schedule, noise_layer,
                                                                             perm, stuck_only, patience)
                                                                            + common_args)

                                                experiments.append({
                                                    "name": f"{model}_{sub}_{noise}_{noise_magnitude}_{noise_schedule}",
                                                    "args": args
                                                })


            # run experiments  for adam
            if opti == 'adam':
                for beta1 in betas1:
                    for beta2 in betas2:
                        # Add baseline experiment
                        args = run_baseline(model, sub, opti, 0, beta1, beta2) + common_args
                        experiments.append({
                            "name": f"{model}_{sub}_baseline",
                            "args": args
                        })

                        for noise in noise_types:
                            for noise_distrib in noise_distribution:
                                for noise_magnitude in noise_magnitudes:
                                    for noise_schedule in noise_schedules:
                                        for noise_layer in noise_layer_config:
                                            for perm in permanent:
                                                for stuck_only in noise_during_stuck_only:
                                                    if noise != "weight":
                                                        break
                                                    # Add noise experiments
                                                    args = (run_noise_experiment(model, sub, opti, 0, beta1, beta2,
                                                                                noise, noise_distrib, noise_magnitude, noise_schedule, noise_layer,
                                                                                 perm, stuck_only, patience)
                                                                                + common_args)

                                                    experiments.append({
                                                        "name": f"{model}_{sub}_{noise}_{noise_magnitude}_{noise_schedule}",
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
        name = f"experiment_{i+1}"
        args = [experiment]  # Wrap the string in a list
    else:
        # Normal case where experiment is a dictionary
        name = experiment["name"]
        args = experiment["args"]

    print(f"\n{'='*80}")
    print(f"Starting experiment {i+1}/{len(experiments)}: {name}")
    print(f"{'='*80}\n")

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

    print(f"\n{'='*80}")
    print(f"Completed experiment {i+1}/{len(experiments)}: {name}")
    print(f"{'='*80}\n")

print(f"\nAll experiments completed. Results saved in {experiments_dir}/")