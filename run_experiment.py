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

experiments = []

common_args = [
    "--data-dir", "./data/cifar10",
    "--batch-size", "128",
    "--epochs", "100",
    "--workers", "4",
    "--lr", "1e-3",
]

test_configs = [
    # --- Baselines ---
    {
        "name": "SGD_base",
        "args": [
            "--classifier", "resnet18",
            "--optimizer", "sgd",
            "--lr", "0.1",
            "--momentum", "0.9",
            "--wd", "5e-4",
            "--save-as", "resnet18_SGD_base.pth"
        ] + common_args
    },
    {
        "name": "Adam_base",
        "args": [
            "--classifier", "resnet18",
            "--optimizer", "adam",
            "--lr", "1e-3",
            "--beta1", "0.9",
            "--beta2", "0.999",
            "--wd", "0",
            "--save-as", "resnet18_Adam_base.pth"
        ] + common_args
    },

    # --- SGD + weight noise ---
    {
        "name": "SGD_weight_noise_exp",
        "args": [
            "--classifier", "resnet18",
            "--optimizer", "sgd",
            "--lr", "0.1",
            "--momentum", "0.9",
            "--wd", "5e-4",
            "--noise-type", "weight",
            "--noise-magnitude", "0.05",
            "--noise-distribution", "gaussian",
            "--noise-schedule", "exp",
            "--save-as", "resnet18_SGD_weight_noise_exp.pth"
        ] + common_args
    },

    # --- SGD + gradient noise ---
    {
        "name": "SGD_grad_noise_linear",
        "args": [
            "--classifier", "resnet18",
            "--optimizer", "sgd",
            "--lr", "0.05",
            "--momentum", "0.9",
            "--wd", "5e-4",
            "--noise-type", "gradient",
            "--noise-magnitude", "0.01",
            "--noise-distribution", "gaussian",
            "--noise-schedule", "linear",
            "--save-as", "resnet18_SGD_grad_noise_linear.pth"
        ] + common_args
    },

    # --- Adam + gradient noise ---
    {
        "name": "Adam_grad_noise_exp",
        "args": [
            "--classifier", "resnet18",
            "--optimizer", "adam",
            "--lr", "1e-3",
            "--beta1", "0.9",
            "--beta2", "0.999",
            "--wd", "1e-4",
            "--noise-type", "gradient",
            "--noise-magnitude", "0.02",
            "--noise-distribution", "gaussian",
            "--noise-schedule", "exp",
            "--save-as", "resnet18_Adam_grad_noise_exp.pth"
        ] + common_args
    },

    # --- Adam + weight noise ---
    {
        "name": "Adam_weight_noise_linear",
        "args": [
            "--classifier", "resnet18",
            "--optimizer", "adam",
            "--lr", "5e-4",
            "--beta1", "0.9",
            "--beta2", "0.999",
            "--wd", "0",
            "--noise-type", "weight",
            "--noise-magnitude", "0.01",
            "--noise-distribution", "gaussian",
            "--noise-schedule", "linear",
            "--save-as", "resnet18_Adam_weight_noise_linear.pth"
        ] + common_args
    }
]


models = ['tiny_cnn']#['simple_cnn', 'resnet18']
noise_types = ['weight', 'gradient']
noise_layer_config = ["layer4.", "conv1", "bias"] # 1: layer4., 2: conv1, 3: bias
subset = [0.4, 1.0]
noise_magnitudes = [0.01, 0.05]
momentums = [0.9]
betas1 = [0, 0.9]
betas2 = [0.99, 0.999]
noise_schedules =[NoiseSchedule.exponential] #TODO CHECK avec l'experience en cours #[NoiseSchedule.constant, NoiseSchedule.linear, NoiseSchedule.cosine, NoiseSchedule.exponential]

for model in models:
    for momentum in momentums:
        for beta1 in betas1:
            for beta2 in betas2:
                for noise in noise_types:
                    for sub in subset:
                        args = [
                               "--classifier", model,
                               "--noise-type", "none",
                               "--momentum", str(momentum),
                               "--save-as", f"{model}_{sub}_baseline.pth",
                               "--subset", str(sub),
                               "--optimizer", 'adam',
                               "--beta1", str(beta1),
                               "--beta2", str(beta2),
                           ] + common_args
                        experiments.append({
                            "name": f"{model}_{sub}_baseline",
                            "args": args
                        })
                        for noise_magnitude in noise_magnitudes:
                            for noise_schedule in noise_schedules:
                                for noise_layer in noise_layer_config:
                                    args = [
                                        "--classifier", model,
                                        "--noise-type", noise,
                                        "--momentum", str(momentum),
                                        "--noise-magnitude", str(noise_magnitude),
                                        "--noise-schedule", noise_schedule,
                                        "--noise-layer", noise_layer,
                                        "--save-as", f"{model}_{sub}_{noise}_{noise_magnitude}_{noise_schedule}.pth",
                                        "--subset", str(sub),
                                        "--optimizer", 'adam',
                                        "--beta1", str(beta1),
                                        "--beta2", str(beta2),
                                    ] + common_args

                                    experiments.append({
                                        "name": f"{model}_{sub}_{noise}_{noise_magnitude}_{noise_schedule}",
                                        "args": args
                                    })

#TODO ATTENTION C'EST PAS LES TESTS DANS LES LOOP MAIS LA LISTE D'AU DESSUS
#experiments = test_configs

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