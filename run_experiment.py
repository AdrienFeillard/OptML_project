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

# Define experiment configurations
# experiments = [
#     # Experiment 1: No noise baseline
#     {
#         "name": "baseline",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "none",
#             "--save-as", f"resnet18_baseline.pth"
#         ] + common_args
#     },
#
#     # Gradient Noise Experiments
#     {
#         "name": "gradient_noise_gaussian_low",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "gradient",
#             "--noise-magnitude", "0.01",
#             "--noise-distribution", "gaussian",
#             "--noise-schedule", "constant",
#             "--save-as", f"resnet18_grad_gauss_low.pth"
#         ] + common_args
#     },
#     {
#         "name": "gradient_noise_gaussian_medium",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "gradient",
#             "--noise-magnitude", "0.05",
#             "--noise-distribution", "gaussian",
#             "--noise-schedule", "constant",
#             "--save-as", f"resnet18_grad_gauss_med.pth"
#         ] + common_args
#     },
#     {
#         "name": "gradient_noise_uniform",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "gradient",
#             "--noise-magnitude", "0.03",
#             "--noise-distribution", "uniform",
#             "--noise-schedule", "constant",
#             "--save-as", f"resnet18_grad_uniform.pth"
#         ] + common_args
#     },
#
#     # Weight Noise Experiments
#     {
#         "name": "weight_noise_gaussian_low",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "weight",
#             "--noise-magnitude", "0.01",
#             "--noise-distribution", "gaussian",
#             "--noise-schedule", "constant",
#             "--save-as", f"resnet18_weight_gauss_low.pth"
#         ] + common_args
#     },
#     {
#         "name": "weight_noise_gaussian_medium",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "weight",
#             "--noise-magnitude", "0.03",
#             "--noise-distribution", "gaussian",
#             "--noise-schedule", "constant",
#             "--save-as", f"resnet18_weight_gauss_med.pth"
#         ] + common_args
#     },
#     {
#         "name": "weight_noise_uniform",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "weight",
#             "--noise-magnitude", "0.02",
#             "--noise-distribution", "uniform",
#             "--noise-schedule", "constant",
#             "--save-as", f"resnet18_weight_uniform.pth"
#         ] + common_args
#     },
#
#     # Input Noise Experiments
#     {
#         "name": "input_noise_gaussian_low",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "input",
#             "--noise-magnitude", "0.01",
#             "--noise-distribution", "gaussian",
#             "--noise-schedule", "constant",
#             "--save-as", f"resnet18_input_gauss_low.pth"
#         ] + common_args
#     },
#     {
#         "name": "input_noise_gaussian_medium",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "input",
#             "--noise-magnitude", "0.05",
#             "--noise-distribution", "gaussian",
#             "--noise-schedule", "constant",
#             "--save-as", f"resnet18_input_gauss_med.pth"
#         ] + common_args
#     },
#     {
#         "name": "input_noise_uniform",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "input",
#             "--noise-magnitude", "0.03",
#             "--noise-distribution", "uniform",
#             "--noise-schedule", "constant",
#             "--save-as", f"resnet18_input_uniform.pth"
#         ] + common_args
#     },
#
#     # Label Noise Experiments
#     {
#         "name": "label_noise_low",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "label",
#             "--noise-magnitude", "0.01",  # 1% chance of label flip
#             "--noise-schedule", "constant",
#             "--save-as", f"resnet18_label_low.pth"
#         ] + common_args
#     },
#     {
#         "name": "label_noise_medium",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "label",
#             "--noise-magnitude", "0.05",  # 5% chance of label flip
#             "--noise-schedule", "constant",
#             "--save-as", f"resnet18_label_med.pth"
#         ] + common_args
#     },
#
#     # Dropout Experiments
#     {
#         "name": "dropout_low",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "dropout",
#             "--dropout-prob", "0.1",  # 10% dropout rate
#             "--save-as", f"resnet18_dropout_low.pth"
#         ] + common_args
#     },
#     {
#         "name": "dropout_medium",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "dropout",
#             "--dropout-prob", "0.3",  # 30% dropout rate
#             "--save-as", f"resnet18_dropout_med.pth"
#         ] + common_args
#     },
#
#     # Different Noise Schedules (using gradient noise as an example)
#     {
#         "name": "gradient_noise_linear_decay",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "gradient",
#             "--noise-magnitude", "0.05",
#             "--noise-distribution", "gaussian",
#             "--noise-schedule", "linear",
#             "--save-as", f"resnet18_grad_linear.pth"
#         ] + common_args
#     },
#     {
#         "name": "gradient_noise_cosine_decay",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "gradient",
#             "--noise-magnitude", "0.05",
#             "--noise-distribution", "gaussian",
#             "--noise-schedule", "cosine",
#             "--save-as", f"resnet18_grad_cosine.pth"
#         ] + common_args
#     },
#     {
#         "name": "gradient_noise_exponential_decay",
#         "args": [
#             "--classifier", "resnet18",
#             "--noise-type", "gradient",
#             "--noise-magnitude", "0.05",
#             "--noise-distribution", "gaussian",
#             "--noise-schedule", "exponential",
#             "--save-as", f"resnet18_grad_exp.pth"
#         ] + common_args
#     }
#   ]

experiments = []

common_args = [
    "--data-dir", "./data/cifar10",
    "--batch-size", "128",
    "--epochs", "100",
    "--workers", "4",
    "--lr", "1e-2",
    "--wd", "1e-3",
]

models = ['resnet18']#['simple_cnn', 'resnet18']
noise_types = ['weight', 'gradient']
noise_layer_config = ["layer4.", "conv1", "bias"] # 1: layer4., 2: conv1, 3: bias
subset = [0.4, 1.0]
noise_magnitudes = [0.01]#, 0.05]
momentum = [0, 0.9]
noise_schedules = #TODO CHECK avec l'experience en cours #[NoiseSchedule.constant, NoiseSchedule.linear, NoiseSchedule.cosine, NoiseSchedule.exponential]

for model in models:
    for momentum in momentum:
        for noise in noise_types:
            for sub in subset:
                args = [
                       "--classifier", model,
                       "--noise-type", "none",
                       "--momentum", str(momentum),
                       "--save-as", f"{model}_{sub}_baseline.pth",
                       "--subset", str(sub)
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
                                "--subset", str(sub)
                            ] + common_args

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