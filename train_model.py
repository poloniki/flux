#!/usr/bin/env python3

import os
import argparse
import toml
import subprocess
import sys
from pathlib import Path
from slugify import slugify

# Add parent directory to path to import captions module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from library import captions

def resize_image(image_path, output_path, size):
    """Resize an image while maintaining aspect ratio."""
    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size/width) * height)
        else:
            new_height = size
            new_width = int((size/height) * width)
        print(f"resize {image_path} : {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def create_dataset(images_folder, destination_folder, size=1024, auto_caption=True):
    """Create a dataset with images and captions."""
    print(f"Creating dataset from {images_folder} to {destination_folder}")
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get all image files in the folder
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(images_folder).glob(f'*{ext}')))
    
    for image_path in image_files:
        # Copy the image to the destination folder
        from shutil import copy
        new_image_path = copy(str(image_path), destination_folder)

        # Resize the image
        resize_image(new_image_path, new_image_path, size)

        # Check for corresponding caption file
        caption_file_name = image_path.stem + ".txt"
        source_caption_path = image_path.parent / caption_file_name
        dest_caption_path = Path(destination_folder) / caption_file_name
        
        # Create or copy caption file
        if source_caption_path.exists():
            print(f"Using existing caption from {source_caption_path}")
            with open(source_caption_path, 'r') as src, open(dest_caption_path, 'w') as dst:
                dst.write(src.read())
        else:
            if auto_caption:
                print(f"Generating caption for {image_path.name}")
                try:
                    # Generate caption using the captions.py module
                    caption_text = captions.generate_caption(Path(image_path))
                    with open(dest_caption_path, 'w') as file:
                        file.write(caption_text)
                    print(f"Created caption: {caption_text[:50]}...")
                except Exception as e:
                    print(f"Error generating caption for {image_path.name}: {str(e)}")
            else:
                print(f"Warning: No caption file found for {image_path.name}")

    print(f"Dataset created at {destination_folder}")
    return destination_folder

def generate_toml(dataset_folder, resolution, class_tokens="", num_repeats=10):
    """Generate TOML configuration for the dataset."""
    config = {
        "general": {
            "shuffle_caption": False,
            "caption_extension": ".txt",
            "keep_tokens": 1
        },
        "datasets": [
            {
                "resolution": resolution,
                "batch_size": 1,
                "keep_tokens": 1,
                "subsets": [
                    {
                        "image_dir": str(dataset_folder),
                        "class_tokens": class_tokens,
                        "num_repeats": num_repeats
                    }
                ]
            }
        ]
    }
    
    return toml.dumps(config)

def create_sample_prompts(trigger_word, num_samples=4):
    """Create sample prompts for training visualization."""
    base_prompts = [
        "a professional photograph",
        "a detailed illustration",
        "a beautiful painting",
        "a stylized rendering"
    ]
    
    return "\n".join([f"{trigger_word}, {prompt}" for prompt in base_prompts[:num_samples]])

def find_flux_train_script():
    """Find the flux_train_network.py script in various possible locations."""
    possible_paths = [
        # Direct paths
        "flux_train_network.py",
        "./flux_train_network.py",
        "../flux_train_network.py",
        
        # Absolute paths
        "/app/flux_train_network.py",
        "/app/sd-scripts/flux_train_network.py",
        
        # Paths relative to current directory
        os.path.join(os.getcwd(), "flux_train_network.py"),
        os.path.join(os.getcwd(), "sd-scripts", "flux_train_network.py"),
        
        # Common locations within the repository
        "/sd-scripts/flux_train_network.py"
    ]
    
    # Find the script using find command for deeper search
    try:
        find_result = subprocess.run(
            ["find", "/", "-name", "flux_train_network.py", "-type", "f"],
            capture_output=True, 
            text=True,
            timeout=30  # Limit search time
        )
        if find_result.returncode == 0 and find_result.stdout:
            additional_paths = find_result.stdout.strip().split('\n')
            possible_paths.extend(additional_paths)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Unable to perform deep file search.")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = [p for p in possible_paths if not (p in seen or seen.add(p))]
    
    # Check each path
    for path in unique_paths:
        if os.path.isfile(path):
            print(f"Found flux_train_network.py at: {path}")
            return path
    
    # If we're here, we couldn't find the script
    print("Error: flux_train_network.py not found. Searched in:")
    for path in unique_paths:
        print(f"  - {path}")
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Train a Flux model with LoRA")
    parser.add_argument("--images_folder", required=True, help="Path to folder containing images and captions")
    parser.add_argument("--model_name", required=True, help="Name for the trained model")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution for images")
    parser.add_argument("--learning_rate", type=str, default="5e-4", help="Learning rate")
    parser.add_argument("--network_dim", type=int, default=8, help="LoRA network dimension")
    parser.add_argument("--max_train_epochs", type=int, default=20, help="Maximum training epochs")
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="Save every N epochs")
    parser.add_argument("--trigger_word", type=str, default="", help="Trigger word for the model")
    parser.add_argument("--num_repeats", type=int, default=20, help="Number of repeats per image")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=2, help="Number of workers")
    parser.add_argument("--no-auto-caption", action="store_true", help="Disable automatic caption generation")
    parser.add_argument("--flux_path", type=str, help="Path to flux_train_network.py if known")
    
    args = parser.parse_args()
    
    # Check OpenAI API key for captioning
    if not args.no_auto_caption and not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set. Auto-captioning will not work.")
        print("Set your API key with: export OPENAI_API_KEY='your-api-key'")
        print("Or disable auto-captioning with --no-auto-caption")
        return 1
    
    # Create output folders
    output_name = slugify(args.model_name)
    output_dir = Path(f"outputs/{output_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset folder
    dataset_dir = Path(f"datasets/{output_name}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images and create dataset
    create_dataset(
        args.images_folder, 
        dataset_dir, 
        args.resolution, 
        auto_caption=not args.no_auto_caption
    )
    
    # Generate TOML configuration
    toml_config = generate_toml(
        dataset_dir, 
        args.resolution, 
        args.trigger_word, 
        args.num_repeats
    )
    
    # Save TOML configuration
    dataset_config_path = output_dir / "dataset.toml"
    with open(dataset_config_path, 'w') as f:
        f.write(toml_config)
    print(f"Generated dataset configuration at {dataset_config_path}")
    
    # Create sample prompts
    sample_prompts = create_sample_prompts(args.trigger_word)
    sample_prompts_path = output_dir / "sample_prompts.txt"
    with open(sample_prompts_path, 'w') as f:
        f.write(sample_prompts)
    print(f"Generated sample prompts at {sample_prompts_path}")
    
    # Get the flux_train_network.py path
    flux_train_path = args.flux_path if args.flux_path else find_flux_train_script()
    
    if not flux_train_path:
        print("Could not find flux_train_network.py. Please provide it with --flux_path")
        print("You can also try running: find / -name flux_train_network.py 2>/dev/null")
        return 1
        
    # Construct training command
    cmd = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_cpu_threads_per_process", "1",
        flux_train_path,
        "--pretrained_model_name_or_path", "models/unet/flux1-dev.sft",
        "--clip_l", "models/clip/clip_l.safetensors",
        "--t5xxl", "models/clip/t5xxl_fp16.safetensors",
        "--ae", "models/vae/ae.sft",
        "--cache_latents_to_disk",
        "--save_model_as", "safetensors",
        "--sdpa", "--persistent_data_loader_workers",
        "--max_data_loader_n_workers", str(args.workers),
        "--seed", str(args.seed),
        "--gradient_checkpointing",
        "--mixed_precision", "bf16",
        "--save_precision", "bf16",
        "--network_module", "networks.lora_flux",
        "--network_dim", str(args.network_dim),
        "--optimizer_type", "adamw8bit",
        "--sample_prompts", str(sample_prompts_path),
        "--sample_every_n_steps", "200",
        "--learning_rate", args.learning_rate,
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--fp8_base",
        "--highvram",
        "--max_train_epochs", str(args.max_train_epochs),
        "--save_every_n_epochs", str(args.save_every_n_epochs),
        "--dataset_config", str(dataset_config_path),
        "--output_dir", str(output_dir),
        "--output_name", output_name,
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", "3.1582",
        "--model_prediction_type", "raw",
        "--guidance_scale", "1",
        "--loss_type", "l2",
        "--enable_bucket",
        "--flip_aug",
        "--lr_scheduler_args", "T_max=10000",
        "--lr_scheduler_min_lr_ratio", "0.1",
        "--lr_scheduler_num_cycles", "1",
        "--lr_scheduler_type", "CosineAnnealingLR",
        "--min_snr_gamma", "5",
        "--multires_noise_discount", "0.3",
        "--multires_noise_iterations", "6",
        "--noise_offset", "0.1",
        "--text_encoder_lr", "5e-5",
        "--train_batch_size", str(args.batch_size)
    ]
    
    # Log the command
    print("\nRunning training command:")
    print(" ".join(cmd))
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
        print(f"\nTraining completed successfully! Model saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 