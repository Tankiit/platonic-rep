#!/usr/bin/env python3
"""
Launch TensorBoard to view multi-model analysis logs
"""

import subprocess
import sys
from pathlib import Path

def launch_tensorboard(log_dir="./results/multi_model_analysis/tensorboard_logs"):
    """Launch TensorBoard with the specified log directory"""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        print(f"Log directory {log_path} does not exist.")
        print("Run some analysis first to generate logs.")
        return
    
    print(f"Launching TensorBoard with logs from: {log_path}")
    print("TensorBoard will be available at: http://localhost:6006")
    print("Press Ctrl+C to stop TensorBoard")
    
    try:
        # Launch TensorBoard
        subprocess.run([
            sys.executable, "-m", "tensorboard.main",
            "--logdir", str(log_path),
            "--port", "6006",
            "--host", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except Exception as e:
        print(f"Error launching TensorBoard: {e}")
        print("Make sure tensorboard is installed: pip install tensorboard")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch TensorBoard for multi-model analysis")
    parser.add_argument("--log_dir", type=str, 
                       default="./results/multi_model_analysis/tensorboard_logs",
                       help="Directory containing TensorBoard logs")
    
    args = parser.parse_args()
    launch_tensorboard(args.log_dir)
