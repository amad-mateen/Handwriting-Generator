"""
Main entry point for Handwriting Synthesis.
Provides full backward compatibility by re-exporting original classes and functions from the src package,
and implements a clean CLI interface for local generation testing with experimental style priming support.
"""

import os
import sys

# Add current workspace to system path to ensure submodules can be resolved
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Re-export classes and functions from the src module to preserve backward compatibility
from src.config import MODEL_PATH, DATA_DIR, OUTPUT_DIR, VOCAB_CHARS
from src.dataset import (
    HandwritingDataset,
    Global,
    train_offset_normalization,
    valid_offset_normalization,
    data_denormalization,
)
from src.model import HandWritingSynthesisNet, sample_from_out_dist, stable_softmax
from src.visualization import plot_stroke, animate_stroke_one_by_one, sanitize_filename
from src.inference import generate_handwriting, get_or_load_resources

if __name__ == "__main__":
    print("=" * 60)
    print("Handwriting Synthesis - Command Line Interface Utility")
    print("=" * 60)
    
    # Leverage local configuration folders
    local_data_dir = os.path.abspath("./data")
    local_model_path = os.path.abspath("./model/handwriting_synthesis_model.pt")
    local_output_dir = os.path.abspath("./outputs")
    
    try:
        input_text = input("Enter text to generate handwriting for: ").strip()
        if not input_text:
            print("[CLI Error] Text input cannot be empty.")
            sys.exit(1)
            
        # Perform validation on characters
        valid_chars = set(VOCAB_CHARS)
        input_chars = set(input_text)
        if not input_chars.issubset(valid_chars):
            invalid = input_chars - valid_chars
            print(f"[CLI Error] Text contains invalid characters: {invalid}")
            print(f"Supported vocabulary is: {VOCAB_CHARS}")
            sys.exit(1)
            
        animate_input = input("Generate animated GIF? (y/n) [default: n]: ").strip().lower()
        animate_gif = animate_input == 'y'
        
        bias_input = input("Enter sampling bias (float) [default: 10.0]: ").strip()
        sampling_bias = 10.0
        if bias_input:
            try:
                sampling_bias = float(bias_input)
            except ValueError:
                print("[CLI Warning] Invalid bias number. Falling back to default (10.0)")

        style_input = input("Enter style preset index (Experimental - e.g. 0=slanted, 3=bold, 5=neat, 12=loose) [default: none]: ").strip()
        style_preset = None
        if style_input:
            try:
                style_preset = int(style_input)
            except ValueError:
                print("[CLI Warning] Invalid style preset index. Generating without priming.")
                style_preset = None

        print("\n[CLI Engine] Generating handwriting representation...")
        result_path = generate_handwriting(
            input_text=input_text,
            model_path=local_model_path,
            data_path=local_data_dir,
            output_dir=local_output_dir,
            animate=animate_gif,
            bias=sampling_bias,
            style_preset=style_preset
        )
        print(f"\n[CLI Success] Output saved to: {result_path}")
        
    except KeyboardInterrupt:
        print("\n[CLI Info] Execution interrupted by user.")
    except Exception as err:
        print(f"\n[CLI Error] An unexpected error occurred: {err}")
        sys.exit(1)