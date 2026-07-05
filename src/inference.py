import os
import torch
import numpy as np
from collections import Counter

_MODEL_CACHE = None
_VOCAB_CACHE = None
_STYLE_PRESETS = {}

def load_vocabulary(data_dir):
    """
    Builds the exact same vocabulary mapping as HandwritingDataset.build_vocab,
    but directly from sentences.txt (without loading strokes.npy).
    """
    sentences_path = os.path.join(data_dir, "sentences.txt")
    if not os.path.exists(sentences_path):
        raise FileNotFoundError(f"Vocabulary source file not found at: {sentences_path}")
        
    with open(sentences_path, "r", encoding="utf-8") as file:
        texts = file.read().splitlines()
        
    n_total = len(texts)
    char_lens = [len(t) for t in texts]
    max_char_len = max(char_lens)

    inp_text = np.ndarray((n_total, max_char_len), dtype='<U1')
    inp_text[:, :] = ' '
    for i, text_len in enumerate(char_lens):
        inp_text[i, :text_len] = list(texts[i])

    counter = Counter()
    for text in inp_text:
        counter.update(text)
    unique_char = sorted(counter)
    vocab_size = len(unique_char)

    id_to_char = dict(zip(np.arange(vocab_size), unique_char))
    char_to_id = dict([(v, k) for (k, v) in id_to_char.items()])
    return char_to_id, id_to_char

def get_style_preset(style_idx, data_dir, device):
    """
    Loads and caches a style preset stroke sequence and its matching text transcript.
    Replicates data_normalization for the selected stroke sequence.
    """
    global _STYLE_PRESETS
    if style_idx not in _STYLE_PRESETS:
        print(f"[Inference Engine] Loading style preset index {style_idx} from dataset...")
        strokes_path = os.path.join(data_dir, "strokes.npy")
        sentences_path = os.path.join(data_dir, "sentences.txt")
        
        if not os.path.exists(strokes_path) or not os.path.exists(sentences_path):
            raise FileNotFoundError("Missing dataset assets for loading style presets.")
            
        strokes = np.load(strokes_path, allow_pickle=True, encoding='bytes')
        with open(sentences_path, "r", encoding="utf-8") as f:
            texts = f.read().splitlines()
            
        if style_idx >= len(strokes):
            raise IndexError(f"Style index {style_idx} is out of bounds of the dataset (size: {len(strokes)}).")
            
        style_stroke = strokes[style_idx].astype(np.float32)
        real_text = texts[style_idx]
        
        # Perform local coordinate offset normalization (mean 0, std 1)
        style_norm = style_stroke.copy()
        mean = style_norm[:, 1:].mean(axis=0)
        style_norm[:, 1:] -= mean
        std = style_norm[:, 1:].std(axis=0)
        std = np.where(std == 0, 1.0, std)  # Prevent division by zero
        style_norm[:, 1:] /= std
        
        # Shape: [1, seq_len, 3]
        style_tensor = torch.from_numpy(style_norm).unsqueeze(0).to(device)
        
        _STYLE_PRESETS[style_idx] = {
            "style_tensor": style_tensor,
            "real_text": real_text
        }
        
    return _STYLE_PRESETS[style_idx]

def get_or_load_resources(model_path=None, data_dir=None):
    """
    Retrieves the cached model instance and vocabulary mapping.
    """
    global _MODEL_CACHE, _VOCAB_CACHE
    from src.config import MODEL_PATH, DATA_DIR, DEVICE
    from src.model import HandWritingSynthesisNet
    
    m_path = model_path or MODEL_PATH
    d_dir = data_dir or DATA_DIR
    
    if _VOCAB_CACHE is None:
        print(f"[Inference Engine] Compiling vocabulary from {d_dir}...")
        _VOCAB_CACHE = load_vocabulary(d_dir)
        
    char_to_id, id_to_char = _VOCAB_CACHE
    
    if _MODEL_CACHE is None:
        print(f"[Inference Engine] Loading model weights into memory ({DEVICE}) from {m_path}...")
        model = HandWritingSynthesisNet(window_size=len(char_to_id))
        model.load_state_dict(torch.load(m_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        _MODEL_CACHE = model
        
    return _MODEL_CACHE, char_to_id, id_to_char

def generate_handwriting(input_text, model_path=None, data_path=None, output_dir=None, animate=False, bias=10.0, batch_size=1, style_preset=None):
    """
    Processes input text, optionally primes the LSTM state using a reference handwriting sample,
    and runs the autoregressive strokes generation.
    """
    from src.config import MODEL_PATH, DATA_DIR, OUTPUT_DIR, TRAIN_MEAN, TRAIN_STD, DEVICE
    from src.dataset import data_denormalization
    from src.visualization import plot_stroke, animate_stroke_one_by_one, sanitize_filename
    
    m_path = model_path or MODEL_PATH
    d_path = data_path or DATA_DIR
    out_dir = output_dir or OUTPUT_DIR
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    model, char_to_id, id_to_char = get_or_load_resources(m_path, d_path)
    
    # Validate character vocabulary compatibility for the target text
    valid_chars = set(char_to_id.keys())
    input_chars = set(input_text)
    if not input_chars.issubset(valid_chars):
        invalid_chars = input_chars - valid_chars
        raise ValueError(
            f"Input contains invalid characters: {invalid_chars}. "
            f"Supported characters: {''.join(sorted(valid_chars))}"
        )
        
    # Setup priming variables
    prime = False
    prime_text = None
    prime_mask = None
    inp = torch.zeros(batch_size, 1, 3, device=DEVICE)
    
    if style_preset is not None:
        try:
            style_idx = int(style_preset)
            preset = get_style_preset(style_idx, d_path, DEVICE)
            prime = True
            real_text = preset["real_text"]
            prime_seq = preset["style_tensor"]
            
            print(f"[Inference Engine] Priming generation with style preset index {style_idx} ('{real_text}')")
            
            # Prepare priming text tokens
            real_seq = np.array(list(real_text))
            # Validate characters of the priming text
            real_chars = set(real_text)
            if not real_chars.issubset(valid_chars):
                # Fall back characters if index contains unsupported punctuation
                real_seq = np.array([c if c in valid_chars else ' ' for c in real_text])
                
            idx_arr = [char_to_id[char] for char in real_seq]
            prime_text = np.array([idx_arr for _ in range(batch_size)]).astype(np.float32)
            prime_text = torch.from_numpy(prime_text).to(DEVICE)
            prime_mask = torch.ones(prime_text.shape, device=DEVICE)
            
            # Input to the LSTM starts with the reference coordinates
            inp = prime_seq.repeat(batch_size, 1, 1)
        except Exception as e:
            print(f"[Inference Engine] Warning: Style priming failed ({e}). Reverting to default generation.")
            prime = False
            prime_text = None
            prime_mask = None
            inp = torch.zeros(batch_size, 1, 3, device=DEVICE)
            
    # Process target text sequence
    char_seq = np.array(list(input_text + "  "))  # Add trailing spacing padding
    text = np.array(
        [[char_to_id[char] for char in char_seq] for _ in range(batch_size)]
    ).astype(np.float32)
    
    text = torch.from_numpy(text).to(DEVICE)
    text_mask = torch.ones(text.shape, device=DEVICE)
    
    # Initialize hidden state structures
    hidden, window_vector, kappa = model.init_hidden(batch_size, DEVICE)
    
    # Execute model sequence generation
    gen_seq = model.generate(
        inp,
        text,
        text_mask,
        prime_text=prime_text,
        prime_mask=prime_mask,
        hidden=hidden,
        window_vector=window_vector,
        kappa=kappa,
        bias=bias,
        is_map=False,
        prime=prime,
    )
    
    # Denormalize stroke coordinate offsets back to original scale
    gen_seq = data_denormalization(TRAIN_MEAN, TRAIN_STD, gen_seq)
    
    # Generate visualization
    safe_text = sanitize_filename(input_text)
    static_output_path = os.path.join(out_dir, f"handwriting_{safe_text}.png")
    
    plot_stroke(gen_seq[0], save_name=static_output_path)
    
    if animate:
        anim_output_path = os.path.join(out_dir, f"handwriting_{safe_text}.gif")
        animate_stroke_one_by_one(gen_seq[0], save_name=anim_output_path)
        return anim_output_path
        
    return static_output_path
