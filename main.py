import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from torch.utils.data import Dataset
import re  # For sanitizing filenames

# --- Dataset Class ---
class HandwritingDataset(Dataset):
    def __init__(self, data_path, split='train', text_req=False, debug=False, max_seq_len=300, data_aug=False):
        self.text_req = text_req
        self.max_seq_len = max_seq_len
        self.data_aug = data_aug

        strokes = np.load(os.path.join(data_path, 'strokes.npy'), allow_pickle=True, encoding='bytes')
        with open(os.path.join(data_path, 'sentences.txt')) as file:
            texts = file.read().splitlines()

        lengths = [len(stroke) for stroke in strokes]
        max_len = np.max(lengths)
        n_total = len(strokes)

        mask_shape = (n_total, max_len)
        mask = np.zeros(mask_shape, dtype=np.float32)

        char_seqs = [list(char_seq) for char_seq in texts]
        char_seqs = np.array(char_seqs, dtype=object)

        char_lens = [len(char_seq) for char_seq in char_seqs]
        max_char_len = np.max(char_lens)

        mask_shape = (n_total, max_char_len)
        char_mask = np.zeros(mask_shape, dtype=np.float32)

        inp_text = np.ndarray((n_total, max_char_len), dtype='<U1')
        inp_text[:, :] = ' '

        data_shape = (n_total, max_len, 3)
        data = np.zeros(data_shape, dtype=np.float32)

        for i, (seq_len, text_len) in enumerate(zip(lengths, char_lens)):
            mask[i, :seq_len] = 1.
            data[i, :seq_len] = strokes[i]
            char_mask[i, :text_len] = 1.
            inp_text[i, :text_len] = char_seqs[i]

        self.id_to_char, self.char_to_id = self.build_vocab(inp_text)
        self.vocab_size = len(self.id_to_char)

        idx_permute = np.random.permutation(n_total)
        data = data[idx_permute]
        mask = mask[idx_permute]
        inp_text = inp_text[idx_permute]
        char_mask = char_mask[idx_permute]

        if debug:
            data = data[:64]
            mask = mask[:64]
            inp_text = inp_text[:64]
            char_mask = char_mask[:64]

        n_train = int(0.9 * data.shape[0])
        self._data = data
        if split == 'train':
            self.dataset = data[:n_train]
            self.mask = mask[:n_train]
            self.texts = inp_text[:n_train]
            self.char_mask = char_mask[:n_train]
            Global.train_mean, Global.train_std, self.dataset = train_offset_normalization(
                self.dataset)
        elif split == 'valid':
            self.dataset = data[n_train:]
            self.mask = mask[n_train:]
            self.texts = inp_text[n_train:]
            self.char_mask = char_mask[n_train:]
            self.dataset = valid_offset_normalization(
                Global.train_mean, Global.train_std, self.dataset)

    def __len__(self):
        return self.dataset.shape[0]

    def idx_to_char(self, id_seq):
        return np.array([self.id_to_char[id] for id in id_seq])

    def char_to_idx(self, char_seq):
        return np.array([self.char_to_id[char] for char in char_seq]).astype(np.float32)

    def build_vocab(self, texts):
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(text)
        unique_char = sorted(counter)
        vocab_size = len(unique_char)

        id_to_char = dict(zip(np.arange(vocab_size), unique_char))
        char_to_id = dict([(v, k) for (k, v) in id_to_char.items()])
        return id_to_char, char_to_id

# --- Global Class for Normalization Parameters ---
class Global:
    train_mean = 0.0
    train_std = 1.0

# --- Normalization Functions ---
def train_offset_normalization(data):
    mean = data[:, :, 1:].mean(axis=(0, 1))
    data[:, :, 1:] -= mean
    std = data[:, :, 1:].std(axis=(0, 1))
    data[:, :, 1:] /= std
    return mean, std, data

def valid_offset_normalization(mean, std, data):
    data[:, :, 1:] -= mean
    data[:, :, 1:] /= std
    return data

def data_denormalization(mean, std, data):
    data[:, :, 1:] *= std
    data[:, :, 1:] += mean
    return data

# --- Model Class ---
class HandWritingSynthesisNet(torch.nn.Module):
    def __init__(self, hidden_size=400, n_layers=3, output_size=121, window_size=77):
        super(HandWritingSynthesisNet, self).__init__()
        self.vocab_size = window_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        K = 10
        self.EOS = False
        self._phi = []

        self.lstm_1 = torch.nn.LSTM(3 + self.vocab_size, hidden_size, batch_first=True)
        self.lstm_2 = torch.nn.LSTM(
            3 + self.vocab_size + hidden_size, hidden_size, batch_first=True
        )
        self.lstm_3 = torch.nn.LSTM(
            3 + self.vocab_size + hidden_size, hidden_size, batch_first=True
        )

        self.window_layer = torch.nn.Linear(hidden_size, 3 * K)
        self.output_layer = torch.nn.Linear(n_layers * hidden_size, output_size)

    def init_hidden(self, batch_size, device):
        initial_hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
        )
        window_vector = torch.zeros(batch_size, 1, self.vocab_size, device=device)
        kappa = torch.zeros(batch_size, 10, 1, device=device)
        return initial_hidden, window_vector, kappa

    def one_hot_encoding(self, text):
        N = text.shape[0]
        U = text.shape[1]
        encoding = text.new_zeros((N, U, self.vocab_size))
        for i in range(N):
            encoding[i, torch.arange(U), text[i].long()] = 1.0
        return encoding

    def compute_window_vector(self, mix_params, prev_kappa, text, text_mask, is_map):
        encoding = self.one_hot_encoding(text)
        mix_params = torch.exp(mix_params)

        alpha, beta, kappa = mix_params.split(10, dim=1)

        kappa = kappa + prev_kappa
        prev_kappa = kappa

        u = torch.arange(text.shape[1], dtype=torch.float32, device=text.device)

        phi = torch.sum(alpha * torch.exp(-beta * (kappa - u).pow(2)), dim=1)
        if phi[0, -1] > torch.max(phi[0, :-1]):
            self.EOS = True
        phi = (phi * text_mask).unsqueeze(2)
        if is_map:
            self._phi.append(phi.squeeze(dim=2).unsqueeze(1))

        window_vec = torch.sum(phi * encoding, dim=1, keepdim=True)
        return window_vec, prev_kappa

    def forward(
        self,
        inputs,
        text,
        text_mask,
        initial_hidden,
        prev_window_vec,
        prev_kappa,
        is_map=False,
    ):
        hid_1 = []
        window_vec = []

        state_1 = (initial_hidden[0][0:1], initial_hidden[1][0:1])

        for t in range(inputs.shape[1]):
            inp = torch.cat((inputs[:, t: t + 1, :], prev_window_vec), dim=2)

            hid_1_t, state_1 = self.lstm_1(inp, state_1)
            hid_1.append(hid_1_t)

            mix_params = self.window_layer(hid_1_t)
            window, kappa = self.compute_window_vector(
                mix_params.squeeze(dim=1).unsqueeze(2),
                prev_kappa,
                text,
                text_mask,
                is_map,
            )

            prev_window_vec = window
            prev_kappa = kappa
            window_vec.append(window)

        hid_1 = torch.cat(hid_1, dim=1)
        window_vec = torch.cat(window_vec, dim=1)

        inp = torch.cat((inputs, hid_1, window_vec), dim=2)
        state_2 = (initial_hidden[0][1:2], initial_hidden[1][1:2])

        hid_2, state_2 = self.lstm_2(inp, state_2)
        inp = torch.cat((inputs, hid_2, window_vec), dim=2)
        state_3 = (initial_hidden[0][2:], initial_hidden[1][2:])

        hid_3, state_3 = self.lstm_3(inp, state_3)

        inp = torch.cat([hid_1, hid_2, hid_3], dim=2)
        y_hat = self.output_layer(inp)

        return y_hat, [state_1, state_2, state_3], window_vec, prev_kappa

    def generate(
        self,
        inp,
        text,
        text_mask,
        prime_text,
        prime_mask,
        hidden,
        window_vector,
        kappa,
        bias,
        is_map=False,
        prime=False,
    ):
        seq_len = 0
        gen_seq = []
        with torch.no_grad():
            batch_size = inp.shape[0]
            if prime:
                y_hat, state, window_vector, kappa = self.forward(
                    inp, prime_text, prime_mask, hidden, window_vector, kappa, is_map
                )

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                hidden = (_hidden, _cell)
                self.EOS = False
                inp = inp.new_zeros(batch_size, 1, 3)
                _, window_vector, kappa = self.init_hidden(batch_size, inp.device)

            while not self.EOS and seq_len < 2000:
                y_hat, state, window_vector, kappa = self.forward(
                    inp, text, text_mask, hidden, window_vector, kappa, is_map
                )

                _hidden = torch.cat([s[0] for s in state], dim=0)
                _cell = torch.cat([s[1] for s in state], dim=0)
                hidden = (_hidden, _cell)
                y_hat = y_hat.squeeze()
                Z = sample_from_out_dist(y_hat, bias)
                inp = Z
                gen_seq.append(Z)

                seq_len += 1

        gen_seq = torch.cat(gen_seq, dim=1)
        gen_seq = gen_seq.cpu().numpy()

        return gen_seq

# --- Sampling Function ---
def sample_from_out_dist(y_hat, bias):
    split_sizes = [1] + [20] * 6
    y = torch.split(y_hat, split_sizes, dim=0)

    eos_prob = torch.sigmoid(y[0])
    mixture_weights = stable_softmax(y[1] * (1 + bias), dim=0)
    mu_1 = y[2]
    mu_2 = y[3]
    std_1 = torch.exp(y[4] - bias)
    std_2 = torch.exp(y[5] - bias)
    correlations = torch.tanh(y[6])

    bernoulli_dist = torch.distributions.Bernoulli(probs=eos_prob)
    eos_sample = bernoulli_dist.sample()

    K = torch.multinomial(mixture_weights, 1)

    mu_k = y_hat.new_zeros(2)

    mu_k[0] = mu_1[K]
    mu_k[1] = mu_2[K]
    cov = y_hat.new_zeros(2, 2)
    cov[0, 0] = std_1[K].pow(2)
    cov[1, 1] = std_2[K].pow(2)
    cov[0, 1], cov[1, 0] = (
        correlations[K] * std_1[K] * std_2[K],
        correlations[K] * std_1[K] * std_2[K],
    )

    x = torch.normal(mean=torch.Tensor([0.0, 0.0]), std=torch.Tensor([1.0, 1.0])).to(
        y_hat.device
    )
    Z = mu_k + torch.mv(cov, x)

    sample = y_hat.new_zeros(1, 1, 3)
    sample[0, 0, 0] = eos_sample.item()
    sample[0, 0, 1:] = Z
    return sample

def stable_softmax(X, dim=2):
    max_vec = torch.max(X, dim, keepdim=True)
    exp_X = torch.exp(X - max_vec[0])
    sum_exp_X = torch.sum(exp_X, dim, keepdim=True)
    X_hat = exp_X / sum_exp_X
    return X_hat

# --- Plotting Function ---
def plot_stroke(stroke, save_name=None):
    f, ax = plt.subplots()
    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])
    size_x = x.max() - x.min() + 1.0
    size_y = y.max() - y.min() + 1.0
    f.set_size_inches(5.0 * size_x / size_y, 5.0)
    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0
    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value], "k-", linewidth=3)
        start = cut_value + 1
    ax.axis("off")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name:
        try:
            plt.savefig(save_name, bbox_inches="tight", pad_inches=0.5)
            print(f"Saved plot to {save_name}")
        except Exception as e:
            print(f"Error saving image to {save_name}: {e}")

    plt.close()

# --- Animation Function ---
def animate_stroke_one_by_one(stroke, save_name=None):
    # Convert (dx, dy) to absolute positions
    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])
    pos = np.stack([x, y], axis=1)

    # Split into stroke segments
    cuts = np.where(stroke[:, 0] == 1)[0]
    segments = []
    start = 0
    for cut in cuts:
        segments.append(pos[start:cut])
        start = cut + 1
    if start < len(pos):
        segments.append(pos[start:])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')

    # Set limits based on overall stroke
    ax.set_xlim(pos[:, 0].min() - 10, pos[:, 0].max() + 10)
    ax.set_ylim(pos[:, 1].min() - 10, pos[:, 1].max() + 10)

    # Create line objects for each segment (thinner lines)
    lines = [ax.plot([], [], 'k-', linewidth=1.0)[0] for _ in segments]

    # Calculate when each segment should start and stop
    segment_lengths = [len(seg) for seg in segments]
    start_frames = np.cumsum([0] + segment_lengths[:-1])
    end_frames = np.cumsum(segment_lengths)
    total_frames = sum(segment_lengths)

    # Optional: Use easing (ease-out) to simulate natural drawing
    def ease_out(t):
        return 1 - (1 - t) ** 2  # Quadratic ease-out

    def update(frame):
        for i, (start_f, end_f) in enumerate(zip(start_frames, end_frames)):
            if frame < start_f:
                lines[i].set_data([], [])
            elif frame >= end_f:
                lines[i].set_data(segments[i][:, 0], segments[i][:, 1])
            else:
                idx_float = frame - start_f
                length = len(segments[i])
                t = idx_float / (end_f - start_f)
                eased = ease_out(t)
                idx = max(1, int(eased * length))
                lines[i].set_data(segments[i][:idx, 0], segments[i][:idx, 1])
        return lines

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=30,  # Faster animation
        blit=True
    )

    if save_name:
        ani.save(save_name, writer='pillow')
        print(f"Saved animation to {save_name}")

    plt.close(fig)  # Close the figure to free memory
    return ani

# --- Helper Function to Sanitize Filenames ---
def sanitize_filename(filename):
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Replace invalid characters for Windows filenames with underscores
    # Invalid characters: < > : " / \ | ? *
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Ensure the filename is not empty and is safe
    if not filename:
        filename = "default_output"
    return filename

# --- Main Function to Generate Handwriting ---
def generate_handwriting(input_text, model_path, data_path, output_dir, animate=False, bias=10.0, batch_size=1):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset to get vocabulary
    dataset = HandwritingDataset(data_path, split="train", text_req=True)
    char_to_id = dataset.char_to_id
    idx_to_char = dataset.idx_to_char

    # Validate input text against the vocabulary
    valid_chars = set(char_to_id.keys())
    input_chars = set(input_text)
    if not input_chars.issubset(valid_chars):
        invalid_chars = input_chars - valid_chars
        raise ValueError(f"Input contains invalid characters: {invalid_chars}. Valid characters are: {valid_chars}")

    # Initialize model
    model = HandWritingSynthesisNet(window_size=len(char_to_id))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Prepare input text
    char_seq = np.array(list(input_text + "  "))  # Add padding
    text = np.array(
        [[char_to_id[char] for char in char_seq] for _ in range(batch_size)]
    ).astype(np.float32)
    text = torch.from_numpy(text).to(device)
    text_mask = torch.ones(text.shape).to(device)

    # Initialize model inputs
    inp = torch.zeros(batch_size, 1, 3).to(device)
    hidden, window_vector, kappa = model.init_hidden(batch_size, device)

    # Generate sequence
    print(f"Generating handwriting for text: {input_text}")
    gen_seq = model.generate(
        inp,
        text,
        text_mask,
        prime_text=None,
        prime_mask=None,
        hidden=hidden,
        window_vector=window_vector,
        kappa=kappa,
        bias=bias,
        is_map=False,
        prime=False,
    )

    # Denormalize the generated sequence
    gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)

    # Sanitize the input text for the filename
    safe_input_text = sanitize_filename(input_text)

    # Save static plot (PNG)
    static_output_path = os.path.join(output_dir, f"handwriting_{safe_input_text}.png")
    plot_stroke(gen_seq[0], save_name=static_output_path)
    print(f"Generated static handwriting saved to {static_output_path}")

    # Save animation (GIF) if requested
    if animate:
        anim_output_path = os.path.join(output_dir, f"handwriting_{safe_input_text}.gif")
        animate_stroke_one_by_one(gen_seq[0], save_name=anim_output_path)
        print(f"Generated animated handwriting saved to {anim_output_path}")
        return anim_output_path
    else:
        return static_output_path

# --- Run the Generation (for standalone testing) ---
if __name__ == "__main__":
    from collections import Counter

    # Configuration
    MODEL_PATH = "C:\\Users\\Abdur Rehman\\Videos\\Handwriting_app\\model\\trainedmodel.pt"  # Path to your trained model
    DATA_PATH = "./data/"  # Path to strokes.npy and sentences.txt
    OUTPUT_DIR = "./outputs/"  # Directory to save generated images
    BIAS = 10.0  # Sampling bias

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Get user input
    input_text = input("Enter text to generate handwriting for: ")
    animate = input("Generate animation (GIF)? (y/n): ").lower() == 'y'

    # Generate handwriting and get the output path
    output_path = generate_handwriting(input_text, MODEL_PATH, DATA_PATH, OUTPUT_DIR, animate=animate, bias=BIAS)
    print(f"Output saved to: {output_path}")