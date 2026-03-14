import nbformat


nb = nbformat.v4.new_notebook()
cells = []

def code(source):
    cells.append(nbformat.v4.new_code_cell(source))

def markdown(source):
    cells.append(nbformat.v4.new_markdown_cell(source))

# Cell 1: GPU check
markdown("## 1. Verify GPU")
code("""/
import torch
print(torch.cuda.get_device_name(0))
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
assert 'A100' in torch.cuda.get_device_name(0), 'WARNING: Not an A100 - need to change runtimes to aquire for training'      
""")

# Cell 2: Mount Google drive
markdown("## 2. Mount Google Drive (checkpoint persistence)")
code("""/
from google.colab import drive
drive.mount('/content/drive)     

import os
CHECKPOINT_DIR = '/content/drive/MyDrive/vektor-guard/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f'Checkppoint dir: {CHECKPOINT_DIR}')
""")

# Cell 3: Clone repo
markdown("## 3. Clone repo and install dependencies")
code("""\
import os
if not os.path.exists('/content/vektor'):
    !git clone https://github.com/emsikes/vektor.git /content/vektor
%cd /content/vektor/platform
!pip install -r requirements.txt -q
""")

# ── Cell 4: Auth ───────────────────────────────────────────────────────────────
markdown("## 4. Authenticate HuggingFace and WandB")
code("""\
from huggingface_hub import login as hf_login
from google.colab import userdata
import wandb

# secrets stored in Colab Secrets (left sidebar → key icon)
hf_login(token=userdata.get('HF_TOKEN'))
wandb.login(key=userdata.get('WANDB_API_KEY'))
""")

# ── Cell 5: Upload data splits ─────────────────────────────────────────────────
markdown("## 5. Upload data splits")
code("""\
# upload train.json, val.json, test.json from your local machine
# Runtime → upload files, or copy from Drive if already uploaded
import os
os.makedirs('data/splits', exist_ok=True)
from google.colab import files
print('Upload train.json, val.json, test.json when prompted')
uploaded = files.upload()
for fname, data in uploaded.items():
    with open(f'data/splits/{fname}', 'wb') as f:
        f.write(data)
    print(f'Saved data/splits/{fname}')
""")

# ── Cell 6: Train ──────────────────────────────────────────────────────────────
markdown("## 6. Train")
code("""\
import sys
sys.path.insert(0, '/content/vektor/platform')

from src.training.trainer import build_trainer

# point output_dir to Drive so checkpoints survive session expiry
trainer = build_trainer()
trainer.args.output_dir = CHECKPOINT_DIR

trainer.train()
""")

# ── Cell 7: Evaluate on test set ───────────────────────────────────────────────
markdown("## 7. Evaluate on test set")
code("""\
from src.training.dataset import load_split, build_tokenizer, tokenize_split
from src.training.metrics import compute_metrics, check_targets
import numpy as np

config_model = 'answerdotai/ModernBERT-large'
tokenizer = build_tokenizer(config_model)
test_dataset = tokenize_split(load_split('test'), tokenizer)

# run inference on test set using best checkpoint
predictions = trainer.predict(test_dataset)
metrics = compute_metrics((predictions.predictions, predictions.label_ids))

print(metrics)
check_targets(metrics)
""")

# ── Cell 8: Push to HuggingFace Hub ───────────────────────────────────────────
markdown("## 8. Push best model to HuggingFace Hub")
code("""\
trainer.model.push_to_hub('theinferenceloop/vektor-guard-v1')
tokenizer.push_to_hub('theinferenceloop/vektor-guard-v1')
print('Model pushed to HuggingFace Hub')
""")

nb.cells = cells

with open('notebooks/train_colab.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('Notebook written to notebooks/train_colab.ipynb')