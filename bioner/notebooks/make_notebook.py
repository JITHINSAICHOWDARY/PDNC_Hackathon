"""Builds the two Kaggle notebooks from the cell lists below (easier to read and edit than raw .ipynb JSON).
  kaggle_train.ipynb : tests, training, evaluation, submission   (run with Save & Run All)
  kaggle_demo.ipynb  : serves the FastAPI backend through a tunnel (run interactively for the demo)
Run from bioner/:  python notebooks/make_notebook.py"""
import json
from pathlib import Path

# Shared first cell. Code comes from the bioner-code dataset; checkpoints + submission come from a
# finished training run (its notebook output added as an input), if there is one.
SETUP = (
    "import glob, os, shutil\n"
    "W = '/kaggle/working/bioner'\n"
    "# the bundle is the only input with requirements.txt next to src/ (a training output has a bioner/src copy without it)\n"
    "cfg = [c for c in glob.glob('/kaggle/input/**/src/config.py', recursive=True)\n"
    "       if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(c)), 'requirements.txt'))]\n"
    "assert cfg, 'bioner-code dataset not found: Add Input -> your dataset'\n"
    "root = os.path.dirname(os.path.dirname(cfg[0]))\n"
    "for sub in ['src', 'data', 'backend']:                 # refresh code, keep existing checkpoints\n"
    "    shutil.rmtree(f'{W}/{sub}', ignore_errors=True)\n"
    "    shutil.copytree(f'{root}/{sub}', f'{W}/{sub}')\n"
    "ck = glob.glob('/kaggle/input/**/bioner/checkpoints/span/span_model.pt', recursive=True)\n"
    "if ck and not os.path.exists(f'{W}/checkpoints/span'):   # reuse a finished training run\n"
    "    prev = ck[0].split('/checkpoints/')[0]\n"
    "    for sub in ['checkpoints', 'submission']:\n"
    "        if os.path.exists(f'{prev}/{sub}'):\n"
    "            shutil.copytree(f'{prev}/{sub}', f'{W}/{sub}', dirs_exist_ok=True)\n"
    "    print('copied checkpoints from', prev)\n"
    "%cd /kaggle/working/bioner\n"
    "!pip install -q seqeval fastapi uvicorn   # torch, transformers, pandas, sklearn are preinstalled\n"
    "!nvidia-smi --query-gpu=name,memory.total --format=csv || echo 'no GPU: running on CPU'"
)

TRAIN = [
    ("md", "# Team Lannisters: Span-based BioNER on Kaggle (training)\n"
           "Settings (right panel): **Accelerator = GPU T4 x2** (we use one GPU), **Internet = On**.\n"
           "Add your `bioner-code` dataset via **Add Input**.\n\n"
           "**Two ways to run**\n"
           "- *Interactive*: run the setup cell, then only the cells for the phase you are testing.\n"
           "- *Final run*: **Save Version -> Save & Run All (Commit)**. It runs every cell in the background "
           "(about 30-40 min on a T4), so you can close the browser. All outputs land in the version's **Output** tab."),
    ("md", "## Setup: copy code + data into /kaggle/working/bioner"),
    ("code", SETUP),
    ("code", "import torch\n"
             "assert torch.cuda.is_available(), 'No GPU: Settings -> Accelerator -> GPU T4 x2, then restart'"),
    ("md", "## Phase 1: data tests"),
    ("code", "!python src/test_data.py"),
    ("md", "## Phase 2: baseline (debug run, then full run)"),
    ("code", "!python src/train.py --model baseline --debug"),
    ("code", "!python src/train.py --model baseline"),
    ("md", "## Phase 3: span model shape check"),
    ("code", "!python src/span_model.py"),
    ("md", "## Phase 4: span model (debug run, then full run)"),
    ("code", "!python src/sampling.py && python src/decode.py      # tiny self-checks"),
    ("code", "!python src/train.py --model span --debug"),
    ("code", "!python src/train.py --model span"),
    ("md", "## Phase 5: tune threshold + compare both models on validation"),
    ("code", "!python src/evaluate.py"),
    ("md", "## Phase 6: test-set predictions -> submission CSVs"),
    ("code", "!python src/predict.py\n"
             "!python src/predict.py --model baseline\n"
             "!cp submission/submission.csv /kaggle/working/submission.csv   # easy to find in Output"),
    ("md", "## Package results for download\n"
           "Zips `checkpoints/baseline`, `checkpoints/span`, `results.md` and `submission/` into "
           "`/kaggle/working/checkpoints.zip` (about 800 MB). Download it from the Output panel, then on your laptop:\n"
           "```\ncd ~/Desktop/PDNC && unzip -o ~/Downloads/checkpoints.zip\n```"),
    ("code", "!rm -rf checkpoints/debug && cd /kaggle/working && zip -rq checkpoints.zip bioner/checkpoints bioner/results.md bioner/submission && ls -lh checkpoints.zip"),
]

DEMO = [
    ("md", "# Team Lannisters: BioNER demo backend\n"
           "Run this **interactively** (not Save & Run All) and keep the tab open during the demo.\n\n"
           "Inputs (right panel -> **Add Input**):\n"
           "1. your `bioner-code` dataset (latest version)\n"
           "2. the **output of your finished `kaggle_train` run** (Add Input -> Your Work / Notebooks -> pick it). "
           "This supplies the trained checkpoints and `submission.csv`.\n\n"
           "Settings: **Accelerator = GPU T4** (fast answers; CPU also works, just slower), **Internet = On**."),
    ("code", SETUP),
    ("md", "## Start the FastAPI backend in the background"),
    # Popen keeps the server attached to the notebook kernel (a "!cmd &" shell can be reaped)
    ("code", "import subprocess\n"
             "server = subprocess.Popen('python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000',\n"
             "                          shell=True, stdout=open('backend.log', 'w'), stderr=subprocess.STDOUT)"),
    ("code", "import json, time, urllib.request\n"
             "def call(path, body=None):\n"
             "    req = urllib.request.Request('http://localhost:8000' + path,\n"
             "                                 data=json.dumps(body).encode() if body else None,\n"
             "                                 headers={'Content-Type': 'application/json'})\n"
             "    return json.loads(urllib.request.urlopen(req).read())\n"
             "up = False\n"
             "for _ in range(60):                     # loading two BioBERTs takes ~30-60 s\n"
             "    if server.poll() is not None:        # the server process died: show why at once\n"
             "        break\n"
             "    try:\n"
             "        print(call('/health')); up = True; break\n"
             "    except Exception:\n"
             "        time.sleep(5)\n"
             "if not up:\n"
             "    print(open('backend.log').read()[-3000:]); raise RuntimeError('backend did not start')"),
    ("md", "## Phase 8 checks: every endpoint"),
    ("code", "text = 'Mutations in the BRCA1 gene cause hereditary breast and ovarian cancer.'\n"
             "s = call('/predict', {'text': text, 'model': 'span'})\n"
             "print('tokens+tags:', list(zip(s['tokens'], s['tags'])))\n"
             "print('span entities:', s['entities'])\n"
             "print('top 5 candidates:', s['candidates'][:5])\n"
             "print('grid:', len(s['grid']), 'x', len(s['grid'][0]), ' threshold:', s['threshold'])\n"
             "b = call('/predict', {'text': text, 'model': 'baseline'})\n"
             "print('baseline entities:', b['entities'])\n"
             "c = call('/compare', {'text': text})\n"
             "print('disagree at word indices:', c['disagree'])\n"
             "print('examples:', call('/examples')['examples'][:3])"),
    ("md", "## Open a public tunnel so the frontend on your laptop can reach this backend\n"
           "Cloudflare quick tunnel: no account needed. It only forwards HTTP to our own server; "
           "all predictions are still computed by our models here on Kaggle."),
    ("code", "![ -f cloudflared ] || wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64\n"
             "!chmod +x cloudflared\n"
             "tunnel = subprocess.Popen('./cloudflared tunnel --url http://localhost:8000', shell=True,\n"
             "                          stdout=open('tunnel.log', 'w'), stderr=subprocess.STDOUT)"),
    ("code", "import re\n"
             "for _ in range(30):\n"
             "    urls = re.findall(r'https://[a-z0-9-]+\\.trycloudflare\\.com', open('tunnel.log').read())\n"
             "    if urls:\n"
             "        print('BACKEND URL (paste into the frontend):', urls[0]); break\n"
             "    time.sleep(2)\n"
             "else:\n"
             "    print(open('tunnel.log').read()[-2000:])"),
]


def cell(kind, text):
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
    # A failing "!cmd" does not stop a notebook, so check IPython's _exit_code after each one;
    # otherwise "Save & Run All" would carry on after a crashed training run.
    out = []
    for line in text.splitlines():
        out.append(line)
        if line.startswith("!"):
            out.append("assert _exit_code == 0, 'command failed, see output above'")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in out[:-1]] + out[-1:]}


for name, cells in [("kaggle_train.ipynb", TRAIN), ("kaggle_demo.ipynb", DEMO)]:
    nb = {
        "cells": [cell(k, t) for k, t in cells],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                     "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).with_name(name)
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)
