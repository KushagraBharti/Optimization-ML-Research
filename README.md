## Installation

I recommend using **Anaconda** AND **uv** for managing everything:

```bash
conda create -n mlresearch python=3.13 -y
conda activate mlresearch

# sanity checks
where.exe python    # should point into .../anaconda3/envs/mlresearch
python --version

# install package in editable mode with dev dependencies
uv sync --all-packages
```