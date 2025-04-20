# Optimization-ML-Research
Problem 1 first with Hybrid Supervised + RL (Phase 1), then Neural Combinatorial Optimization (Phase 2). Afterwards, move to Problem 2 with two analogous approaches (Phases 3 and 4).

## Project Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/KushagraBharti/Optimization-ML-Research.git
   cd Optimization-ML-Research
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions for the Docker Container

1. Build the Docker image:
   ```bash
   docker build -t optimization-ml-research .
   ```

2. Run the Docker container:
   ```bash
   docker run -it --rm optimization-ml-research
   ```

## Project Structure

```
/drones_project
  /data - raw and processed data
  /src
  /envs - Gym environment code
  /models - network definitions
  /solvers - classical GS+DP implementations
  /train - training scripts
  /eval - evaluation notebooks
  /docs
  /notebooks
requirements.txt
README.md
Dockerfile
```
