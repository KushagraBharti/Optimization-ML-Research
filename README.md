# Optimization‑ML‑Research

**Phase 1: Hybrid Supervised + RL for Covering Line Segments with Drone Tours**

---

## 📚 Overview

This repository implements **Phase 1** of a larger optimization–ML research project:

1. **Classical Solvers**  
   - **GS**: Greedy strategy to minimize number of tours  
   - **DP**: Dynamic‑programming to minimize total distance  
2. **Data Generation**  
   - Synthetic, random segment instances
3. **Supervised Pre‑training**  
   - Pointer‑network to imitate DP tours  
4. **RL Fine‑tuning**  
   - PPO agent to adapt to dynamic changes  
5. **Evaluation**  
   - Compare GS, DP and RL on held‑out instances  

---

## 🔧 Setup

1. **Clone & enter repo**
   ```bash
   git clone https://github.com/KushagraBharti/Optimization-ML-Research.git
   cd Optimization-ML-Research```

2. **Create & Activate Env** 
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1```

2. **Install Dependencies** 
   ```bash
   python.exe -m pip install --upgrade pip
   pip install -r requirements.txt
   $env:PYTHONPATH = $PWD```