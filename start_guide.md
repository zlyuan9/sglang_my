# Browser Agent Complete Setup Guide

This guide walks you through setting up and running the complete browser agent system, which consists of three components:
1. **sglang_my**: The SGLang serving engine (modified version)
2. **sglang_log**: Scripts to launch and monitor the SGLang server
3. **BrowserUseScript**: The browser agent that uses the LLM for web automation

## System Overview
```
┌─────────────────────┐
│  BrowserUseScript   │  ← Browser automation agent
│  (agent_new_*.py)   │
└──────────┬──────────┘
           │ HTTP requests
           ↓
┌─────────────────────┐
│   sglang_log        │  ← Launch scripts
│ (start_sglang_*.sh) │
└──────────┬──────────┘
           │ launches
           ↓
┌─────────────────────┐
│    sglang_my        │  ← SGLang serving engine
│  (Python package)   │
└─────────────────────┘
```

---

## Prerequisites

- **GPU Access**: You are connected to the GPU host via VS Code Remote SSH (the integrated terminal runs on the GPU machine)
- **CUDA**: CUDA-enabled GPU (NVIDIA recommended)
- **Python**: Python 3.12+
- **Memory**: Model-dependent (Qwen3-VL-30B needs ~60GB GPU memory)

---

## Part 1: Environment Setup

### 1.1 Initial Setup (First Time Only)

You are already on the GPU host via VS Code Remote SSH. Use the VS Code integrated terminal for these steps.

Set up directories (if not already done):
```bash
cd ~
mkdir -p projects
cd projects

# Clone the repositories (if not already cloned)
git clone https://github.com/JhengLu/sglang_my.git
git clone https://github.com/JhengLu/sglang_log.git
git clone https://github.com/JhengLu/BrowserUseScript.git
```

### 1.2 Install Miniconda (if not installed)
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 1.3 Create Python Environment

Create the environment for BrowserUse:
```bash
conda create -n browseruse python=3.12 -y
conda activate browseruse
```

Install uv (fast Python package installer):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install browser-use and dependencies:
```bash
uv pip install browser-use
pip install steel-sdk
```

### 1.4 Install SGLang (sglang_my)

Navigate to the sglang_my directory and install in development mode:
```bash
cd sglang_my/python
pip install -e ".[all]"
```

This installs SGLang with all optional dependencies, allowing you to modify the source code.

### 1.5 Set Environment Variables

Set the HuggingFace models directory (adjust path as needed):
```bash
export HF_MODELS="/home/colinz/DSL/BrowserUse/models"
# Or if using shared models:
export HF_MODELS="/shared/models"
```

Add to `~/.bashrc` to make permanent:
```bash
echo 'export HF_MODELS="/home/colinz/DSL/BrowserUse/models"' >> ~/.bashrc
```

---

## Part 2: Download Model (One-time Setup)

### 2.1 Download Qwen3-VL Model

If the model isn't already available, download it:
```bash
cd ~/projects
python << EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
    local_dir="$HF_MODELS/Qwen/Qwen3-VL-30B-A3B-Instruct",
    local_dir_use_symlinks=False
)
EOF
```

This will download ~60GB of model files.

---

## Part 3: Launch SGLang Server

### 3.1 Configure GPU Visibility

Check available GPUs:
```bash
nvidia-smi
```

Set which GPUs to use (example using GPUs 0-3):
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 3.2 Launch Server (Interactive Mode)

Navigate to the sglang_log directory:
```bash
cd ~sglang_log
```

Launch the server:
```bash
./start_sglang_qwen3_2b.sh
```

**What happens:**
- Kills any existing SGLang processes
- Starts SGLang server on port 8000
- Auto-detects number of GPUs from CUDA_VISIBLE_DEVICES
- Creates logs in `qwen3vl-log/runtime_qwen3vl_<timestamp>.log`
- Waits for server to be ready (looks for "The server is fired up and ready to roll!")
- Sends test requests using `send_sglang_request.py`
- Keeps server running

**Monitor the logs:**
```bash
# In another terminal
tail -f sglang_log/qwen3vl-log/runtime_qwen3vl_*.log
```

### 3.3 Launch Server (Background/SLURM)

For running on a SLURM cluster:
```bash
cd ~/projects/sglang_log
sbatch run_slurm_qwen3vl_job.sh
```

Check job status:
```bash
squeue -u $USER
```

View output:
```bash
cat logs/sglang-serve-<job_id>.out
```

### 3.4 Verify Server is Running

Test the server with curl:
```bash
curl http://localhost:8000/v1/models
```

Expected output:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-VL-30B-A3B-Instruct",
      "object": "model",
      ...
    }
  ]
}
```

---

## Part 4: Configure BrowserUseScript

### 4.1 Set Up Environment File

Navigate to BrowserUseScript:
```bash
cd ~/projects/BrowserUseScript
```

Create/edit `.env` file:
```bash
cat > .env <<'EOF'
VLLM_URL=https://wantonly-subchorioidal-cinderella.ngrok-free.dev
MODEL_NAME=Qwen3-VL-2B-Instruct
STEEL_API_KEY=ste-JKqc61j7COb40Jm9nTfin7qglizy6WNSTAxYIg6Pb44IpAKLfzpBcjB0XdL4c2i1Qrfom8c2PO7xHgDflSFhDKhDBXvTqLLQsPC
EOF
```

**Key parameters:**
- `VLLM_URL`: SGLang server endpoint (use `/v1` suffix for OpenAI compatibility)
- `MODEL_NAME`: Model name as registered in SGLang
- `STEEL_API_KEY`: API key for Steel remote browser service

### 4.2 Configure Content Filtering (Optional)

Edit `agent_new_content_filter.py` to toggle features:
```python
# Line 20-22
ENABLE_NEW_CONTENT_FILTER = False  # Set to True to filter DOM to only new content
ENABLE_VISION_USAGE = "auto"       # "auto", True, or False
```

---

## Part 5: Run Browser Agent

### 5.1 Activate Environment
```bash
conda activate browseruse
cd ~/projects/BrowserUseScript
```

### 5.2 Run a Single Task
```bash
python agent_new_content_filter.py
```

By default, this will:
- Connect to SGLang server at `http://127.0.0.1:8000/v1`
- Use the Steel remote browser
- Create logs in `agent_logs/<timestamp>/`
- Save screenshots, actions, and browser states

### 5.3 Run Benchmark Suite

To run all 25 tasks from `tasks.md`:
```bash
# Set up environment variables
source setup_env.sh

# Run benchmark
python run_benchmark.py
```

**What it does:**
- Parses all 25 tasks from `tasks.md`
- Runs each task through the agent
- Saves results in `benchmark_results/run_<timestamp>/`
- Creates:
  - `results_summary.json`: Completion status and outputs
  - `expected_answers.json`: Template for validation

### 5.4 Validate Benchmark Results

After benchmark completes:

1. Fill in expected answers:
```bash
nano benchmark_results/run_<timestamp>/expected_answers.json
# Add correct answers in the "expected_answer" field for each task
```

2. Run validation:
```bash
python validate_results.py benchmark_results/run_<timestamp>/
```

This generates:
- `validation_report.json`: Accuracy metrics
- Shows which tasks passed/failed

---

## Part 6: Understanding the Logs

### 6.1 SGLang Server Logs

Location: `~/projects/sglang_log/qwen3vl-log/runtime_qwen3vl_<timestamp>.log`

Key things to look for:
- "The server is fired up and ready to roll!" → Server ready
- Token throughput metrics
- Memory usage
- Request/response times

### 6.2 Agent Logs

Location: `~/projects/BrowserUseScript/agent_logs/<timestamp>/`

Structure:
```
agent_logs/20260122_123456/
├── screenshots/              # Step-by-step screenshots
│   ├── step_001.png
│   ├── step_002.png
│   └── ...
├── vllm_requests/           # Requests sent to SGLang
│   ├── request_001/
│   │   ├── request.json
│   │   └── images/
│   └── ...
├── full_session.log         # Complete session log
├── actions.jsonl            # Agent actions (one per line)
└── browser_states.jsonl     # Browser states (one per line)
```

### 6.3 Benchmark Results

Location: `~/projects/BrowserUseScript/benchmark_results/run_<timestamp>/`

Files:
- `results_summary.json`: Task outcomes
- `expected_answers.json`: Validation template (you fill this)
- `validation_report.json`: Accuracy metrics

---

## Part 7: Common Workflows

Running chatui: python3 chat_server.py from browseruse

### 7.1 Daily Development Workflow

Terminal 1 (SGLang Server):
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ~/projects/sglang_log
./start_sglang_qwen3vl_default_attn.sh
```

Terminal 2 (Browser Agent):
```bash
conda activate browseruse
cd ~/projects/BrowserUseScript
python agent_new_content_filter.py
```

### 7.2 Quick Test Workflow

Test if everything works:
```bash
# Terminal 1: Start server
cd ~/projects/sglang_log
./start_sglang_qwen3vl_default_attn.sh

# Terminal 2: Run quick test
cd ~/projects/sglang_log
python send_sglang_request.py
```

### 7.3 Debugging Workflow

If something goes wrong:

1. Check SGLang server status:
```bash
curl http://localhost:8000/v1/models
```

2. Check GPU memory:
```bash
nvidia-smi
```

3. View recent logs:
```bash
tail -100 ~/projects/sglang_log/qwen3vl-log/runtime_qwen3vl_*.log
```

4. Kill stuck processes:
```bash
pkill -f sglang.launch_server
```

---

## Part 8: Customization and Advanced Usage

### 8.1 Modify Task List

Edit tasks:
```bash
nano ~/projects/BrowserUseScript/tasks.md
```

Format:
```markdown
## Category Name
Task description: "Detailed task instructions here"

Another task: "More instructions"
```

### 8.2 Change Model

To use a different model:

1. Download the model to `$HF_MODELS`
2. Edit `start_sglang_qwen3vl_default_attn.sh`:
```bash
MODEL_PATH="${HF_MODELS}/path/to/your/model"
```
3. Update `.env` in BrowserUseScript:
```bash
MODEL_NAME=your-model-name
```

### 8.3 Adjust Server Parameters

Edit `start_sglang_qwen3vl_default_attn.sh`:
```bash
# Key parameters:
--tp ${TP_SIZE}              # Tensor parallel size (number of GPUs)
--context-length 40960       # Max context length
--port 8000                  # Server port
--log-level debug            # Logging verbosity
```

### 8.4 Enable Attention Logging

For your Quest research, enable attention weight logging:
```bash
export SGLANG_SAVE_ATTN=1
export SGLANG_SAVE_ATTN_LAYERS=0,1,9,16,23
export SGLANG_SAVE_ATTN_HEADS=0,1
```

Then use `start_sglang_qwen3vl_sdpa_save_attn.sh` instead.

---

## Part 9: Troubleshooting

### Problem: "Could not connect to SGLang server"

**Solution:**
```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Restart server
cd ~/projects/sglang_log
pkill -f sglang.launch_server
./start_sglang_qwen3vl_default_attn.sh
```

### Problem: "CUDA out of memory"

**Solution:**
```bash
# Use fewer GPUs or smaller model
export CUDA_VISIBLE_DEVICES=0,1  # Use only 2 GPUs

# Or reduce context length in launch script:
--context-length 20480  # Instead of 40960
```

### Problem: Server takes too long to start

**Solution:**
- Check logs: `tail -f ~/projects/sglang_log/qwen3vl-log/runtime_qwen3vl_*.log`
- Model loading can take 5-10 minutes for large models
- Wait for "The server is fired up and ready to roll!"

### Problem: Agent fails with timeout

**Solution:**
```bash
# Increase timeout in agent_new_content_filter.py
# Edit the ChatOpenAI initialization:
llm = ChatOpenAI(
    base_url=VLLM_URL,
    model_name=MODEL_NAME,
    timeout=1200  # Increase from default 600
)
```

---

## Part 10: Quick Reference Commands

### Start Everything
```bash
# Terminal 1: SGLang Server
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ~/projects/sglang_log
./start_sglang_qwen3vl_default_attn.sh

# Terminal 2: Browser Agent
conda activate browseruse
cd ~/projects/BrowserUseScript
source setup_env.sh
python agent_new_content_filter.py
```

### Stop Everything
```bash
# Kill SGLang server
pkill -f sglang.launch_server

# Deactivate conda
conda deactivate
```

### Check Status
```bash
# Server status
curl http://localhost:8000/v1/models

# GPU usage
nvidia-smi

# View logs
tail -f ~/projects/sglang_log/qwen3vl-log/runtime_qwen3vl_*.log
tail -f ~/projects/BrowserUseScript/agent_logs/*/full_session.log
```

---

## Part 11: File Locations Cheat Sheet
```
~/projects/
├── sglang_my/                          # SGLang engine (Python package)
│   └── python/sglang/launch_server.py  # Main server launcher
│
├── sglang_log/                         # Launch scripts & logging
│   ├── start_sglang_qwen3vl_default_attn.sh   # Main launch script
│   ├── send_sglang_request.py          # Test script
│   ├── qwen3vl-log/                    # Server logs
│   └── run_slurm_qwen3vl_job.sh        # SLURM job script
│
└── BrowserUseScript/                   # Browser agent
    ├── .env                            # Environment configuration
    ├── agent_new_content_filter.py     # Main agent script
    ├── tasks.md                        # Task definitions
    ├── run_benchmark.py                # Benchmark runner
    ├── agent_logs/                     # Agent execution logs
    └── benchmark_results/              # Benchmark outputs
```

---

## Next Steps for Your Research

Given your Quest project focus, here are suggested next steps:

1. **Get familiar with the system:**
   - Run a simple agent task
   - Examine the logs in `agent_logs/`
   - Look at attention patterns (if enabled)

2. **Understand the data flow:**
   - BrowserUseScript → HTTP request → SGLang → Model inference
   - Trace a single request through the logs

3. **Enable attention logging:**
   - Use `start_sglang_qwen3vl_sdpa_save_attn.sh`
   - Examine attention weights for KV cache analysis

4. **Modify SGLang for Quest:**
   - Your Quest implementation lives in `sglang_my/`
   - Test modifications by running browser tasks
   - Measure latency improvements

---

## Support and Resources

- **SGLang Documentation**: https://docs.sglang.ai/
- **Browser-Use GitHub**: https://github.com/browser-use/browser-use
- **Steel Browser**: https://steel.dev/

For help, reach out to your advisor Jiaheng or check the respective GitHub repositories.

---

**Last Updated**: January 2025
**Author**: Setup guide for Quest project