# RF-LeWM Demo Backend

FastAPI server for the RF-LeWM demo. Runs on CPU.

## Setup

```bash
# From project root
cd demo/backend

# Place data files in demo/data/:
#   test.h5              — test dataset
#   norm_stats.json      — normalization statistics
#   lewm_rf_epoch_99_object.ckpt — model checkpoint

# Install deps
pip install -r requirements.txt

# Run
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | Dataset statistics |
| GET | `/regimes` | Regime counts |
| GET | `/trajectory/{id}` | Raw trajectory + metadata |
| POST | `/predict` | Encode + predict + surprise + PCA |
| POST | `/inject` | Perturbation experiment |
| GET | `/pca_background` | Pre-computed PCA scatter |
