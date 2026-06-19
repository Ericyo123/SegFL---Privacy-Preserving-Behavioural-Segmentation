from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import uuid
import json
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Optional
from ml.processor import process_csv, prepare_tenant_datasets
from ml.segmenter import (
    TAL_Adapter, GlobalBottleneckAE, FederatedKMeans, 
    formal_aggregator, compute_epsilon
)
from sklearn.metrics import silhouette_score, davies_bouldin_score

app = FastAPI(title="SegFL API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "online", "message": "SegFL Backend is running"}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory storage for jobs and results
jobs = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_training_task(job_id: str, file_path: str, nrows: int, g_epochs: int, l_epochs: int, sigma: float):
    try:
        jobs[job_id]["status"] = "processing_data"
        raw_df = process_csv(file_path, nrows=nrows)
        tr_dls, ev_dls, raw_info = prepare_tenant_datasets(raw_df)
        
        counts = [len(dl.dataset) for dl in tr_dls]
        shared_dim = 4
        glob_in = shared_dim # default for TAL
        
        # Training logic
        jobs[job_id]["status"] = "training_ml"
        glob_m = GlobalBottleneckAE(glob_in, shared_dim).to(device)
        adapters = [TAL_Adapter(r['dim'], shared_dim).to(device) for r in raw_info]
        
        lr = 0.005
        
        for g_rnd in range(g_epochs):
            st_collection = []
            for t_idx, dl in enumerate(tr_dls):
                loc_m = GlobalBottleneckAE(glob_in, shared_dim).to(device)
                loc_m.load_state_dict(glob_m.state_dict())
                params = list(loc_m.parameters()) + list(adapters[t_idx].parameters())
                opt = optim.Adam(params, lr=lr)
                
                for _ in range(l_epochs):
                    loc_m.train()
                    adapters[t_idx].train()
                    for b in dl:
                        original_x = b[0].to(device)
                        opt.zero_grad()
                        
                        a_out, a_rec = adapters[t_idx](original_x)
                        g_rec, _ = loc_m(a_out)
                        
                        loss = (1.0 * F.mse_loss(a_rec, original_x)) + (1.0 * F.mse_loss(g_rec, a_out))
                        loss.backward()
                        
                        if sigma > 0.0:
                            torch.nn.utils.clip_grad_norm_(params, 1.0)
                            for p in params:
                                if p.grad is not None:
                                    p.grad += torch.randn_like(p.grad) * sigma
                        
                        opt.step()
                
                st_collection.append(loc_m.state_dict())
                
            agr_dict = formal_aggregator(st_collection, counts)
            glob_m.load_state_dict(agr_dict)
            jobs[job_id]["progress"] = int(((g_rnd + 1) / g_epochs) * 100)

        # Evaluation & Clustering
        jobs[job_id]["status"] = "clustering"
        
        def eval_set(model, test_dl, adapter):
            model.eval()
            adapter.eval()
            lats = []
            with torch.no_grad():
                for b in test_dl:
                    x = b[0].to(device)
                    _, lat = model(adapter(x)[0])
                    lats.append(lat.cpu().numpy())
            return np.vstack(lats)

        lats = [eval_set(glob_m, ev_dls[i]['test'], adapters[i]) for i in range(len(tr_dls))]
        
        n_clusters = 5
        fed_k = FederatedKMeans(n_clusters=n_clusters)
        fed_labels = fed_k.fit_predict_federated(lats)
        
        # Calculate scores
        results = {
            "metrics": [],
            "profiles": [],
            "privacy": {
                "sigma": sigma,
                "epsilon": compute_epsilon(sigma, g_epochs * l_epochs, sum(counts))
            }
        }
        
        all_sils = []
        all_dbis = []
        for i, lt in enumerate(lats):
            if len(np.unique(fed_labels[i])) > 1:
                sil = silhouette_score(lt, fed_labels[i])
                dbi = davies_bouldin_score(lt, fed_labels[i])
                all_sils.append(float(sil))
                all_dbis.append(float(dbi))
        
        results["avg_silhouette"] = float(np.mean(all_sils)) if all_sils else 0
        results["avg_dbi"] = float(np.mean(all_dbis)) if all_dbis else 0
        
        # Semantic Profiles (Tenant 0 as sample)
        eval_target_df = raw_info[0]['raw_target'].copy()
        # Take the end of the dataframe as 'eval' portion (simplified)
        test_size = len(fed_labels[0])
        eval_target_df = eval_target_df.tail(test_size)
        eval_target_df['cluster'] = fed_labels[0]
        
        profile = eval_target_df.groupby('cluster').mean(numeric_only=True).round(3).to_dict(orient='index')
        sizes = eval_target_df.groupby('cluster').size().to_dict()
        
        profile_list = []
        for c_id, vals in profile.items():
            vals['cluster'] = int(c_id)
            vals['size'] = int(sizes[c_id])
            profile_list.append(vals)
            
        results["profiles"] = profile_list
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results"] = results
        
    except Exception as e:
        import traceback
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["traceback"] = traceback.format_exc()

@app.post("/train")
async def train(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    use_local: bool = Form(False),
    nrows: int = Form(50000),
    g_epochs: int = Form(5),
    l_epochs: int = Form(2),
    sigma: float = Form(0.1)
):
    job_id = str(uuid.uuid4())
    
    if use_local:
        file_path = os.path.join("data", "clicks_train.csv")
        if not os.path.exists(file_path):
            # Try to find any CSV in data folder
            csvs = [f for f in os.listdir("data") if f.endswith(".csv")]
            if csvs:
                file_path = os.path.join("data", csvs[0])
            else:
                return {"error": "Local data folder 'backend/data' is empty. Please upload a file or add CSVs to the data folder."}
        filename = os.path.basename(file_path)
    else:
        if not file:
            return {"error": "No file uploaded and local mode is disabled."}
        file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        filename = file.filename
    
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "filename": filename,
        "params": {"nrows": nrows, "g_epochs": g_epochs, "l_epochs": l_epochs, "sigma": sigma, "local": use_local}
    }
    
    background_tasks.add_task(run_training_task, job_id, file_path, nrows, g_epochs, l_epochs, sigma)
    
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    return jobs[job_id]

if __name__ == "__main__":
    print("🚀 Starting SegFL Backend on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
