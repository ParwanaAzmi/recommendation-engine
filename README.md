# Hybrid Recommendation Engine
This project builds a production-grade recommendation system using:
- Matrix Factorization (SVD)
- Neural Collaborative Filtering
- Redis Caching for fast inference
- A/B Testing framework
  
## Dataset
MovieLens 25M Dataset

## Task 1
Data Ingestion & Exploratory Data Analysis
- Rating distribution analysis
- User activity histogram
- Movie popularity bias analysis
- User-item interaction matrix (CSR sparse)
- Cold-start user detection
  
## Task 2: Matrix Factorization (FunkSVD)
- Implemented latent factor model using PyTorch
- User and item embeddings
- Bias terms for users and items
- Mini-batch training with Adam optimizer
- RMSE evaluation
  
## Task 3: Neural Collaborative Filtering (Two-Tower Model)
- Implemented Two-Tower architecture
- Separate user and item embedding networks
- Bayesian Personalized Ranking (BPR) loss
- Negative sampling strategy
- Optimized using Adam
- Scalable architecture for production systems
  
- ## Task 4: Content-Based Recommendation (FAISS + SBERT)
- Used Sentence-BERT (all-MiniLM-L6-v2) for semantic embeddings
- Combined movie title and genres as input features
- Built FAISS index for fast similarity search
- Achieved sub-millisecond nearest neighbor retrieval
- Implemented content-based recommendation function

  ## Task 5: Hybrid Recommendation & Re-ranking
- Combined Collaborative Filtering, Content-Based, and Popularity signals
- Applied weighted hybrid scoring (0.5, 0.3, 0.2)
- Implemented diversity-aware re-ranking using genre penalties
- Improved recommendation variety
- Evaluated system using novelty metric

  ## Task 6: Production Deployment (FastAPI + Redis)
- Built REST API using FastAPI
- Implemented Redis caching for low-latency inference
- Achieved simulated <10ms response time for cached results
- Added endpoints:
  - /recommend/{user_id}
  - /similar/{item_id}
  - /trending
  - /health
- Simulated production traffic using TestClient

  ## Task 7: A/B Testing & Model Evaluation
- Compared SVD, Neural CF, and Hybrid models
- Evaluated using CTR and WTR metrics
- Visualized results using Matplotlib
- Implemented automatic winner selection
- Hybrid model showed highest engagement
- Simulated production rollout decision



