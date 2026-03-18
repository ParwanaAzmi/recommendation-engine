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
