# Skin-Disease-Detection-System
AI-powered skin disease detection system using deep learning, Flask backend, and a React frontend. Users upload skin images for instant analysis and risk assessment. Includes database integration, modular architecture, and real-time feedback to support early identification of potential skin conditions.




# Best Preprocessing Pipeline

This pipeline implements the **highest-performing preprocessing techniques** from your Researches:

- Hair & Artifact Removal → DullRazor + Inpainting (97.13% paper)
- Denoising → Median Filter (97.9% paper)
- Color Enhancement → CLAHE on V channel (used in ALL 96%+ models)

Just put your images in `Test_Images/` and run:
```bash
python Main_Pipeline.py