# Mini Attention Notebook: Q/K/V Structural Mechanics & Self-Attention Visualizer

Welcome to the **Mini Attention Notebook**, an interactive educational dashboard designed to demystify how Transformer Self-Attention operates mathematically and visually. This project is part of the **DevJam LLM Learning 10-Project Series**.

---
<img width="1920" height="1225" alt="today" src="https://github.com/user-attachments/assets/0057b5a4-0524-4b3e-bfbf-6caff8499ef4" />

## 📖 Architectural Documentation Hub

As a novice student, we recommend reading the architecture and engineering documents below before writing any code:

1. 📋 **[Requirements Brief (docs/requirements.md)](file:///c:/Users/AVICK/Avick_Projects/ai_project/mini-attention-notebook/docs/requirements.md)**: Product expectations, boundaries, API contract schemas, and acceptance criteria.
2. 📐 **[Architectural Decision Record (docs/adr-architecture-decisions.md)](file:///c:/Users/AVICK/Avick_Projects/ai_project/mini-attention-notebook/docs/adr-architecture-decisions.md)**: Tech stack choices, compute locality decisions, and state visualization design systems.
3. 🔢 **[Linear Algebra Guide (docs/linear-algebra.md)](file:///c:/Users/AVICK/Avick_Projects/ai_project/mini-attention-notebook/docs/linear-algebra.md)**: Foundation of vectors, matrix multiplication constraints, and transposes.
4. 🧠 **[Self-Attention Mechanics (docs/self-attention.md)](file:///c:/Users/AVICK/Avick_Projects/ai_project/mini-attention-notebook/docs/self-attention.md)**: The mathematical breakdown of $Q$, $K$, $V$ projection formulas.
5. 🛡️ **[Causal Masking (docs/causal-masking.md)](file:///c:/Users/AVICK/Avick_Projects/ai_project/mini-attention-notebook/docs/causal-masking.md)**: Autoregressive decoding logic and the $-\infty$ masking trick.
6. 🧩 **[Multi-Head Attention (docs/multi-head-attention.md)](file:///c:/Users/AVICK/Avick_Projects/ai_project/mini-attention-notebook/docs/multi-head-attention.md)**: Splitting model dimensions, representation subspaces, and final projections.
7. 🎓 **[Interview Defense Guide (docs/interview-defense.md)](file:///c:/Users/AVICK/Avick_Projects/ai_project/mini-attention-notebook/docs/interview-defense.md)**: Clear, production-ready answers to the 5 core attention questions asked in ML interviews.

---

## 🛠️ Project Tech Stack

* **Frontend**: Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, shadcn/ui, Zustand (Global State Store), React Query
* **Backend**: FastAPI (Python 3.10+), NumPy (Math calculation engine), Pydantic (Data validation)
* **Testing**: Pytest

---

## 📂 Folder Structure

```text
mini-attention-notebook/
├── apps/
│   ├── web-client/                # Next.js 15 Dashboard App
│   └── math-api/                  # FastAPI Backend API
├── packages/
│   └── shared-types/              # Shared Typescript Definitions
├── docs/
│   ├── requirements.md
│   ├── adr-architecture-decisions.md
│   ├── linear-algebra.md
│   ├── self-attention.md
│   ├── causal-masking.md
│   ├── multi-head-attention.md
│   └── interview-defense.md
└── README.md                      # This File
```

---

## 🚀 Setup & Local Development

### 1. Backend Setup (FastAPI + NumPy)
Navigate to the math API app directory:
```bash
cd apps/math-api
```
Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Run the FastAPI development server:
```bash
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup (Next.js 15)
Navigate to the web client app directory:
```bash
cd apps/web-client
```
Install dependencies and run the development server:
```bash
npm install
npm run dev
```
Open `http://localhost:3000` in your web browser.
