Markdown# 🔬 SEM Drift Sub-Pixel Registration & Inspection Suite

An ultra-high precision, real-time Sub-Pixel Registration and Defect Analysis Suite built for semiconductor metrology. Powered by **NVIDIA CUDA (cuFFT)** and a **C++20 multi-threaded fallback engine (FFTW3)**, this tool achieves sub-pixel alignment accuracy under **1.5 ms** latency.

---

## 📌 Executive Summary & Key Highlights
* **High Precision:** Sub-pixel alignment ($\le \pm 0.5\text{ px}$) using parabolic sub-grid Taylor refinement.
* **Ultra-Fast Performance:** $<1.5\text{ ms}$ processing time per frame via custom `cuFFT` Sobolev kernels.
* **Open-Physics Synthetic Engine:** Eliminates the need for NDA-restricted fab data by generating synthetic surface models (Kanaya-Okayama, Carrier Mobility, Space-Charge).
* **Failure Diagnostics:** Automated detection of spatial aliasing in repeating periodic arrays (e.g., DRAM, FinFET).
* **Multi-Modal Extension:** Supports both SEM grayscale and 3-channel RGB optical microscopy alignment.

---

## 📌 Submission Folder Structure

```text
semex/
├── run.py                 # Primary entry script (reads input .npy -> writes output .npy)
├── requirements.txt       # All dependencies with version details
├── README.md              # Setup and execution instructions
├── models/                # Required model weights and supporting binaries
├── CMakeLists.txt         # C++/CUDA CMake Build Configuration
├── sem_drift_cuda.cu      # NVIDIA GPU Engine (cuFFT, Sobolev, Subpixel)
├── sem_drift_core.cpp     # CPU C++20 Fallback Engine (FFTW3 + AVX/SIMD)
├── sem_drift_engine.py    # Physics Synthetic SEM Image Generator
├── sem_analyzer.py        # Benchmark Runner & Diagnostic Engine
├── app.py                 # Gradio / Streamlit Web UI Integration
└── failure_analysis.md    # Periodic Aliasing Diagnostic & Physics Literature Citations
🚀 Execution InstructionsThe solution runs completely offline on an NVIDIA GPU without requiring internet access, API keys, additional model downloads, user interaction, or manual configuration.Bashpython run.py <input-dir> <output-dir>
📐 Mathematical Pipeline & Core Formulas1. 10x Area Binning Downsampling$$T(x,y) = \frac{1}{100} \sum_{dy=0}^{9} \sum_{dx=0}^{9} R(10y + dy, 10x + dx)$$2. Sobolev Edge Gradient Mapping$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I$$$$G_{\text{Sobolev}}(x,y) = \sqrt{G_x(x,y)^2 + G_y(x,y)^2}$$3. Fourier Phase Correlation$$\mathcal{F}_{\text{cross}}(u,v) = \frac{\mathcal{F}\{G_S\} \cdot \mathcal{F}\{G_T\}^*}{N_x \cdot N_y}$$$$\mathcal{C}(x,y) = \mathcal{F}^{-1}\left\{ \mathcal{F}_{\text{cross}}(u,v) \right\}$$4. Sub-Pixel Quadratic Refinement$$\delta_x = \frac{\mathcal{C}(x_p+1, y_p) - \mathcal{C}(x_p-1, y_p)}{2 \left(2\mathcal{C}(x_p, y_p) - \mathcal{C}(x_p-1, y_p) - \mathcal{C}(x_p+1, y_p)\right)}$$
