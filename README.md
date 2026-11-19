# Experiment 5: Neural Network Layers in CUDA

<p align="center">
  <b>Implementation and Performance Analysis of Neural Network Layers Using Custom CUDA Kernels</b>
</p>

---

## 1. Project Overview

This experiment focuses on implementing basic neural network operations using custom CUDA kernels and comparing their execution on CPU and GPU.

A two-layer feed-forward neural network is implemented using three fundamental operations:

- Matrix Multiplication with Bias
- ReLU Activation Function
- Sigmoid Activation Function

The same forward propagation process is executed on both CPU and GPU. Their outputs are compared to verify numerical correctness, and execution times are measured to evaluate GPU performance.

## 2. Aim

To implement and evaluate basic neural network layers using CUDA programming, validate CPU and GPU results, measure execution time, and calculate GPU speedup.

## 3. Technologies and Tools

| Component | Technology |
|---|---|
| Programming Language | C++ / CUDA C++ |
| GPU Programming | NVIDIA CUDA |
| Compiler | NVCC |
| GPU Environment | NVIDIA CUDA-capable GPU |
| Notebook Environment | Jupyter Notebook / Colab, if used |
| GPU Verification | `nvidia-smi` |

**Hardware and software details:** Record the GPU model, CUDA version and compiler version from the actual execution environment.

## 4. Neural Network Architecture

The experiment implements a two-layer feed-forward neural network.

| Layer | Dimensions | Activation |
|---|---|---|
| Input | 256 × 128 | None |
| Hidden | 256 × 64 | ReLU |
| Output | 256 × 10 | Sigmoid |

### Architecture Flow

```text
       INPUT DATA
       X (256 × 128)
             |
             v
     MATRIX MULTIPLICATION
        X × W1 + b1
             |
             v
       ReLU ACTIVATION
        A1 (256 × 64)
             |
             v
     MATRIX MULTIPLICATION
        A1 × W2 + b2
             |
             v
     SIGMOID ACTIVATION
        Y (256 × 10)
             |
             v
        FINAL OUTPUT
```

## 5. Mathematical Operations

### 5.1 First Layer: Matrix Multiplication with Bias

The input matrix is multiplied by the first weight matrix, and the bias is added.

**Equation:**

Z1 = X × W1 + b1

Where:
- X = Input matrix
- W1 = First weight matrix
- b1 = First bias vector
- Z1 = First layer output before activation

### 5.2 ReLU Activation

ReLU introduces non-linearity into the neural network.

**Equation:**

ReLU(x) = max(0, x)

- Negative values become zero.
- Positive values remain unchanged.

Output:

A1 = ReLU(Z1)

### 5.3 Second Layer: Matrix Multiplication with Bias

The hidden layer output is multiplied by the second weight matrix.

**Equation:**

Z2 = A1 × W2 + b2

### 5.4 Sigmoid Activation

Sigmoid converts values into a range between 0 and 1.

**Equation:**

Sigmoid(x) = 1 / (1 + exp(-x))

Final output:

Y = Sigmoid(Z2)

## 6. CUDA Kernels Implemented

Three custom CUDA kernels are implemented.

### Kernel 1: Matrix Multiplication with Bias

**Purpose:** Performs matrix multiplication and adds the bias vector.

**Working:**
1. CUDA threads identify output matrix positions.
2. Each thread calculates an output element.
3. The corresponding bias value is added.
4. The result is stored in GPU memory.

### Kernel 2: ReLU Activation

**Purpose:** Applies the ReLU activation function to every element.

**Working:**
1. Each CUDA thread accesses an input element.
2. It checks whether the value is positive or negative.
3. Negative values are replaced with zero.
4. The result is stored in the output array.

### Kernel 3: Sigmoid Activation

**Purpose:** Applies the sigmoid activation function.

**Working:**
1. Each CUDA thread accesses an input value.
2. It calculates the sigmoid function.
3. The output is stored in GPU memory.

## 7. CPU and GPU Implementation

### CPU Implementation

The CPU implementation performs the neural network operations using standard C++ functions.

It includes:
- CPU Matrix Multiplication with Bias
- CPU ReLU
- CPU Sigmoid
- CPU Forward Propagation

### GPU Implementation

The GPU implementation performs the same operations using custom CUDA kernels.

It includes:
- GPU Memory Allocation
- Host-to-Device Memory Transfer
- CUDA Kernel Execution
- Device-to-Host Memory Transfer
- GPU Memory Deallocation

### CPU vs GPU Comparison

| CPU | GPU |
|---|---|
| Executes operations on CPU cores | Executes operations using many GPU threads |
| Uses host memory | Uses GPU device memory for CUDA operations |
| Uses standard C++ functions | Uses custom CUDA kernels |
| Serves as the reference implementation | Results are compared against CPU output |

## 8. Implementation Workflow

1. Detect and identify the available NVIDIA GPU.
2. Verify the CUDA compiler using NVCC.
3. Define neural network dimensions and constants.
4. Implement the matrix multiplication with bias kernel.
5. Implement ReLU and sigmoid kernels.
6. Implement equivalent CPU functions.
7. Implement GPU forward propagation.
8. Test individual CUDA kernels.
9. Allocate GPU memory and transfer input data.
10. Execute GPU kernels and measure execution time.
11. Compare CPU and GPU outputs.
12. Calculate speedup and analyze performance.

## 9. Correctness Validation

Individual CUDA kernels are tested using known input values and expected outputs.

| Test | Expected Result |
|---|---|
| Matrix Multiplication + Bias | PASSED |
| ReLU Activation | PASSED |
| Sigmoid Activation | PASSED |
| CPU and GPU Output Comparison | MATCH within tolerance |

The maximum absolute difference is calculated as:

Maximum Difference = max(|CPU Output - GPU Output|)

A small difference within the chosen numerical tolerance indicates that the CPU and GPU implementations produce matching results.

## 10. Performance Evaluation

The experiment measures the following performance parameters:

- CPU Average Compute Time
- GPU Kernel-only Execution Time
- GPU End-to-End Execution Time
- Kernel-only Speedup
- End-to-End Speedup
- Maximum Absolute Output Difference

### Speedup Calculation

**Kernel-only Speedup**

CPU Average Time / GPU Kernel-only Time

**End-to-End Speedup**

CPU Average Time / GPU End-to-End Time

End-to-end timing includes GPU-related overheads such as memory transfers and execution setup, according to the program's timing boundaries.

## 11. How to Compile and Run

### Step 1: Verify NVIDIA GPU

```bash
nvidia-smi
```

### Step 2: Verify CUDA Compiler

```bash
nvcc --version
```

### Step 3: Compile the CUDA Program

```bash
nvcc -O2 neural_network.cu -o neural_network
```

### Step 4: Execute the Program

```bash
./neural_network
```

If using a notebook environment, prefix shell commands with `!` inside notebook cells.

## 12. Project Structure

```text
Experiment-5-Neural-Network-Layers-CUDA/
│
├── README.md
├── neural_network.cu
├── Experiment_5_Neural_Network_Layers_CUDA.ipynb
│
└── Screenshots/
    ├── 01_GPU_Detection.png
    ├── 02_CUDA_Compiler.png
    ├── 03_Neural_Network_Architecture.png
    ├── 04_CUDA_Kernels.png
    ├── 05_CPU_Implementation.png
    ├── 06_GPU_Implementation.png
    ├── 07_Kernel_Test_Results.png
    ├── 08_Final_Performance.png
    └── 09_Output_Validation.png
```

## 13. Screenshots and Execution Evidence

### 1. GPU Detection

Shows the GPU model detected by the execution environment.

<img width="970" height="535" alt="01-GPU_Detection_Colab" src="https://github.com/user-attachments/assets/826a2792-6a5d-4f9d-9496-5d4ef6a874d0" />

### 2. CUDA Compiler Verification

Shows the installed CUDA compiler version.

<img width="787" height="210" alt="02-CUDA_compiler_check" src="https://github.com/user-attachments/assets/6aba9026-5697-43c1-bbf6-590b53b742db" />


### 3. CUDA Kernel Implementation

Shows the custom matrix multiplication, ReLU and sigmoid CUDA kernels.

<img width="937" height="642" alt="03-CUDA_Kernels png" src="https://github.com/user-attachments/assets/9617160b-124f-48b3-8d88-c3212a0cbb42" />


### 4. CPU and GPU Implementation

Shows the CPU functions used as the reference implementation.

<img width="897" height="642" alt="04-CPU_GPU_Implementation png" src="https://github.com/user-attachments/assets/1a557ea7-953b-4e4d-a806-eacecee86d66" />


### 5. Individual Kernel Test Results

Shows the correctness validation of the three CUDA kernels.

<img width="870" height="645" alt="05-Kernel_test_results png" src="https://github.com/user-attachments/assets/7f3aca88-c182-4de0-99fe-5dc83bbeff62" />


### 6. Final Performance Results

Shows CPU and GPU execution times and calculated speedups.

<img width="997" height="625" alt="06-Final_Output png" src="https://github.com/user-attachments/assets/966873a3-4e3e-4df7-a1ff-62cd32f29801" />




## 14. Important Note

This experiment implements forward propagation (inference) only.

It does not implement:
- Model Training
- Loss Calculation
- Backpropagation
- Weight Optimization

The purpose is to understand CUDA kernel implementation, parallel execution and performance comparison between CPU and GPU.

## 15. Learning Outcomes

After completing this experiment, the following concepts are understood:

1. Basic neural network forward propagation.
2. Matrix multiplication and bias addition.
3. ReLU and sigmoid activation functions.
4. CUDA kernels, blocks and threads.
5. GPU memory allocation and data transfers.
6. CPU and GPU numerical validation.
7. Execution time measurement and speedup calculation.

## 16. Conclusion

In this experiment, three fundamental neural network operations were implemented using custom CUDA kernels.

The same forward propagation process was executed on CPU and GPU. Individual kernel tests were performed to verify correctness, and CPU and GPU outputs were compared using numerical tolerance.

Execution times were measured to understand the performance differences between CPU and GPU implementations.

Thus, this experiment demonstrates how CUDA can be used to accelerate neural network computations through parallel processing.

---

**Experiment:** 5 – Neural Network Layers in CUDA  
**Subject:** Cloud / GPU Computing Laboratory  
**Programming Language:** C++ / CUDA C++  
**Platform:** Record actual execution environment  
**GPU:** Record actual GPU model
