# PyTorch Internals Study Plan

A comprehensive study guide for understanding PyTorch's internal architecture and implementation. Check off each component as you complete your study.

---

## Table of Contents

1. [Foundation Layer](#1-foundation-layer)
2. [Core Tensor System (ATen)](#2-core-tensor-system-aten)
3. [Dispatcher Mechanism](#3-dispatcher-mechanism)
4. [Autograd System](#4-autograd-system)
5. [Memory Management](#5-memory-management)
6. [CUDA Caching Allocator (Deep Dive)](#6-cuda-caching-allocator-deep-dive)
7. [CUDA Integration](#7-cuda-integration)
8. [Python/C++ Bindings](#8-pythonc-bindings)
9. [Operator Registration](#9-operator-registration)
10. [Neural Network Modules](#10-neural-network-modules)
11. [Data Loading Pipeline](#11-data-loading-pipeline)
12. [Serialization System](#12-serialization-system)
13. [Distributed Training](#13-distributed-training)
14. [FX Graph System](#14-fx-graph-system)
15. [torch.compile and Dynamo](#15-torchcompile-and-dynamo)
16. [Mixed Precision (AMP)](#16-mixed-precision-amp)
17. [Quantization](#17-quantization)
18. [Export and ONNX](#18-export-and-onnx)
19. [Profiling and Debugging](#19-profiling-and-debugging)

---

## 1. Foundation Layer

The c10 library provides core abstractions used throughout PyTorch.

### Key Files
- [ ] `c10/core/Device.h` - Device abstraction (CPU, CUDA, etc.)
- [ ] `c10/core/DeviceType.h` - Device type enumeration
- [ ] `c10/core/Stream.h` - Stream abstraction for async execution
- [ ] `c10/core/StreamGuard.h` - RAII guard for stream management
- [ ] `c10/core/ScalarType.h` - Data type definitions (float, int, etc.)
- [ ] `c10/util/intrusive_ptr.h` - Reference-counted smart pointer
- [ ] `c10/util/Optional.h` - Optional value wrapper
- [ ] `c10/util/ArrayRef.h` - Non-owning array reference
- [ ] `c10/util/SmallVector.h` - Stack-allocated small vector

### Core Concepts
- [ ] Understand Device and DeviceType abstraction
- [ ] Learn Stream and StreamGuard for async execution
- [ ] Study ScalarType and dtype system
- [ ] Master intrusive_ptr reference counting
- [ ] Understand c10::optional and c10::ArrayRef utilities

---

## 2. Core Tensor System (ATen)

ATen (A Tensor) is the fundamental tensor library in PyTorch.

### Key Files
- [ ] `c10/core/TensorImpl.h` - Core tensor implementation class
- [ ] `c10/core/TensorImpl.cpp` - TensorImpl implementation
- [ ] `c10/core/Storage.h` - Raw data storage abstraction
- [ ] `c10/core/StorageImpl.h` - Storage implementation
- [ ] `aten/src/ATen/core/Tensor.h` - Main Tensor class (handle)
- [ ] `aten/src/ATen/TensorUtils.h` - Tensor utility functions
- [ ] `aten/src/ATen/core/TensorBody.h` - Generated tensor methods

### Native Operations
- [ ] `aten/src/ATen/native/` - Directory of native CPU operations
- [ ] `aten/src/ATen/native/TensorShape.cpp` - reshape, view, squeeze, unsqueeze
- [ ] `aten/src/ATen/native/Indexing.cpp` - Advanced indexing operations
- [ ] `aten/src/ATen/native/ReduceOps.cpp` - sum, mean, max, min
- [ ] `aten/src/ATen/native/UnaryOps.cpp` - abs, sin, cos, exp
- [ ] `aten/src/ATen/native/BinaryOps.cpp` - add, sub, mul, div
- [ ] `aten/src/ATen/native/LinearAlgebra.cpp` - matmul, mm, bmm

### Core Concepts
- [ ] Understand TensorImpl structure (sizes, strides, storage_offset)
- [ ] Learn Storage and data ownership model
- [ ] Study view vs copy semantics
- [ ] Understand stride-based memory layout
- [ ] Learn contiguous vs non-contiguous tensors
- [ ] Study metadata-only operations (view, transpose, permute)

### Study Order
1. Start with `c10/core/TensorImpl.h` to understand tensor internals
2. Study `c10/core/Storage.h` for data storage
3. Read `aten/src/ATen/core/Tensor.h` for the user-facing API
4. Explore native operations in `aten/src/ATen/native/`

---

## 3. Dispatcher Mechanism

The dispatcher routes operator calls to the appropriate kernel implementation.

### Key Files
- [ ] `c10/core/DispatchKey.h` - Dispatch key definitions
- [ ] `c10/core/DispatchKeySet.h` - Set of dispatch keys
- [ ] `aten/src/ATen/core/dispatch/Dispatcher.h` - Main dispatcher class
- [ ] `aten/src/ATen/core/dispatch/Dispatcher.cpp` - Dispatcher implementation
- [ ] `aten/src/ATen/core/dispatch/OperatorEntry.h` - Per-operator kernel table
- [ ] `aten/src/ATen/core/dispatch/OperatorEntry.cpp` - OperatorEntry implementation
- [ ] `aten/src/ATen/core/boxing/KernelFunction.h` - Kernel wrapper

### Dispatch Keys (Priority Order)
- [ ] Understand `Undefined` - No dispatch key
- [ ] Understand `FuncTorchBatched` - vmap batching
- [ ] Understand `Functionalize` - Functionalization pass
- [ ] Understand `AutogradCPU/CUDA` - Autograd handling
- [ ] Understand `AutocastCPU/CUDA` - Automatic mixed precision
- [ ] Understand `CPU/CUDA/MPS` - Backend-specific kernels
- [ ] Understand `CompositeExplicitAutograd` - Cross-backend ops with explicit autograd
- [ ] Understand `CompositeImplicitAutograd` - Cross-backend ops with implicit autograd

### Core Concepts
- [ ] Learn dispatch key precedence and ordering
- [ ] Understand kernel registration and lookup
- [ ] Study boxed vs unboxed kernels
- [ ] Learn fallback kernel mechanism
- [ ] Understand redispatch and dispatch key exclusion
- [ ] Study operator schema and type system

### Study Order
1. Start with `c10/core/DispatchKey.h` to understand key types
2. Read `aten/src/ATen/core/dispatch/Dispatcher.h` for the main API
3. Study `OperatorEntry.h` for per-operator kernel management
4. Trace through a dispatch call to understand the flow

---

## 4. Autograd System

PyTorch's automatic differentiation engine.

### Key Files - C++ Core
- [ ] `torch/csrc/autograd/variable.h` - Variable (tensor with grad) definition
- [ ] `torch/csrc/autograd/function.h` - Node base class for autograd graph
- [ ] `torch/csrc/autograd/edge.h` - Edge connecting nodes in the graph
- [ ] `torch/csrc/autograd/engine.h` - Backward engine declaration
- [ ] `torch/csrc/autograd/engine.cpp` - Backward engine implementation
- [ ] `torch/csrc/autograd/grad_mode.h` - Gradient computation mode
- [ ] `torch/csrc/autograd/saved_variable.h` - Saved tensors for backward

### Key Files - Python Layer
- [ ] `torch/autograd/__init__.py` - Public autograd API
- [ ] `torch/autograd/grad_mode.py` - no_grad, enable_grad context managers
- [ ] `torch/autograd/function.py` - Custom Function base class
- [ ] `torch/autograd/gradcheck.py` - Gradient checking utilities
- [ ] `torch/autograd/profiler.py` - Autograd profiling

### Generated Derivatives
- [ ] `tools/autograd/derivatives.yaml` - Derivative definitions for operators
- [ ] `tools/autograd/gen_autograd.py` - Derivative code generation

### Core Concepts
- [ ] Understand Node (autograd Function) and Edge structure
- [ ] Learn computational graph construction during forward
- [ ] Study backward engine execution (topological sort, worker threads)
- [ ] Understand gradient accumulation and hooks
- [ ] Learn saved tensor management and memory implications
- [ ] Study grad_mode (no_grad, inference_mode, enable_grad)
- [ ] Understand custom Function (forward/backward/ctx)

### Study Order
1. Start with `torch/csrc/autograd/function.h` (Node class)
2. Read `torch/csrc/autograd/edge.h` for graph structure
3. Study `torch/csrc/autograd/engine.cpp` for backward execution
4. Read `tools/autograd/derivatives.yaml` for derivative specifications
5. Explore `torch/autograd/function.py` for custom Functions

---

## 5. Memory Management

Memory allocation and management across devices.

### Key Files - Core Allocator
- [ ] `c10/core/Allocator.h` - Base allocator interface
- [ ] `c10/core/Allocator.cpp` - Allocator implementation
- [ ] `c10/core/CPUAllocator.h` - CPU memory allocator
- [ ] `c10/core/CPUAllocator.cpp` - CPU allocator implementation

### Key Files - CUDA Allocator
- [ ] `c10/cuda/CUDACachingAllocator.h` - CUDA caching allocator interface
- [ ] `c10/cuda/CUDACachingAllocator.cpp` - CUDA caching allocator implementation
- [ ] `c10/cuda/CUDAAllocatorConfig.h` - Allocator configuration
- [ ] `torch/cuda/memory.py` - Python memory management API

### Core Concepts
- [ ] Understand Allocator interface (allocate, deallocate)
- [ ] Learn DataPtr and its ownership semantics
- [ ] Study CUDA caching allocator block management
- [ ] Understand memory pools and fragmentation
- [ ] Learn pinned memory for async CPU-GPU transfers
- [ ] Study memory snapshots and debugging tools
- [ ] Understand PYTORCH_CUDA_ALLOC_CONF environment variable

### Memory Debugging
- [ ] `torch.cuda.memory_stats()` - Memory statistics
- [ ] `torch.cuda.memory_snapshot()` - Memory block snapshot
- [ ] `torch.cuda.memory_summary()` - Human-readable summary
- [ ] `torch.cuda.reset_peak_memory_stats()` - Reset peak tracking

### Study Order
1. Start with `c10/core/Allocator.h` for the interface
2. Study `c10/cuda/CUDACachingAllocator.cpp` for CUDA implementation
3. Read `torch/cuda/memory.py` for Python API
4. Experiment with memory debugging tools

---

## 6. CUDA Caching Allocator (Deep Dive)

A comprehensive deep dive into PyTorch's GPU memory management system.

### Key Files
- [ ] `c10/cuda/CUDACachingAllocator.cpp` - Main allocator implementation (3500+ lines)
- [ ] `c10/cuda/CUDACachingAllocator.h` - Public API and data structures
- [ ] `c10/cuda/CUDAAllocatorConfig.h` - Configuration parser
- [ ] `c10/cuda/CUDAAllocatorConfig.cpp` - Configuration implementation
- [ ] `c10/core/CachingDeviceAllocator.h` - DeviceStats structure definition
- [ ] `torch/cuda/memory.py` - Python memory management API

### Core Data Structures

#### Block Structure
- [ ] Understand `Block` struct (lines 193-256 in CUDACachingAllocator.cpp)
  - `device` - GPU device ID
  - `stream` - Allocation stream
  - `stream_uses` - Other streams that used this block
  - `size` - Block size in bytes
  - `requested_size` - Original requested size
  - `pool` - Owner BlockPool
  - `ptr` - Memory address
  - `allocated` - In-use flag
  - `mapped` - Backed by physical pages (expandable segments)
  - `prev/next` - Linked list for splits
  - `event_count` - Outstanding CUDA events
  - `expandable_segment_` - For expandable segments

#### BlockPool Structure
- [ ] Understand `BlockPool` struct (lines 168-189)
  - `blocks` - Free blocks sorted by size (best-fit)
  - `unmapped` - Unmapped blocks for expandable segments
  - `is_small` - Small pool flag (<=1MB allocations)
  - `owner_PrivatePool` - CUDA graph owner

#### ExpandableSegment
- [ ] Understand `ExpandableSegment` class (lines 367-777)
  - Virtual memory using CUDA driver API
  - `map()` - Map physical to virtual memory
  - `unmap()` - Release physical pages
  - Page granularity: 2MB (small), 20MB (large)

#### DeviceCachingAllocator
- [ ] Understand `DeviceCachingAllocator` class (lines 1144-3409)
  - `mutex` - Protects all state
  - `stats` - DeviceStats for metrics
  - `large_blocks` - Cached blocks > 1MB
  - `small_blocks` - Cached blocks <= 1MB
  - `active_blocks` - In-use allocations
  - `expandable_segments_` - Dynamic segments
  - `graph_pools` - Private pools for CUDA graphs

### Size Constants and Thresholds
- [ ] `kMinBlockSize` = 512 bytes - Minimum allocation unit
- [ ] `kSmallSize` = 1 MB - Small/large pool threshold
- [ ] `kSmallBuffer` = 2 MB - Segment size for small allocations
- [ ] `kMinLargeAlloc` = 10 MB - Threshold for kLargeBuffer
- [ ] `kLargeBuffer` = 20 MB - Segment size for 1-10MB allocations
- [ ] `kRoundLarge` = 2 MB - Rounding granule for large allocations

### Allocation Algorithm (malloc function, line 1312)
- [ ] Step 1: Gather context (stack trace if enabled)
- [ ] Step 2: Acquire lock (`std::recursive_mutex`)
- [ ] Step 3: Process completed CUDA events
- [ ] Step 4: Round size (`round_size()` at line 2202)
- [ ] Step 5: Select pool (`get_pool()` - small vs large)
- [ ] Step 6: Calculate allocation size (`get_allocation_size()` at line 2702)
- [ ] Step 7: Attempt block reuse (`get_free_block()` - best-fit search)
- [ ] Step 8: If no block found:
  - [ ] Try garbage collection
  - [ ] Try `alloc_block()` (cudaMalloc)
  - [ ] Release available cached blocks and retry
  - [ ] Release all non-split blocks and retry
- [ ] Step 9: Handle OOM (detailed error message)
- [ ] Step 10: Split remainder if applicable (`should_split()`)
- [ ] Step 11: Finalize allocation (`alloc_found_block()`)

### Deallocation Algorithm (free function, line 1625)
- [ ] Step 1: Gather context
- [ ] Step 2: Acquire lock
- [ ] Step 3: Mark block as unallocated
- [ ] Step 4: Record trace
- [ ] Step 5: Update statistics
- [ ] Step 6: Handle stream synchronization
  - [ ] If no cross-stream uses: call `free_block()` directly
  - [ ] If cross-stream uses: insert CUDA events for synchronization
- [ ] Step 7: `free_block()` implementation (lines 2548-2622)
  - [ ] Try merge with neighbors
  - [ ] Remove from active_blocks
  - [ ] Insert into pool
  - [ ] Update fragmentation stats

### Memory Pool Management
- [ ] Small Pool (size <= 1 MB)
  - Allocation size: Always 2 MB
  - Splitting threshold: 512 bytes
  - Packs many small allocations into segments

- [ ] Large Pool (size > 1 MB)
  - 1MB < size < 10MB: 20 MB segments
  - size >= 10MB: Round up to 2 MB granules
  - Splitting: Only if remaining > 1MB

### Stream-Ordered Memory Allocation
- [ ] Understand stream association per block
- [ ] Learn cross-stream synchronization with CUDA events
- [ ] Study `recordStream()` API (lines 1726-1737)
- [ ] Understand event pooling (`EventPool`)
- [ ] Learn deferred events during CUDA graph capture

### Expandable Segments
- [ ] Understand motivation (variable batch size fragmentation)
- [ ] Learn `cuMemAddressReserve` for VA space (1.125x GPU memory)
- [ ] Study `cuMemCreate` for physical allocation
- [ ] Understand `cuMemMap` for VA-to-physical mapping
- [ ] Learn dynamic page mapping/unmapping
- [ ] Study IPC support (POSIX file descriptors, fabric handles)
- [ ] Understand limitations with multiprocessing data loaders

### Configuration Options (PYTORCH_CUDA_ALLOC_CONF)
- [ ] `expandable_segments:{True|False}` - Enable dynamic expansion
- [ ] `garbage_collection_threshold:{0.0-1.0}` - GC trigger threshold
- [ ] `max_split_size:{bytes}` - Blocks >= this cannot be split
- [ ] `roundup_power2_divisions:{count}` - Power-of-2 size rounding
- [ ] `release_lock_on_cudamalloc:{True|False}` - Reduce lock contention
- [ ] `pinned_use_cuda_host_register:{True|False}` - Pinned memory method
- [ ] `pinned_num_register_threads:{count}` - Thread pool size

### Memory Statistics and Debugging
- [ ] `torch.cuda.memory_stats(device)` - Get statistics
- [ ] `torch.cuda.memory_summary(device)` - Human-readable summary
- [ ] `torch.cuda.memory_snapshot()` - Block-level snapshot
- [ ] `torch.cuda.memory_allocated(device)` - Active allocations
- [ ] `torch.cuda.memory_reserved(device)` - Total reserved (allocated + cached)
- [ ] `torch.cuda.reset_peak_memory_stats()` - Reset peak tracking
- [ ] `torch.cuda.empty_cache()` - Release cached blocks

### DeviceStats Structure
- [ ] `allocated_bytes` - Currently used by tensors
- [ ] `reserved_bytes` - Allocated + cached in pools
- [ ] `active_bytes` - In-use bytes
- [ ] `inactive_split_bytes` - Fragmentation from splits
- [ ] `requested_bytes` - Unrounded user requests
- [ ] `num_alloc_retries` - Cache flush attempts
- [ ] `num_ooms` - Out-of-memory events
- [ ] `num_device_alloc` - cudaMalloc calls
- [ ] `num_device_free` - cudaFree calls

### Garbage Collection
- [ ] Understand GC trigger conditions
- [ ] Learn age-based block selection (`gc_count()`)
- [ ] Study `garbage_collect_cached_blocks()` (line 2780)
- [ ] Understand memory reclamation hierarchy:
  1. Reuse from pool (free)
  2. Trigger user callbacks
  3. New cudaMalloc
  4. Release oversized blocks
  5. Full cache flush
  6. OOM pools (CUDA graphs)

### Private Pools and CUDA Graphs
- [ ] Understand `PrivatePool` structure (lines 917-949)
- [ ] Learn graph capture memory isolation
- [ ] Study `beginAllocateToPool()` / `endAllocateToPool()`
- [ ] Understand pool lifecycle (capture → replay → release)
- [ ] Learn multi-graph pool sharing
- [ ] Study checkpointing for FSDP + CUDA graphs

### Integration with cudaMalloc/cudaFree
- [ ] Understand caching benefits (100-1000x faster allocation)
- [ ] Learn allocation path through `alloc_block()` → `cudaMallocMaybeCapturing()`
- [ ] Study deallocation path through `release_block()` → `cudaFree()`
- [ ] Understand GPU trace hooks for profiling

### OOM Debugging
- [ ] Understand OOM error message breakdown
- [ ] Learn to interpret "reserved but unallocated" memory
- [ ] Study fragmentation diagnosis
- [ ] Understand when to use `expandable_segments:True`
- [ ] Learn when to call `torch.cuda.empty_cache()`

### Study Order
1. Start with `c10/cuda/CUDACachingAllocator.h` for API overview
2. Study Block and BlockPool structures
3. Trace through `malloc()` function step by step
4. Trace through `free()` and `free_block()`
5. Study expandable segments in detail
6. Learn configuration options
7. Explore memory debugging tools
8. Study CUDA graph integration

---

## 7. CUDA Integration

PyTorch's CUDA backend implementation.

### Key Files - Core CUDA
- [ ] `c10/cuda/CUDAStream.h` - CUDA stream wrapper
- [ ] `c10/cuda/CUDAStream.cpp` - Stream implementation
- [ ] `c10/cuda/CUDAGuard.h` - Device/stream guards
- [ ] `c10/cuda/CUDAFunctions.h` - CUDA utility functions
- [ ] `c10/cuda/CUDAException.h` - CUDA error handling

### Key Files - ATen CUDA
- [ ] `aten/src/ATen/cuda/CUDAContext.h` - CUDA context management
- [ ] `aten/src/ATen/cuda/CUDAContext.cpp` - Context implementation
- [ ] `aten/src/ATen/cuda/CUDABlas.h` - cuBLAS wrapper
- [ ] `aten/src/ATen/cuda/CUDABlas.cpp` - cuBLAS implementation
- [ ] `aten/src/ATen/cudnn/Descriptors.h` - cuDNN descriptor wrappers

### Key Files - CUDA Kernels
- [ ] `aten/src/ATen/native/cuda/` - Native CUDA operations
- [ ] `aten/src/ATen/native/cuda/Reduce.cuh` - Reduction kernels
- [ ] `aten/src/ATen/native/cuda/Loops.cuh` - Element-wise kernel helpers
- [ ] `aten/src/ATen/native/cuda/KernelUtils.cuh` - Kernel utilities

### Core Concepts
- [ ] Understand CUDAStream and multi-stream execution
- [ ] Learn CUDAGuard for device/stream context management
- [ ] Study cuBLAS integration for linear algebra
- [ ] Understand cuDNN integration for convolutions
- [ ] Learn kernel launch patterns (grid, block sizing)
- [ ] Study CUDA event synchronization
- [ ] Understand GPU memory hierarchy and coalescing

### Study Order
1. Start with `c10/cuda/CUDAStream.h` for stream abstraction
2. Read `c10/cuda/CUDAGuard.h` for context management
3. Study `aten/src/ATen/cuda/CUDAContext.h` for global context
4. Explore kernels in `aten/src/ATen/native/cuda/`

---

## 8. Python/C++ Bindings

How Python interfaces with C++ implementation.

### Key Files - Core Bindings
- [ ] `torch/csrc/Module.cpp` - Main torch._C module initialization
- [ ] `torch/csrc/Module.h` - Module declarations
- [ ] `torch/csrc/utils/python_arg_parser.h` - Argument parsing
- [ ] `torch/csrc/utils/python_arg_parser.cpp` - Parser implementation
- [ ] `torch/csrc/utils/pybind.h` - pybind11 utilities

### Key Files - Tensor Bindings
- [ ] `torch/csrc/autograd/python_variable.h` - Python tensor wrapper
- [ ] `torch/csrc/autograd/python_variable.cpp` - Tensor binding implementation
- [ ] `torch/csrc/autograd/python_function.h` - Python autograd Function
- [ ] `torch/csrc/autograd/python_function.cpp` - Function implementation

### Key Files - Generated Bindings
- [ ] `tools/autograd/gen_python_functions.py` - Python function generation
- [ ] `torch/csrc/autograd/generated/` - Generated binding code

### Core Concepts
- [ ] Understand THPVariable (Python tensor wrapper)
- [ ] Learn PythonArgParser for flexible argument handling
- [ ] Study method binding patterns (static, instance, module)
- [ ] Understand GIL (Global Interpreter Lock) management
- [ ] Learn error translation (C++ exceptions to Python)
- [ ] Study tensor/storage object lifecycle
- [ ] Understand generated vs hand-written bindings

### Study Order
1. Start with `torch/csrc/Module.cpp` for initialization
2. Read `torch/csrc/autograd/python_variable.cpp` for tensor bindings
3. Study `torch/csrc/utils/python_arg_parser.h` for arg parsing
4. Explore generated code in `torch/csrc/autograd/generated/`

---

## 9. Operator Registration

How operators are defined and registered in PyTorch.

### Key Files - Native Functions
- [ ] `aten/src/ATen/native/native_functions.yaml` - Operator definitions
- [ ] `tools/autograd/derivatives.yaml` - Derivative formulas
- [ ] `aten/src/ATen/native/README.md` - Native functions documentation

### Key Files - Registration System
- [ ] `aten/src/ATen/core/op_registration/op_registration.h` - Registration API
- [ ] `torch/library.h` - TORCH_LIBRARY macro
- [ ] `aten/src/ATen/core/library.cpp` - Library implementation
- [ ] `c10/core/OperatorName.h` - Operator naming

### Key Files - Python Registration
- [ ] `torch/library.py` - Python registration API
- [ ] `torch/_ops.py` - Operator lookup and calling

### Core Concepts
- [ ] Understand native_functions.yaml schema
- [ ] Learn structured kernels vs legacy kernels
- [ ] Study TORCH_LIBRARY and TORCH_LIBRARY_IMPL macros
- [ ] Understand namespace and operator naming conventions
- [ ] Learn torch.library Python API for custom ops
- [ ] Study meta functions for shape inference
- [ ] Understand abstract implementations for new backends

### native_functions.yaml Schema
- [ ] `func` - Function signature
- [ ] `variants` - function, method, or both
- [ ] `dispatch` - Backend-specific implementations
- [ ] `structured` - Use structured kernel
- [ ] `autogen` - Auto-generate certain variants

### Study Order
1. Read `aten/src/ATen/native/README.md` for overview
2. Study `native_functions.yaml` entries
3. Understand `derivatives.yaml` format
4. Learn TORCH_LIBRARY macros
5. Explore `torch/library.py` for Python registration

---

## 10. Neural Network Modules

The nn module system for building neural networks.

### Key Files - Python Module System
- [ ] `torch/nn/modules/module.py` - Base Module class
- [ ] `torch/nn/modules/container.py` - Sequential, ModuleList, ModuleDict
- [ ] `torch/nn/modules/linear.py` - Linear layer
- [ ] `torch/nn/modules/conv.py` - Convolution layers
- [ ] `torch/nn/modules/activation.py` - Activation functions
- [ ] `torch/nn/modules/batchnorm.py` - Batch normalization
- [ ] `torch/nn/modules/dropout.py` - Dropout layers
- [ ] `torch/nn/modules/loss.py` - Loss functions
- [ ] `torch/nn/modules/rnn.py` - RNN, LSTM, GRU

### Key Files - Parameters and Buffers
- [ ] `torch/nn/parameter.py` - Parameter class
- [ ] `torch/nn/utils/parametrize.py` - Parametrization API

### Key Files - Functional API
- [ ] `torch/nn/functional.py` - Stateless operations
- [ ] `torch/_C/_nn.pyi` - C++ functional stubs

### Key Files - Initialization
- [ ] `torch/nn/init.py` - Weight initialization methods

### Core Concepts
- [ ] Understand Module base class structure
- [ ] Learn Parameter vs Buffer distinction
- [ ] Study module registration (_modules, _parameters, _buffers)
- [ ] Understand hooks (forward, backward, state_dict)
- [ ] Learn train/eval mode and its effects
- [ ] Study state_dict and load_state_dict
- [ ] Understand named_modules, named_parameters iteration
- [ ] Learn module.to(device) and dtype conversion

### Module Hooks
- [ ] `register_forward_pre_hook` - Before forward
- [ ] `register_forward_hook` - After forward
- [ ] `register_backward_hook` - During backward (deprecated)
- [ ] `register_full_backward_hook` - Full backward hook
- [ ] `_register_state_dict_hook` - State dict customization

### Study Order
1. Start with `torch/nn/modules/module.py` - the foundation
2. Study `torch/nn/parameter.py` for Parameter
3. Read a simple module like `linear.py`
4. Explore `container.py` for Sequential, ModuleList
5. Study functional API in `torch/nn/functional.py`

---

## 11. Data Loading Pipeline

Efficient data loading for training.

### Key Files
- [ ] `torch/utils/data/dataloader.py` - DataLoader implementation
- [ ] `torch/utils/data/dataset.py` - Dataset base classes
- [ ] `torch/utils/data/sampler.py` - Sampling strategies
- [ ] `torch/utils/data/_utils/collate.py` - Batch collation
- [ ] `torch/utils/data/_utils/fetch.py` - Data fetching
- [ ] `torch/utils/data/_utils/worker.py` - Worker process management
- [ ] `torch/utils/data/_utils/pin_memory.py` - Pinned memory utilities

### Dataset Types
- [ ] `Dataset` - Map-style dataset (index-based)
- [ ] `IterableDataset` - Iterable-style dataset (stream-based)
- [ ] `TensorDataset` - Wrap tensors as dataset
- [ ] `ConcatDataset` - Concatenate datasets
- [ ] `Subset` - Subset of a dataset

### Sampler Types
- [ ] `SequentialSampler` - Sequential indices
- [ ] `RandomSampler` - Random indices
- [ ] `SubsetRandomSampler` - Random from subset
- [ ] `BatchSampler` - Wrap sampler to yield batches
- [ ] `DistributedSampler` - For distributed training

### Core Concepts
- [ ] Understand map-style vs iterable-style datasets
- [ ] Learn multi-process data loading architecture
- [ ] Study worker process lifecycle and communication
- [ ] Understand pin_memory for faster GPU transfer
- [ ] Learn custom collate_fn for complex batching
- [ ] Study DistributedSampler for multi-GPU training
- [ ] Understand prefetching and data pipeline optimization

### Study Order
1. Start with `dataset.py` for Dataset base class
2. Read `sampler.py` for sampling strategies
3. Study `dataloader.py` for DataLoader implementation
4. Explore `_utils/worker.py` for multi-process loading

---

## 12. Serialization System

Saving and loading models and tensors.

### Key Files
- [ ] `torch/serialization.py` - Main save/load implementation
- [ ] `torch/_weights_only_unpickler.py` - Safe unpickling
- [ ] `torch/storage.py` - Storage serialization
- [ ] `torch/nn/modules/module.py` - state_dict methods

### Core Concepts
- [ ] Understand pickle-based serialization
- [ ] Learn state_dict structure and conventions
- [ ] Study weights_only mode for security
- [ ] Understand storage sharing and views
- [ ] Learn mmap loading for large models
- [ ] Study checkpoint sharding for huge models
- [ ] Understand device mapping during load

### Key Functions
- [ ] `torch.save(obj, f)` - Save object to file
- [ ] `torch.load(f)` - Load object from file
- [ ] `module.state_dict()` - Get model state
- [ ] `module.load_state_dict(sd)` - Load model state
- [ ] `torch.save` with `_use_new_zipfile_serialization`

### Security Considerations
- [ ] Understand pickle security risks
- [ ] Learn weights_only=True for safe loading
- [ ] Study allowed classes in safe mode
- [ ] Understand torch.serialization.add_safe_globals

### Study Order
1. Start with `torch/serialization.py`
2. Study state_dict in `torch/nn/modules/module.py`
3. Read `torch/_weights_only_unpickler.py` for safety
4. Understand storage serialization

---

## 13. Distributed Training

Multi-GPU and multi-node training infrastructure.

### Key Files - Core Distributed
- [ ] `torch/distributed/distributed_c10d.py` - Core distributed API
- [ ] `torch/distributed/c10d_logger.py` - Distributed logging
- [ ] `torch/csrc/distributed/c10d/ProcessGroup.hpp` - ProcessGroup base
- [ ] `torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp` - NCCL backend
- [ ] `torch/csrc/distributed/c10d/ProcessGroupGloo.hpp` - Gloo backend

### Key Files - DDP (DistributedDataParallel)
- [ ] `torch/nn/parallel/distributed.py` - DDP implementation
- [ ] `torch/csrc/distributed/c10d/reducer.hpp` - Gradient reducer
- [ ] `torch/csrc/distributed/c10d/reducer.cpp` - Reducer implementation

### Key Files - FSDP (FullyShardedDataParallel)
- [ ] `torch/distributed/fsdp/fully_sharded_data_parallel.py` - FSDP main
- [ ] `torch/distributed/fsdp/_runtime_utils.py` - Runtime utilities
- [ ] `torch/distributed/fsdp/_state_dict_utils.py` - State dict handling

### Key Files - RPC
- [ ] `torch/distributed/rpc/__init__.py` - RPC framework
- [ ] `torch/distributed/rpc/api.py` - RPC API

### Collective Operations
- [ ] `all_reduce` - Reduce and broadcast result
- [ ] `all_gather` - Gather tensors from all ranks
- [ ] `broadcast` - Broadcast from one rank
- [ ] `reduce_scatter` - Reduce and scatter
- [ ] `barrier` - Synchronization point

### Core Concepts
- [ ] Understand ProcessGroup abstraction
- [ ] Learn NCCL vs Gloo backends
- [ ] Study DDP gradient synchronization
- [ ] Understand bucket-based gradient reduction
- [ ] Learn FSDP sharding strategies
- [ ] Study mixed precision with distributed
- [ ] Understand checkpoint saving in distributed

### Study Order
1. Start with `torch/distributed/distributed_c10d.py`
2. Study `torch/nn/parallel/distributed.py` for DDP
3. Read reducer implementation for gradient sync
4. Explore FSDP for memory-efficient training

---

## 14. FX Graph System

PyTorch's intermediate representation for graph capture and transformation.

### Key Files
- [ ] `torch/fx/__init__.py` - FX public API
- [ ] `torch/fx/graph.py` - Graph data structure
- [ ] `torch/fx/node.py` - Node in the graph
- [ ] `torch/fx/graph_module.py` - GraphModule (executable graph)
- [ ] `torch/fx/symbolic_trace.py` - Symbolic tracing
- [ ] `torch/fx/interpreter.py` - Graph interpretation
- [ ] `torch/fx/proxy.py` - Proxy for tracing
- [ ] `torch/fx/passes/` - Graph transformation passes

### Node Operations
- [ ] `placeholder` - Function inputs
- [ ] `get_attr` - Access module attributes
- [ ] `call_function` - Call a free function
- [ ] `call_method` - Call a method on a value
- [ ] `call_module` - Call a submodule
- [ ] `output` - Function output

### Core Concepts
- [ ] Understand Graph and Node structure
- [ ] Learn symbolic tracing vs concrete execution
- [ ] Study Proxy objects for trace capture
- [ ] Understand GraphModule as nn.Module wrapper
- [ ] Learn Interpreter for graph execution
- [ ] Study graph transformations and passes
- [ ] Understand Tracer customization

### Common Transformations
- [ ] Shape propagation
- [ ] Operator fusion
- [ ] Dead code elimination
- [ ] Subgraph matching and replacement

### Study Order
1. Start with `torch/fx/graph.py` and `node.py`
2. Read `torch/fx/symbolic_trace.py` for tracing
3. Study `torch/fx/graph_module.py` for execution
4. Explore `torch/fx/interpreter.py`
5. Look at passes in `torch/fx/passes/`

---

## 15. torch.compile and Dynamo

PyTorch's JIT compiler and graph capture system.

### Key Files - Dynamo Core
- [ ] `torch/_dynamo/__init__.py` - Dynamo entry point
- [ ] `torch/_dynamo/eval_frame.py` - Frame evaluation hook
- [ ] `torch/_dynamo/symbolic_convert.py` - Bytecode to FX conversion
- [ ] `torch/_dynamo/output_graph.py` - Graph output handling
- [ ] `torch/_dynamo/guards.py` - Guard generation
- [ ] `torch/_dynamo/variables/` - Variable tracking classes
- [ ] `torch/_dynamo/bytecode_transformation.py` - Bytecode manipulation

### Key Files - Inductor Backend
- [ ] `torch/_inductor/__init__.py` - Inductor entry point
- [ ] `torch/_inductor/compile_fx.py` - FX compilation
- [ ] `torch/_inductor/graph.py` - Inductor IR graph
- [ ] `torch/_inductor/lowering.py` - Op lowering
- [ ] `torch/_inductor/codegen/` - Code generation
- [ ] `torch/_inductor/scheduler.py` - Fusion scheduling

### Key Files - AOT Autograd
- [ ] `torch/_functorch/aot_autograd.py` - Ahead-of-time autograd
- [ ] `torch/_functorch/partitioners.py` - Graph partitioning

### Core Concepts
- [ ] Understand frame evaluation hooking (PEP 523)
- [ ] Learn bytecode capture and transformation
- [ ] Study guard system for validity checks
- [ ] Understand graph breaks and their causes
- [ ] Learn FX graph generation from bytecode
- [ ] Study AOT Autograd for backward graph capture
- [ ] Understand Inductor code generation (Triton, C++)
- [ ] Learn fusion and scheduling optimization

### Compilation Modes
- [ ] `torch.compile(model)` - Default compilation
- [ ] `mode="default"` - Balanced optimization
- [ ] `mode="reduce-overhead"` - Lower dispatch overhead
- [ ] `mode="max-autotune"` - Maximum optimization
- [ ] `fullgraph=True` - Require single graph
- [ ] `dynamic=True` - Dynamic shapes support

### Study Order
1. Start with `torch/_dynamo/__init__.py` and `eval_frame.py`
2. Read `symbolic_convert.py` for bytecode capture
3. Study `guards.py` for guard system
4. Explore `torch/_inductor/` for backend compilation
5. Read `torch/_functorch/aot_autograd.py` for backward capture

---

## 16. Mixed Precision (AMP)

Automatic mixed precision training for performance.

### Key Files
- [ ] `torch/amp/__init__.py` - AMP public API
- [ ] `torch/amp/autocast_mode.py` - Autocast implementation
- [ ] `torch/amp/grad_scaler.py` - Gradient scaler
- [ ] `torch/cuda/amp/autocast_mode.py` - CUDA-specific autocast
- [ ] `torch/cuda/amp/grad_scaler.py` - CUDA gradient scaler
- [ ] `torch/cpu/amp/autocast_mode.py` - CPU autocast

### Core Concepts
- [ ] Understand autocast context manager
- [ ] Learn which ops run in FP16/BF16 vs FP32
- [ ] Study GradScaler for loss scaling
- [ ] Understand inf/nan handling and skipping updates
- [ ] Learn autocasting rules and op categories
- [ ] Study CPU vs CUDA autocast differences
- [ ] Understand BF16 vs FP16 tradeoffs

### Autocast Op Categories
- [ ] Ops that run in lower precision (matmul, conv)
- [ ] Ops that run in FP32 (softmax, layer_norm, loss)
- [ ] Ops that promote to widest input type

### Key Functions
- [ ] `torch.autocast(device_type, dtype)` - Context manager
- [ ] `torch.cuda.amp.autocast()` - CUDA autocast
- [ ] `torch.cuda.amp.GradScaler()` - Gradient scaling
- [ ] `scaler.scale(loss)` - Scale loss
- [ ] `scaler.step(optimizer)` - Unscale and step
- [ ] `scaler.update()` - Update scale factor

### Study Order
1. Start with `torch/amp/autocast_mode.py`
2. Read `torch/amp/grad_scaler.py`
3. Understand op categorization for autocasting
4. Study integration with training loop

---

## 17. Quantization

Model quantization for inference optimization.

### Key Files - Core Quantization
- [ ] `torch/ao/quantization/__init__.py` - Quantization entry point
- [ ] `torch/ao/quantization/observer.py` - Observers for calibration
- [ ] `torch/ao/quantization/fake_quantize.py` - Fake quantization
- [ ] `torch/ao/quantization/qconfig.py` - Quantization configuration
- [ ] `torch/ao/quantization/quantize.py` - Quantization functions

### Key Files - Quantized Modules
- [ ] `torch/ao/nn/quantized/modules/` - Quantized nn modules
- [ ] `torch/ao/nn/quantized/modules/linear.py` - Quantized linear
- [ ] `torch/ao/nn/quantized/modules/conv.py` - Quantized convolutions
- [ ] `torch/ao/nn/quantized/dynamic/modules/` - Dynamic quantization

### Key Files - PT2 Quantization
- [ ] `torch/ao/quantization/pt2e/` - PT2E quantization
- [ ] `torch/ao/quantization/quantizer/` - Quantizer framework

### Quantization Approaches
- [ ] Post-Training Quantization (PTQ) - Quantize after training
- [ ] Quantization-Aware Training (QAT) - Train with fake quantization
- [ ] Dynamic Quantization - Quantize weights, dynamic activations
- [ ] Static Quantization - Quantize weights and activations

### Core Concepts
- [ ] Understand Observer for range calibration
- [ ] Learn FakeQuantize for QAT simulation
- [ ] Study QConfig for quantization settings
- [ ] Understand per-tensor vs per-channel quantization
- [ ] Learn symmetric vs asymmetric quantization
- [ ] Study backend-specific constraints
- [ ] Understand PT2E quantization workflow

### Study Order
1. Start with `torch/ao/quantization/observer.py`
2. Read `torch/ao/quantization/fake_quantize.py`
3. Study `torch/ao/quantization/qconfig.py`
4. Explore quantized modules in `torch/ao/nn/quantized/`
5. Learn PT2E approach in `torch/ao/quantization/pt2e/`

---

## 18. Export and ONNX

Exporting models for deployment.

### Key Files - torch.export
- [ ] `torch/export/__init__.py` - Export entry point
- [ ] `torch/export/exported_program.py` - ExportedProgram class
- [ ] `torch/export/dynamic_shapes.py` - Dynamic shape specification
- [ ] `torch/export/_trace.py` - Export tracing
- [ ] `torch/export/unflatten.py` - Unflatten exported program

### Key Files - ONNX Export
- [ ] `torch/onnx/__init__.py` - ONNX export entry
- [ ] `torch/onnx/utils.py` - Export utilities
- [ ] `torch/onnx/symbolic_helper.py` - Symbolic helpers
- [ ] `torch/onnx/_internal/` - Internal ONNX implementation
- [ ] `torch/onnx/symbolic_opset*.py` - Opset implementations

### Core Concepts - torch.export
- [ ] Understand ExportedProgram structure
- [ ] Learn dynamic shapes specification
- [ ] Study guards and assumptions
- [ ] Understand graph signature (inputs, outputs)
- [ ] Learn module hierarchy preservation
- [ ] Study serialization of exported programs
- [ ] Understand AOT Inductor for ahead-of-time compilation

### Core Concepts - ONNX
- [ ] Understand ONNX operator mapping
- [ ] Learn opset version handling
- [ ] Study dynamic axes specification
- [ ] Understand custom operator export
- [ ] Learn ONNX Runtime integration

### Key Functions
- [ ] `torch.export.export(model, args)` - Export model
- [ ] `exported.module()` - Get callable module
- [ ] `torch.onnx.export(model, args, f)` - Export to ONNX
- [ ] `torch.onnx.dynamo_export()` - Dynamo-based ONNX export

### Study Order
1. Start with `torch/export/__init__.py`
2. Read `torch/export/exported_program.py`
3. Study `torch/export/dynamic_shapes.py`
4. Explore ONNX export in `torch/onnx/`

---

## 19. Profiling and Debugging

Tools for performance analysis and debugging.

### Key Files - Profiler
- [ ] `torch/profiler/__init__.py` - Profiler entry point
- [ ] `torch/profiler/profiler.py` - Main profiler implementation
- [ ] `torch/profiler/_memory_profiler.py` - Memory profiling
- [ ] `torch/autograd/profiler.py` - Autograd profiling
- [ ] `torch/autograd/profiler_util.py` - Profiler utilities

### Key Files - Debugging
- [ ] `torch/autograd/anomaly_mode.py` - Anomaly detection
- [ ] `torch/autograd/gradcheck.py` - Gradient checking
- [ ] `torch/testing/_internal/common_utils.py` - Test utilities

### Profiler Features
- [ ] `torch.profiler.profile()` - Context manager for profiling
- [ ] `torch.profiler.record_function()` - Label code regions
- [ ] `torch.profiler.schedule()` - Profiling schedule
- [ ] `torch.profiler.tensorboard_trace_handler()` - TensorBoard export

### Core Concepts - Profiling
- [ ] Understand Kineto integration for GPU profiling
- [ ] Learn CPU time vs CUDA time distinction
- [ ] Study memory profiling capabilities
- [ ] Understand trace export (Chrome, TensorBoard)
- [ ] Learn operator-level statistics
- [ ] Study stack trace collection

### Core Concepts - Debugging
- [ ] Understand detect_anomaly for debugging NaN/Inf
- [ ] Learn gradcheck for verifying gradients
- [ ] Study model hooks for inspection
- [ ] Understand torch.autograd.set_detect_anomaly
- [ ] Learn print/breakpoint debugging in traced code

### Key Functions
- [ ] `torch.profiler.profile()` - Start profiling
- [ ] `prof.key_averages()` - Aggregate statistics
- [ ] `prof.export_chrome_trace()` - Export trace
- [ ] `torch.autograd.set_detect_anomaly(True)` - Enable anomaly detection
- [ ] `torch.autograd.gradcheck(func, inputs)` - Check gradients

### Study Order
1. Start with `torch/profiler/profiler.py`
2. Read `torch/autograd/profiler.py`
3. Study memory profiling capabilities
4. Learn anomaly detection in `torch/autograd/anomaly_mode.py`
5. Explore gradient checking utilities

---

## Recommended Study Path

### Phase 1: Foundations (Weeks 1-2)
- [ ] Complete Section 1: Foundation Layer
- [ ] Complete Section 2: Core Tensor System
- [ ] Complete Section 8: Python/C++ Bindings (basics)

### Phase 2: Core Mechanics (Weeks 3-4)
- [ ] Complete Section 3: Dispatcher Mechanism
- [ ] Complete Section 4: Autograd System
- [ ] Complete Section 9: Operator Registration

### Phase 3: Execution Infrastructure (Weeks 5-6)
- [ ] Complete Section 5: Memory Management
- [ ] Complete Section 6: CUDA Caching Allocator (Deep Dive)
- [ ] Complete Section 7: CUDA Integration
- [ ] Complete Section 16: Mixed Precision (AMP)

### Phase 4: High-Level APIs (Weeks 7-8)
- [ ] Complete Section 10: Neural Network Modules
- [ ] Complete Section 11: Data Loading Pipeline
- [ ] Complete Section 12: Serialization System

### Phase 5: Distributed and Scale (Weeks 9-10)
- [ ] Complete Section 13: Distributed Training
- [ ] Complete Section 17: Quantization

### Phase 6: Compilation and Optimization (Weeks 11-12)
- [ ] Complete Section 14: FX Graph System
- [ ] Complete Section 15: torch.compile and Dynamo
- [ ] Complete Section 18: Export and ONNX
- [ ] Complete Section 19: Profiling and Debugging

---

## Tips for Studying

1. **Read code with a debugger**: Set breakpoints and trace execution
2. **Write small examples**: Create minimal scripts to test each concept
3. **Use print statements**: Add prints to understand data flow
4. **Check tests**: Test files often show expected behavior
5. **Read documentation**: Comments in code are often insightful
6. **Build incrementally**: Start with simple ops, work up to complex ones
7. **Ask questions**: Use PyTorch forums and GitHub discussions

---

## Progress Tracking

| Section | Status | Notes |
|---------|--------|-------|
| 1. Foundation Layer | [ ] | |
| 2. Core Tensor System | [ ] | |
| 3. Dispatcher Mechanism | [ ] | |
| 4. Autograd System | [ ] | |
| 5. Memory Management | [ ] | |
| 6. CUDA Caching Allocator | [ ] | |
| 7. CUDA Integration | [ ] | |
| 8. Python/C++ Bindings | [ ] | |
| 9. Operator Registration | [ ] | |
| 10. Neural Network Modules | [ ] | |
| 11. Data Loading Pipeline | [ ] | |
| 12. Serialization System | [ ] | |
| 13. Distributed Training | [ ] | |
| 14. FX Graph System | [ ] | |
| 15. torch.compile/Dynamo | [ ] | |
| 16. Mixed Precision (AMP) | [ ] | |
| 17. Quantization | [ ] | |
| 18. Export and ONNX | [ ] | |
| 19. Profiling and Debugging | [ ] | |

---

*Last updated: December 2024*
