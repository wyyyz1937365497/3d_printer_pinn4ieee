function gpu_info = setup_gpu(gpu_id)
% SETUP_GPU - Configure and initialize GPU for MATLAB simulation
%
% This function sets up the specified GPU device for computation,
% with fallback to CPU if GPU is not available.
%
% Inputs:
%   gpu_id - GPU device ID (e.g., 0 for cuda0, 1 for cuda1)
%           Set to [] to auto-select the GPU with most free memory
%
% Output:
%   gpu_info - Structure with GPU information:
%     .available - true if GPU is available
%     .device    - GPU device object
%     .name      - GPU device name
%     .memory    - Available memory (GB)
%     .use_gpu   - Whether to use GPU for computation
%
% Example:
%   % Use cuda1 (GPU 1)
%   gpu = setup_gpu(1);
%
%   % Auto-select GPU with most free memory
%   gpu = setup_gpu([]);

    fprintf('============================================================\n');
    fprintf('GPU Setup for MATLAB Simulation\n');
    fprintf('============================================================\n');
    fprintf('\n');

    gpu_info = struct();
    gpu_info.available = false;
    gpu_info.device = [];
    gpu_info.name = '';
    gpu_info.memory = 0;
    gpu_info.use_gpu = false;

    %% Check if Parallel Computing Toolbox is available
    if ~exist('gpuDevice', 'file')
        fprintf('WARNING: Parallel Computing Toolbox not found!\n');
        fprintf('GPU acceleration is not available.\n');
        fprintf('Simulation will run on CPU.\n');
        fprintf('\n');
        fprintf('To enable GPU acceleration:\n');
        fprintf('  1. Install Parallel Computing Toolbox\n');
        fprintf('  2. Ensure CUDA-compatible GPU is detected\n');
        fprintf('\n');
        return;
    end

    gpu_info.available = true;
    fprintf('Parallel Computing Toolbox: Available ✓\n');
    fprintf('\n');

    %% List all available GPUs
    try
        num_gpus = gpuDeviceCount();
        fprintf('Detected GPUs: %d\n', num_gpus);

        if num_gpus == 0
            fprintf('WARNING: No CUDA-compatible GPUs detected!\n');
            fprintf('Simulation will run on CPU.\n');
            fprintf('\n');
            return;
        end

        % Display GPU information
        fprintf('\nAvailable GPU Devices:\n');
        for i = 0:num_gpus-1
            try
                gpu_dev = gpuDevice(i);
                free_mem_gb = gpu_dev.FreeMemory / 1e9;
                total_mem_gb = gpu_dev.TotalMemory / 1e9;
                fprintf('  GPU %d: %s\n', i, gpu_dev.Name);
                fprintf('         Compute Capability: %s\n', gpu_dev.ComputeCapability);
                fprintf('         Total Memory: %.2f GB\n', total_mem_gb);
                fprintf('         Free Memory:  %.2f GB\n', free_mem_gb);
                fprintf('        利用率: %.1f%%\n', (1 - free_mem_gb/total_mem_gb) * 100);
            catch ME
                fprintf('  GPU %d: Error reading info - %s\n', i, ME.message);
            end
        end
        fprintf('\n');

    catch ME
        fprintf('ERROR: Failed to query GPU devices!\n');
        fprintf('Error: %s\n', ME.message);
        fprintf('Simulation will run on CPU.\n');
        fprintf('\n');
        return;
    end

    %% Select GPU device
    if isempty(gpu_id)
        % Auto-select: Choose GPU with most free memory
        fprintf('Auto-selecting GPU with most free memory...\n');
        max_free_mem = 0;
        best_gpu = 0;

        for i = 0:num_gpus-1
            try
                gpu_dev = gpuDevice(i);
                free_mem = gpu_dev.FreeMemory;
                if free_mem > max_free_mem
                    max_free_mem = free_mem;
                    best_gpu = i;
                end
            catch
                continue;
            end
        end

        gpu_id = best_gpu;
        fprintf('Selected GPU %d ( %.2f GB free)\n', gpu_id, max_free_mem/1e9);

    else
        % Use specified GPU
        if gpu_id < 0 || gpu_id >= num_gpus
            fprintf('WARNING: Invalid GPU ID %d (available: 0-%d)\n', gpu_id, num_gpus-1);
            fprintf('Auto-selecting GPU with most free memory...\n');

            max_free_mem = 0;
            best_gpu = 0;
            for i = 0:num_gpus-1
                try
                    gpu_dev = gpuDevice(i);
                    free_mem = gpu_dev.FreeMemory;
                    if free_mem > max_free_mem
                        max_free_mem = free_mem;
                        best_gpu = i;
                    end
                catch
                    continue;
                end
            end

            gpu_id = best_gpu;
        end
    end

    %% Set the selected GPU
    try
        gpu_device = gpuDevice(gpu_id);
        gpu_info.device = gpu_device;
        gpu_info.name = gpu_device.Name;
        gpu_info.memory = gpu_device.FreeMemory / 1e9;
        gpu_info.use_gpu = true;

        fprintf('\nSelected GPU Configuration:\n');
        fprintf('  Device: %s\n', gpu_device.Name);
        fprintf('  GPU ID: %d\n', gpu_id);
        fprintf('  Free Memory: %.2f GB\n', gpu_info.memory);
        fprintf('  Total Memory: %.2f GB\n', gpu_device.TotalMemory / 1e9);
        fprintf('  Multiprocessor Count: %d\n', gpu_device.MultiprocessorCount);
        fprintf('\n');

        % Reset GPU to ensure clean state
        reset(gpu_device);
        fprintf('GPU ready for computation ✓\n');
        fprintf('\n');

    catch ME
        fprintf('ERROR: Failed to initialize GPU %d!\n', gpu_id);
        fprintf('Error: %s\n', ME.message);
        fprintf('Simulation will run on CPU.\n');
        fprintf('\n');
        gpu_info.use_gpu = false;
        gpu_info.device = [];
    end

    fprintf('============================================================\n');
    fprintf('\n');
end
