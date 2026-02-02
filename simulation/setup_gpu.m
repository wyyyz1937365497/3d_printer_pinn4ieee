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

    %% Force use GPU 1 (cuda1)
    gpu_id = 1;
    fprintf('Using GPU 1 (cuda1)\n');

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
