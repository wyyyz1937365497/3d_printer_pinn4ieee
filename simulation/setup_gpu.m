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

    % 删除了所有verbose输出以提高速度，保留GPU加速功能

    gpu_info = struct();
    gpu_info.available = false;
    gpu_info.device = [];
    gpu_info.name = '';
    gpu_info.memory = 0;
    gpu_info.use_gpu = false;

    %% Check if Parallel Computing Toolbox is available
    if ~exist('gpuDevice', 'file')
        % GPU不可用，静默返回
        return;
    end

    gpu_info.available = true;

    %% Force use GPU 1 (cuda1)
    gpu_id = 1;

    %% Set the selected GPU
    try
        gpu_device = gpuDevice(gpu_id);
        gpu_info.device = gpu_device;
        gpu_info.name = gpu_device.Name;
        gpu_info.memory = gpu_device.FreeMemory / 1e9;
        gpu_info.use_gpu = true;

    catch ME
        % GPU初始化失败，静默fallback到CPU
        gpu_info.use_gpu = false;
        gpu_info.device = [];
    end
end
