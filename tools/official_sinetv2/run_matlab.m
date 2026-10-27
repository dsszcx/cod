function run_matlab(predRoot, dataRoot, upstreamRoot, outFile)
% Execute the original author's metric functions; needs MATLAB + toolboxes.
% Example: run_matlab('/path/run/predictions','/path/TestDataset', ...
%                    '/path/SINet-V2','/path/run/metrics_matlab.json')
addpath(fullfile(upstreamRoot, 'eval'));
datasets = {'CAMO', 'COD10K'};
expected = [250, 2026];
thresholds = 1:-1/255:0;
report.protocol = 'Original SINet-V2 MATLAB metric functions';
for d = 1:numel(datasets)
    dataset = datasets{d};
    files = dir(fullfile(dataRoot, dataset, 'GT', '*.png'));
    if strcmp(dataset, 'COD10K')
        files = files(contains({files.name}, '-CAM-'));
    end
    assert(numel(files) == expected(d), 'Unexpected benchmark count');
    values = zeros(numel(files), 4);
    for i = 1:numel(files)
        gt = imread(fullfile(files(i).folder, files(i).name));
        if ndims(gt) > 2, gt = rgb2gray(gt); end
        if ~islogical(gt), gt = gt(:,:,1) > 128; end
        assert(any(gt(:)), 'Empty GT is outside CAM-only protocol');
        cam = imread(fullfile(predRoot, dataset, files(i).name));
        cam = im2double(cam(:,:,1));
        assert(isequal(size(cam), size(gt)), 'Prediction size mismatch');
        cam = reshape(mapminmax(cam(:)', 0, 1), size(cam));
        curve = zeros(1, numel(thresholds));
        for t = 1:numel(thresholds)
            curve(t) = Enhancedmeasure(cam > thresholds(t), gt);
        end
        values(i,:) = [StructureMeasure(cam, gt), mean(curve), ...
                       original_WFb(cam, gt), mean2(abs(double(gt)-cam))];
        if mod(i, 200) == 0 || i == numel(files)
            fprintf('MATLAB_EVAL %s %d/%d\n', dataset, i, numel(files));
        end
    end
    assert(all(isfinite(values(:))), 'Nonfinite metric, do not skip sample');
    avg = mean(values, 1);
    report.datasets.(dataset) = struct('count',numel(files), ...
        'sm',avg(1),'mean_e',avg(2),'wfm',avg(3),'mae',avg(4));
end
fid = fopen(outFile, 'w');
assert(fid ~= -1, 'Cannot open output JSON');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(report));
disp('OFFICIAL_SINETV2_MATLAB_EVAL_DONE');
end
