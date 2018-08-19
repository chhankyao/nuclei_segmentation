load('SVMs.mat');
load('shape.mat');

l = dir('../stage1_test');
f = fopen('stage1_test_submission.csv', 'w');
fprintf(f, "ImageId,EncodedPixels\n");

count_all = zeros(length(l)-3,1);
class_pred = zeros(length(l)-3, 1);
verb = '';

for i = 4 : length(l)
    fprintf(repmat('\b',[1, length(verb)]));
    verb = sprintf('i = %d', i);
    fprintf(verb);
    
    % Read image
    imId = l(i).name;
    im = im2double(imread(strcat(imId, '.png')));
    im = im - min(im(:));
    im = im / max(im(:));
    [y, x, ~] = size(im);
    [im_X, im_Y] = meshgrid(1:x, 1:y);
    grid = (im_X-1) * y + im_Y;
    
    % Classify the image
    if any(im(:,:,1) ~= im(:,:,2))
        class_pred(i-3) = 1;
    elseif mean(im(:)) < 0.5
        class_pred(i-3) = 2;
    else
        class_pred(i-3) = 3;
    end
    im = rgb2gray(im);   
    
    % Superpixel segmentation
    if class_pred(i-3) == 1
        N_superpixels = 1/area1;
        compact = 5;
    elseif class_pred(i-3) == 2
        N_superpixels = 1/area2;
        compact = 10;
    else
        N_superpixels = 1/area3;
        compact = 10;
    end
    
    % ==================== Prediction layer 1 ====================
    [L1, NumLabels1] = superpixels(im, round(5.0*N_superpixels), 'Compactness', compact*2, 'Method', 'slic');
    
    im_pred = zeros(size(im));
    im_pred2 = zeros(size(im));
    pred = zeros(NumLabels1, 1);
    
    for k = 1 : NumLabels1
        hist = (mean(im(L1 == k)) - mean(im(:))) / std(im(:));
        
        if class_pred(i-3) == 1
            pred(k) = predict(SVM_c1, hist);
        elseif class_pred(i-3) == 2
            pred(k) = predict(SVM_c2, hist);
        else
            pred(k) = predict(SVM_c3, hist);
        end
        
        if pred(k) == 1
            im_pred(L1 == k) = k;
        end
    end        
    
    % ==================== Prediction layer 2 ====================
    
    [L2, NumLabels2] = superpixels(im, round(2.0*N_superpixels), 'Compactness', compact, 'Method', 'slic');
    for k = 1 : NumLabels2   
        proportion = sum(sum(L2 == k & im_pred > 0)) / sum(sum(L2 == k));
        if proportion > 0.8
            labels = unique(im_pred(L2 == k));
            for j = 1 : length(labels)
                propotion2 = sum(sum(L2 == k & im_pred == labels(j))) / sum(sum(im_pred == labels(j)));
                if propotion2 < 0.5
                    labels(j) = 0;
                end
            end
            for j = 1 : length(labels)
                if labels(j) > 0
                    im_pred(L2 == k) = max(labels);
                end
            end
        end
    end

    % ==================== Prediction layer 3 ====================
    
    [L3, NumLabels3] = superpixels(im, round(1.0*N_superpixels), 'Compactness', compact, 'Method', 'slic');
    for k = 1 : NumLabels3
        proportion = sum(sum(L3 == k & im_pred > 0)) / sum(sum(L3 == k));
        if proportion > 0.8
            labels = unique(im_pred(L3 == k));
            for j = 1 : length(labels)
                propotion2 = sum(sum(L3 == k & im_pred == labels(j))) / sum(sum(im_pred == labels(j)));
                if propotion2 < 0.5
                    labels(j) = 0;
                end
            end
            for j = 1 : length(labels)
                if labels(j) > 0
                    im_pred(L3 == k) = max(labels);
                end
            end
        end
    end
    
    
    % ==================== Prediction layer 4 ====================
    
    [L4, NumLabels4] = superpixels(im, round(0.5*N_superpixels), 'Compactness', compact, 'Method', 'slic');
    for k = 1 : NumLabels4
        proportion = sum(sum(L4 == k & im_pred > 0)) / sum(sum(L4 == k));
        if proportion > 0.8
            labels = unique(im_pred(L4 == k));
            for j = 1 : length(labels)
                propotion2 = sum(sum(L4 == k & im_pred == labels(j))) / sum(sum(im_pred == labels(j)));
                if propotion2 < 0.5
                    labels(j) = 0;
                end
            end
            for j = 1 : length(labels)
                if labels(j) > 0
                    im_pred(L4 == k) = max(labels);
                end
            end
        end
    end
    
    % ==================== Prediction layer 5 ====================
    
    [L5, NumLabels5] = superpixels(im, round(0.2*N_superpixels), 'Compactness', compact/2, 'Method', 'slic');
    for k = 1 : NumLabels5
        proportion = sum(sum(L5 == k & im_pred > 0)) / sum(sum(L5 == k));
        if proportion > 0.8
            labels = unique(im_pred(L5 == k));
            for j = 1 : length(labels)
                propotion2 = sum(sum(L5 == k & im_pred == labels(j))) / sum(sum(im_pred == labels(j)));
                if propotion2 < 0.5
                    labels(j) = 0;
                end
            end
            for j = 1 : length(labels)
                if labels(j) > 0
                    im_pred(L5 == k) = max(labels);
                end
            end
        end
    end
    
    % =============== Remove the predictions with unusual size ===============
    area = 0;
    num = 0;
    for k = 1 : NumLabels1
        if sum(sum(im_pred == k)) > 50
            area = area + sum(sum(im_pred == k));
            num = num + 1;
        end
    end
    area = area / num;
    
    for k = 1 : NumLabels1
        if sum(sum(im_pred == k)) < area/5 || sum(sum(im_pred == k)) > area*5
            im_pred(im_pred == k) = 0;
        end
    end
    
%     im_output = zeros(size(im,1), size(im,2), 3);
%     im_output(:,:,1) = (mod(im_pred, 3) == 0 | im_pred == 0) .* (im_pred/max(im_pred(:)));
%     im_output(:,:,2) = (mod(im_pred, 3) == 1 | im_pred == 0) .* (im_pred/max(im_pred(:)));
%     im_output(:,:,3) = (mod(im_pred, 3) == 2 | im_pred == 0) .* (im_pred/max(im_pred(:)));
%     imwrite(im_output, strcat('test/', imId, '_superpixel.png'));
    
    
    % =============== Clustering on predicted mask =============== 
    [L, NumLabels] = superpixels(im_pred, round(numel(im)/(2*area)), 'Compactness', 1, 'Method', 'slic');
    for k = 1 : NumLabels
        if sum(sum(L == k)) > 50 && sum(sum(L == k & im_pred > 0)) > 0.5 * sum(sum(L == k))
            im_pred2(L == k) = k;
        end
    end
    
    
    for k = 1 : NumLabels
        if sum(sum(im_pred2 == k)) > 50
            arr = grid(im_pred2 == k);
            count = 1;
            anchor = arr(1);
            prev = arr(1);
            pixels = '';
            for j = 2 : length(arr)
                if (arr(j) ~= prev + 1 || j == length(arr))
                    pixels = strcat(pixels, num2str(anchor), " ", num2str(count), " ");
                    anchor = arr(j);
                    prev = arr(j);
                    count = 1;
                else
                    count = count + 1;
                    prev = arr(j);
                end
            end
            fprintf(f, strcat(imId, ",", pixels, "\n"));
            count_all(i-3) = 1;
        end
    end
    
    % Output predicted mask
%     im_output = zeros(size(im,1), size(im,2), 3);
%     im_output(:,:,1) = (mod(im_pred2, 3) == 0 | im_pred2 == 0) .* (im_pred2/max(im_pred2(:)));
%     im_output(:,:,2) = (mod(im_pred2, 3) == 1 | im_pred2 == 0) .* (im_pred2/max(im_pred2(:)));
%     im_output(:,:,3) = (mod(im_pred2, 3) == 2 | im_pred2 == 0) .* (im_pred2/max(im_pred2(:)));
%     imwrite(im_output, strcat('test/', imId, '_superpixel.png'));
end
fprintf(repmat('\b',[1, length(verb)]));