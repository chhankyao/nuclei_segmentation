% ========== Training ==========
l = dir('stage1_train');
idx_c1 = 1;
idx_c2 = 1;
idx_c3 = 1;
idx_class = 1;
class = zeros(length(l)-3, 1);
label_class = zeros(length(l)-3, 1);
hist_class = zeros(length(l)-3, 256);
background_c1 = zeros(length(l)-3, 1);
background_c2 = zeros(length(l)-3, 1);
background_c3 = zeros(length(l)-3, 1);
foreground_c1 = zeros(length(l)-3, 1);
foreground_c2 = zeros(length(l)-3, 1);
foreground_c3 = zeros(length(l)-3, 1);

for i = 4 : length(l)
    imId = l(i).name;
    im = im2double(imread(strcat(imId, '.png')));
    mask = im2double(imread(strcat('train/', imId, '_mask.png')));
    im = im - min(im(:));
    im = im / max(im(:));

    if any(im(:,:,1) ~= im(:,:,2))
        class(i-3) = 1;
    elseif mean(im(:)) < 0.5
        class(i-3) = 2;
        label_class(idx_class) = class(i-3);
        hist_class(idx_class,:) = imhist(im)';
        idx_class = idx_class + 1;
    else
        class(i-3) = 3;
        label_class(idx_class) = class(i-3);
        hist_class(idx_class,:) = imhist(im)';
        idx_class = idx_class + 1;
    end
    
    im = rgb2gray(im);
    foreground = im(mask == 1);
    background = im(mask == 0);
%     h_foreground = imhist((foreground - mean(im(:))) / std(im(:)))';
%     h_background = imhist((background - mean(im(:))) / std(im(:)))';
    h_foreground = (mean(foreground) - mean(im(:))) / std(im(:));
    h_background = (mean(background) - mean(im(:))) / std(im(:));
    
    if (class(i-3) == 1)
        background_c1(idx_c1,:) = h_background;% / sum(h_background);
        foreground_c1(idx_c1,:) = h_foreground;% / sum(h_foreground);
        idx_c1 = idx_c1 + 1;
    elseif (class(i-3) == 2)
        background_c2(idx_c2,:) = h_background;% / sum(h_background);
        foreground_c2(idx_c2,:) = h_foreground;% / sum(h_foreground);
        idx_c2 = idx_c2 + 1;
    else
        background_c3(idx_c3,:) = h_background;% / sum(h_background);
        foreground_c3(idx_c3,:) = h_foreground;% / sum(h_foreground);
        idx_c3 = idx_c3 + 1;
    end
end
hist_class = hist_class(1:idx_class-1,:);
label_class = label_class(1:idx_class-1);
background_c1 = background_c1(1:idx_c1-1,:);
foreground_c1 = foreground_c1(1:idx_c1-1,:);
background_c2 = background_c2(1:idx_c2-1,:);
foreground_c2 = foreground_c2(1:idx_c2-1,:);
background_c3 = background_c3(1:idx_c3-1,:);
foreground_c3 = foreground_c3(1:idx_c3-1,:);

SVM_class = fitcsvm(hist_class, label_class);
train_c1 = [background_c1; foreground_c1];
train_c2 = [background_c2; foreground_c2];
train_c3 = [background_c3; foreground_c3];
label_c1 = [zeros(size(background_c1,1),1); ones(size(background_c1,1),1)];
label_c2 = [zeros(size(background_c2,1),1); ones(size(background_c2,1),1)];
label_c3 = [zeros(size(background_c3,1),1); ones(size(background_c3,1),1)];
SVM_c1 = fitcsvm(train_c1, label_c1);
SVM_c2 = fitcsvm(train_c2, label_c2);
SVM_c3 = fitcsvm(train_c3, label_c3);

save('SVMs.mat', 'SVM_class', 'SVM_c1', 'SVM_c2', 'SVM_c3');