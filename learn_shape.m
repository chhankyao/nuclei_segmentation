M = readtable('stage1_train_labels.csv');

imageId = '';
count1 = 0.0;
count2 = 0.0;
count3 = 0.0;
area1 = 0.0;
area2 = 0.0;
area3 = 0.0;

for i = 1 : height(M)
    fileName = M(i,1).ImageId{1};
    if ~strcmp(fileName, imageId)
        imageId = fileName;
        im = im2double(imread(strcat(imageId,'.png')));    
        mask_all = im2double(imread(strcat('train/',imageId,'_mask.png')));
        if any(im(:,:,1) ~= im(:,:,2))
            class = 1;
        elseif mean(im(:)) < 0.5
            class = 2;
        else
            class = 3;
        end
        im = rgb2gray(im);
    end
    labels = str2num(M(i,2).EncodedPixels{1});
    mask = zeros(size(im));
    for j = 1 : length(labels) / 2
        idx = 2 * j -1;
        mask(labels(idx):labels(idx)+labels(idx+1)-1) = 1;
    end
    if class == 1
        area1 = area1 + sum(sum(mask == 1)) / numel(im);
        count1 = count1 + 1;
    elseif class == 2
        area2 = area2 + sum(sum(mask == 1)) / numel(im);
        count2 = count2 + 1;
    else
        area3 = area3 + sum(sum(mask == 1)) / numel(im);
        count3 = count3 + 1;
    end    
end

area1 = area1 / count1;
area2 = area2 / count2;
area3 = area3 / count3;

save('shape.mat', 'area1', 'area2', 'area3');