addpath(genpath('*/Kaggle/stage1_test/'));

l = dir('stage1_test');
f = fopen('test_ensemble.csv', 'w');
fprintf(f, "ImageId,EncodedPixels\n");

for i = 4 : length(l)
    imId = l(i).name;
    im1 = rgb2gray(im2double(imread(strcat('test/', imId, '_model1.png'))));
    im2 = rgb2gray(im2double(imread(strcat('test/', imId, '_model2.png'))));
    im3 = rgb2gray(im2double(imread(strcat('test/', imId, '_model3.png'))));
    im4 = rgb2gray(im2double(imread(strcat('test/', imId, '_model4.png'))));
    im5 = rgb2gray(im2double(imread(strcat('test/', imId, '_model5.png'))));
    im6 = rgb2gray(im2double(imread(strcat('test/', imId, '_model6.png'))));
    im7 = rgb2gray(im2double(imread(strcat('test/', imId, '_model7.png'))));
    im8 = rgb2gray(im2double(imread(strcat('test/', imId, '_model8.png'))));
    im9 = rgb2gray(im2double(imread(strcat('test/', imId, '_model9.png'))));
    
    im1(im1 > 0) = 1;
    im2(im2 > 0) = 1;
    im3(im3 > 0) = 1;
    im4(im4 > 0) = 1;
    im5(im5 > 0) = 1;
    im6(im6 > 0) = 1;
    im7(im7 > 0) = 1;
    im8(im8 > 0) = 1;
    im9(im9 > 0) = 1;
    
    im_ensemble = im1 + im2 + im3 + im4 + im5 + im6 + im7 + im8 + im9;
    im_ensemble(im_ensemble < 5) = 0;
    im_ensemble(im_ensemble >= 5) = 1;
    
    CC = bwconncomp(im_ensemble, 4);
    L = labelmatrix(CC);
    L = imfill(L, 'holes');

    
    NumLabels = max(L(:));
    for k = 1 : NumLabels
        if sum(sum(L == k)) < 50
            L(L == k) = 0;
        end
    end
    
    im_output = zeros(size(im1,1), size(im1,2), 3);
    im_output(:,:,1) = (mod(L, 2) == 0 & L > 0);
    im_output(:,:,2) = (mod(L, 2) == 1);
    im_output(:,:,3) = (mod(L, 3) == 1);
    imwrite(im_output, strcat('test/', imId, '_ensemble.png'));
    
    
    % ========== Write mask labels to output file ==========
    [y, x] = size(im1);
    [im_X, im_Y] = meshgrid(1:x, 1:y);
    grid = (im_X-1) * y + im_Y;
        
    for k = 1 : NumLabels
        if sum(sum(L == k)) > 0
            arr = grid(L == k);
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
        end
    end
end