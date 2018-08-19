% ========== Write mask labels to output file ==========

f = fopen('stage1_test_submission.csv', 'w');
fprintf(f, "ImageId,EncodedPixels\n");

im = imread(strcat(imageId,'.png'));
[y, x, ~] = size(im);
[im_X, im_Y] = meshgrid(1:x, 1:y);
grid = (im_X-1) * y + im_Y;
        
for k = 1 : NumLabels
    if sum(sum(im_pred == k)) > 0
        arr = grid(im_pred == k);
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
        fprintf(f, strcat(imageId, ",", pixels, "\n"));
    end
end