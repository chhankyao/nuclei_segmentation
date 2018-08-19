l = dir('stage1_test');
f = fopen('stage1_test_ensemble.csv', 'w');
fprintf(f, "ImageId,EncodedPixels\n");

M = readtable('test_ensemble.csv');

verb = '';
imageId = '';
for i = 1 : height(M)+1
    fprintf(repmat('\b',[1, length(verb)]));
    verb = sprintf('i = %d', i);
    fprintf(verb);
    
    if i ~= height(M)+1
        fileName = M(i,1).ImageId{1};
    end
    if i == height(M)+1 || ~strcmp(fileName, imageId)
        if imageId ~= ""
            im_output = zeros(size(im));
            im_output(:,:,1) = (mod(im_pred, 2) == 0 & im_pred > 0);
            im_output(:,:,2) = (mod(im_pred, 2) == 1);
            im_output(:,:,3) = (mod(im_pred, 3) == 1);
            imwrite(im_output, strcat('test/', imageId, '_gmm.png'));

            % ========== EM with Gaussian Mixture Model ==========
            areas = unique(areas);
            area = sum(areas) / (length(areas)-1);
            im_pred2 = im_pred;
            count_label = idx_label;
            
            if all(im(:,:,1) == im(:,:,2))
                threshold = 0.015;
            else
                threshold = 0.035;
            end
            
            for k = 1 : idx_label
                 data = [im_X(im_pred == k), im_Y(im_pred == k)];
                 
                 if length(data) < max(50,area-2*std(areas))
                     im_pred2(im_pred == k) = 0;
                     
                 elseif idx_label > 5 && length(data) > max(50,area)                  
                     n_cluster = 3;
                     label_cluster = zeros(n_cluster, length(data));
                     ll_neg = zeros(n_cluster, 1);
                     
                     for s = 1 : n_cluster
                         GMModel = fitgmdist(data, s, 'Options', statset('MaxIter',1000), 'RegularizationValue', 0.001, 'Replicates', 10);
                         ll_neg(s) = GMModel.NegativeLogLikelihood / length(data);
                         label_cluster(s,:) = cluster(GMModel, data)';
                     end
                     
                     diff_ll = diff(diff(ll_neg));
                     [diff_max, n_cluster] = max(diff_ll);
                     if diff_max > threshold
                         n_cluster = n_cluster + 1;
                     else
                         n_cluster = 1;
                     end
                     label_cluster = label_cluster(n_cluster, :)';

                     NumLabels = max(label_cluster);
                     im_pred2(im_pred == k) = label_cluster + count_label;
                     count_label = count_label + NumLabels;
                 end
            end
            
            % ========== EM with Gaussian Mixture Model ==========
            idx_label = count_label;    
            
            for k = 1 : idx_label
                 data = [im_X(im_pred2 == k), im_Y(im_pred2 == k)];
                 
                 if length(data) < max(50,area-2*std(areas))
                     im_pred2(im_pred2 == k) = 0;
                     
                 elseif idx_label > 10 && length(data) > max(50,area)
                     n_cluster = 3;
                     label_cluster = zeros(n_cluster, length(data));
                     ll_neg = zeros(n_cluster, 1);

                     for s = 1 : n_cluster
                         GMModel = fitgmdist(data, s, 'Options', statset('MaxIter',1000), 'RegularizationValue', 0.001, 'Replicates', 10);
                         ll_neg(s) = GMModel.NegativeLogLikelihood / length(data);
                         label_cluster(s,:) = cluster(GMModel, data)';
                     end

                     diff_ll = diff(diff(ll_neg));
                     [diff_max, n_cluster] = max(diff_ll);
                     if diff_max > threshold 
                         n_cluster = n_cluster + 1;
                     else
                         n_cluster = 1;
                     end
                     label_cluster = label_cluster(n_cluster, :)';


                     NumLabels = max(label_cluster);
                     im_pred2(im_pred2 == k) = label_cluster + count_label;
                     count_label = count_label + NumLabels;
                 end
            end
            
            % ========== Superpixel segmentation ==========
%             area_superpixel = max(50, area-std(areas));
%             N_superpixels = round(numel(im_pred) / area_superpixel);
%             [L, N_superpixels] = superpixels(im, N_superpixels, 'Compactness', 10, 'Method', 'slic');
%             for k = 1 : N_superpixels
%                 area_k = sum(sum(L == k));
%                 mode_k = mode(im_pred2(L == k));
%                 if sum(sum(L == k & im_pred2 == mode_k)) > 0.75*area_k
%                     im_pred2(L == k) = mode_k;
%                 end
%             end
            
            % ========== Remove the nuclei with irragular size ==========
            NumLabels = max(im_pred2(:));
            for k = 1 : NumLabels
                data = [im_X(im_pred2 == k), im_Y(im_pred2 == k)];
                area_k = length(data);
                if count_label < 10
                    if area_k < max(50, area-2*std(areas))
                        im_pred2(im_pred2 == k) = 0;
                    end
                elseif count_label < 100
                    if area_k < max(50, area-2*std(areas)) || area_k > area+2.5*std(areas)
                        im_pred2(im_pred2 == k) = 0;
                    end
                else
                    if area_k < max(50, area-std(areas)) || area_k > area+1.5*std(areas)
                        im_pred2(im_pred2 == k) = 0;
                    end   
                end
            end
            
            im_output(:,:,1) = (mod(im_pred2, 2) == 0 & im_pred2 > 0);
            im_output(:,:,2) = (mod(im_pred2, 2) == 1);
            im_output(:,:,3) = (mod(im_pred2, 3) == 1);
            imwrite(im_output, strcat('test/', imageId, '_gmm.png'));
            
            % ========== Write mask labels to output file ==========
            for k = 1 : NumLabels
                if sum(sum(im_pred2 == k)) > 0
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
                    fprintf(f, strcat(imageId, ",", pixels, "\n"));
                end
            end
        end
        
        imageId = fileName;
        im = imread(strcat(imageId,'.png'));
        [y, x, ~] = size(im);
        [im_X, im_Y] = meshgrid(1:x, 1:y);
        grid = (im_X-1) * y + im_Y;
        im_pred = zeros(size(rgb2gray(im)));
        idx_label = 1;
        areas = zeros(1,500);
    end
    
    if i ~= height(M)+1
        labels = str2num(M(i,2).EncodedPixels{1});
        for j = 1 : length(labels) / 2
            idx = 2 * j -1;
            im_pred(labels(idx):labels(idx)+labels(idx+1)-1) = idx_label;
        end
        if sum(sum(im_pred == idx_label)) > 50
            areas(idx_label) = sum(sum(im_pred == idx_label));
        else
            im_pred(im_pred == idx_label) = 0;
        end
        idx_label = idx_label + 1;
    end
end
fprintf(repmat('\b',[1, length(verb)]));