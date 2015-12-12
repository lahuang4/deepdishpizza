% install MatConvNet first at http://www.vlfeat.org/matconvnet/

% load pre-trained model
load('categoryIDX.mat');
path_model = 'ref-drop2-net-epoch-35.mat';
path_avgs = 'img-avgs.mat';
load([path_model]) ;
load([path_avgs]) ;

% load and preprocess images
fnames = dir('data/places/images/test/*.jpg');
%results = {};
resultsfile = fopen('results-ref-drop2.txt', 'wt');
for pic = 1:length(fnames)
%for pic = 1:15
    im = imread(fullfile('data/places/images/test/', fnames(pic).name)) ;
    im_resize = imresize(im, net.normalization.imageSize(1:2)) ;
    im_ = single(im_resize) ; 
    for i=1:3
        im_(:,:,i) = im_(:,:,i)-avgs(i);
    end

    % change the last layer of CNN from softmaxloss to softmax
    net.layers{1,end}.type = 'softmax';
    net.layers{1,end}.name = 'prob';

    % run the CNN
    res = vl_simplenn(net, im_) ;

    scores = squeeze(gather(res(end).x)) ;
    [score_sort, idx_sort] = sort(scores,'descend') ;
%     figure, imagesc(im_resize) ;
%    disp(sprintf('scores for %s:', fnames(pic).name));
    if mod(pic, 100) == 0
        disp(sprintf('working on %s\n', fnames(pic).name));
    end
%    result = cell(5,1);
    fprintf(resultsfile, '%s %d %d %d %d %d\n', ['test/' fnames(pic).name], idx_sort(1), idx_sort(2), idx_sort(3), idx_sort(4), idx_sort(5));
%    for i=1:5
%        result{i} = sprintf('%s (%d), score %.3f', categoryIDX{idx_sort(i),1}, idx_sort(i), score_sort(i));
%        disp(sprintf('%s (%d), score %.3f', categoryIDX{idx_sort(i),1}, idx_sort(i), score_sort(i)));
%    end
%    results{pic} = result;
end

fclose(resultsfile);

%save 'results.mat' results
