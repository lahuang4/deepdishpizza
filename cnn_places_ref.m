% updated 12:51am, run in MatConvNet directory (not examples directory!)

function [net, info] = cnn_places_ref(varargin)
% CNN_PLACES_REF Reference mini places CNN

run(fullfile(fileparts(mfilename('fullpath')),...
  'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile('data','places-ref-all') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','places') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 100 ;
opts.train.numEpochs = 21 ;
opts.train.continue = true ;
opts.train.gpus = [4] ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getPlacesImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-v7.3', '-struct', 'imdb') ;
end

disp('loaded imdb. starting training...');

net = sample_refNet_initial() ;
bopts = net.normalization ;
% bopts.transformation = 'stretch' ;
% bopts.averageImage = rgbMean ;
% bopts.rgbVariance = 0.1*sqrt(d)*v' ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

% [net, info] = cnn_train(net, imdb, @getBatch, ...
%     opts.train, ...
%     'val', find(imdb.images.set == 3)) ;
fn = getBatchSimpleNNWrapper(bopts) ;
[net, info] = cnn_train(net, imdb, fn, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
                            'prefetch', nargout == 0) ;
labels = imdb.images.label(batch) ;

% --------------------------------------------------------------------
function imdb = getPlacesImdb(opts)
% --------------------------------------------------------------------
% Prepare the imdb structure, returns image data with mean image subtracted
files = {'data', ...
         'development_kit'} ;

disp('Preparing image database...');

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

% data/places/ 
if ~exist(fullfile(opts.dataDir, 'images'), 'file')
  url = 'http://6.869.csail.mit.edu/fa15/challenge/data.tar.gz';
  fprintf('downloading %s\n', url) ;
  gunzip(url, opts.dataDir) ;
  untar(fullfile(opts.dataDir, 'data.tar'), opts.dataDir); % creates folder under data directory called data, with folders images/ and objects/ inside it
end

if ~exist(fullfile(opts.dataDir, 'development_kit'), 'file')
  url = 'http://6.869.csail.mit.edu/fa15/challenge/development_kit.tar.gz';
  fprintf('downloading %s\n', url) ;
  gunzip(url, opts.dataDir) ;
  untar(fullfile(opts.dataDir, 'development_kit.tar'), opts.dataDir); % creates folder under data directory called development_kit, with folders data/, evaluation/, and util/ inside it
end

opts.imgDir = fullfile(opts.dataDir, 'images');
opts.labelDir = fullfile(opts.dataDir, 'development_kit', 'data');

disp('Parsing category files and labels...');

if ~exist(fullfile(opts.dataDir, 'parsed.mat'), 'file')
  % trainNames is a thing with all the filename lines of train.txt in it
  % trainLabels is a thing with all the label lines of train.txt in it
  trainNames = [];
  trainLabels = [];
  f = fopen(fullfile(opts.labelDir, 'train.txt'));
  line = fgetl(f);
  while ischar(line)
    % 1st el -> trainNames, 2nd el -> trainLabels
    trainArray = strsplit(line,' ');
    trainNames = [trainNames; trainArray(1)];
    trainLabels = [trainLabels; str2double(trainArray(2))]; 
    line = fgetl(f);
  end
  fclose(f);

  % valNames is a thing with all the filename lines of val.txt in it
  % valLabels is a thing with all the label lines of val.txt in it
  valNames = [];
  valLabels = [];
  f = fopen(fullfile(opts.labelDir, 'val.txt'));
  line = fgetl(f);
  while ischar(line)
    valArray = strsplit(line,' ');
    valNames = [valNames; valArray(1)]; 
    valLabels = [valLabels; str2double(valArray(2))]; 
    line = fgetl(f);
  end
  fclose(f);
  savename = fullfile(opts.dataDir, 'parsed.mat');
  save(savename, 'trainNames', 'trainLabels', 'valNames', 'valLabels');
else
  savename = fullfile(opts.dataDir, 'parsed.mat');
  load(savename);
end



cats = [];
descrs = [];

f = fopen(fullfile(opts.labelDir, 'categories.txt'));
line = fgetl(f);
while ischar(line)
  arr = strsplit(line,' ');
  descrs = [descrs; arr(1)];
  cats = [cats; str2double(arr(2))]; 
  line = fgetl(f);
end
fclose(f);

imdb.classes.name = cats;
imdb.classes.description = descrs;
imdb.imageDir = fullfile(opts.dataDir, 'images') ;

fprintf('searching training images ...\n') ;
names = {} ;
labels = {} ;

catCount = 0;
for c = 1:length(descrs)
  subcat = descrs{c};
  ims = dir(fullfile(imdb.imageDir, 'train', subcat, '*.jpg')) ;
  names{end+1} = strcat(['train', filesep, subcat, filesep], {ims.name}) ;
  labels{end+1} = ones(1, numel(ims)) * catCount ;
  catCount = catCount + 1;
  fprintf('.') ;
  if mod(numel(names), 50) == 0, fprintf('\n') ; end
  %fprintf('found %s with %d images\n', d.name, numel(ims)) ;
end

names = horzcat(names{:}) ;
labels = horzcat(labels{:}) ;

if numel(names) ~= 100000
  warning('Found %d training images instead of 100,000. Dropping training set.', numel(names)) ;
  names = {} ;
  labels =[] ;
end

fprintf('Fetched %d training image names\n', numel(names));

imdb.images.id = 1:numel(names);
imdb.images.name = names;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels;



fprintf('searching validation images ...\n') ;
ims = dir(fullfile(imdb.imageDir, 'val', '*.jpg')) ;
names = sort({ims.name}) ;
labels = valLabels;

fprintf('Fetched %d validation image names\n', numel(ims));

if numel(ims) ~= 10000
  warning('Found %d instead of 10,000 validation images. Dropping validation set.', numel(ims))
  names = {} ;
  labels =[] ;
end

names = strcat(['val' filesep], names);

imdb.images.id = horzcat(imdb.images.id, (1:numel(names) + 1e7 - 1));
imdb.images.name = horzcat(imdb.images.name, names);
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels') ;

% asdfkljaksfsd
% 
% disp('Compiling images...');
% 
% numTrainIms = 10000;
% 
% %x1 = zeros(128,128,100000); % train images
% %x2 = zeros(128,128,10000); % val images
% x1 = zeros(128,128,3,numTrainIms); % train images
% x2 = zeros(128,128,3,10000); % val images
% y1 = trainLabels(1:numTrainIms)'; % train labels
% y2 = valLabels'; % val labels
% 
% if ~exist(fullfile(opts.dataDir, 'trainMatrices.mat'), 'file')
% %  for i = 1:length(trainNames)
%   for i = 1:numTrainIms
%     imgName = fullfile(opts.imgDir, trainNames(i));
%     imgName = imgName{1,1};
%     im=double(imread(imgName)) ;
%     x1(:,:,:,i)=im;
%   end
%   savename = fullfile(opts.dataDir, 'trainMatrices.mat');
%   save(savename, '-v7.3', 'x1', 'y1');
% else
%   savename = fullfile(opts.dataDir, 'trainMatrices.mat');
%   load(savename);
% end
% 
% if ~exist(fullfile(opts.dataDir, 'valMatrices.mat'), 'file')
%   for i = 1:length(valNames)
%     imgName = fullfile(opts.imgDir, valNames(i));
%     imgName = imgName{1,1};
%     im=double(imread(imgName)) ;
%     x2(:,:,:,i)=im;
%   end
%   savename = fullfile(opts.dataDir, 'valMatrices.mat');
%   save(savename, '-v7.3', 'x2', 'y2');
% else
%   savename = fullfile(opts.dataDir, 'valMatrices.mat');
%   load(savename);
% end
% 
% set = [ones(1,numel(y1)) 2*ones(1,numel(y2))];
% data = single(reshape(cat(4, x1, x2),128,128,3,1,[]));
% dataMean = mean(data(:,:,:,set == 1), 4);
% data = bsxfun(@minus, data, dataMean) ;
% 
% imdb.images.data = data ;
% imdb.images.data_mean = dataMean;
% imdb.images.labels = cat(2, y1, y2) ;
% imdb.images.set = set ;
% imdb.meta.sets = {'train', 'val', 'test'} ;
% imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;

