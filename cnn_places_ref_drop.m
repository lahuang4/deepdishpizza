% Run in MatConvNet directory (not examples directory!)

function [net, info] = cnn_places_ref_drop(varargin)
% CNN_PLACES_REF_DROP Reference mini places CNN with dropout

run(fullfile(fileparts(mfilename('fullpath')),...
  'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile('data','places-ref-drop') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','places') ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
opts.train.batchSize = 100 ;
opts.train.numEpochs = 90 ;
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

disp('Loaded imdb. starting training...');

net = cnn_places_ref_drop_init() ;
bopts = net.normalization ;

imageStatsPath = fullfile(opts.dataDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

[v,d] = eig(rgbCovariance) ;
bopts.transformation = 'stretch' ;
bopts.averageImage = rgbMean ;
bopts.rgbVariance = 0.1*sqrt(d)*v' ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

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
  % trainNames will contain all the filename lines of train.txt in it
  % trainLabels will contain all the label lines of train.txt in it
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

  % valNames will contain all the filename lines of val.txt in it
  % valLabels will contain all the label lines of val.txt in it
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

fprintf('Searching training images ...\n') ;
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

fprintf('Searching validation images ...\n') ;
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

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 101: end);
bs = 256 ;
fn = getBatchSimpleNNWrapper(opts) ;
for t=1:bs:numel(train)
  batch_time = tic ;
  batch = train(t:min(t+bs-1, numel(train))) ;
  fprintf('Collecting image stats: batch starting with image %d ...', batch(1)) ;
  temp = fn(imdb, batch) ;
  z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{t} = mean(temp, 4) ;
  rgbm1{t} = sum(z,2)/n ;
  rgbm2{t} = z*z'/n ;
  batch_time = toc(batch_time) ;
  fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
