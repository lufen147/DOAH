function opts = im_config(opts)
    % config file of project on your assign data sets
    % hyper parameters config
    % Authors: F. Lu. 2020.

    opts.file.tempdir = tempdir;
    opts.file.root = fileparts(mfilename('fullpath'));  % get this project's root, opts, a new define struct type
    root = opts.file.root;          % define a simply variable for using below
    opts.file.name = mfilename;     % get this script file name, not include extend name
    opts.file.format_txt = '.txt';  % config txt file format, noted that "."
    opts.file.format_jpg = '.jpg';  % config jpg file format, noted that "."
    opts.file.format_mat = '.mat';  % config mat file format, noted that "."
    opts.file.format_npy = '.npy';  % config npy file format, noted that "."
    opts.file.format_dat = '.dat';  % config dat file format, noted that "."
    opts.file.format_cvs = '.cvs';  % config cvs file format, noted that "."
    opts.file.format_common = '*';  % config the images name, * is any name

    opts.run.temp = 'none';
    opts.datasets.temp = 'none';
    opts.extract.temp = 'none';
    opts.features.temp = 'none';
    opts.match.temp = 'none';
    opts.param.temp = 'none';

    opts.run.useGPU = 0;                % config the useGPU logic for runing computing, 0 use CPU, 1 use GPU
    if ~isfield(opts.run, 'load_aggregate')
        opts.run.load_aggregate = 1;    % config the load_aggregate_features logic run, 0 not run,1 run
    end
    if ~isfield(opts.run, 'epoch')
        opts.run.epoch = 1;                 % if need running of epoch, option of [1 2 3 ...].
    end
    if ~isfield(opts.run, 'data_temp')
        opts.run.data_temp = './data_temp/';       % generate temp mat data
    end
    if ~exist(opts.run.data_temp, 'dir')
        mkdir(opts.run.data_temp);
    end

    opts.extract.batchsize = 1;     % config the batch images number input to CNN while extract feature, option of [1, 64, 128, 256]

    opts.datasets.datapath = '../../data/';   % config the data file path

    datapath = opts.datasets.datapath;        % define a simply variable for using below
    if ~isfield(opts.datasets, 'name')
        opts.datasets.name = 'oxford5k';        % config datasets name, one of [oxford5k, paris6k, roxford5k, rparis6k, oxford105k, paris106k, holidays, ukbench, flickr100k]
    end
    if ~isfield(opts.datasets, 'eachclassnum')
        opts.datasets.eachclassnum = 100;        % config each class images of datasets
    end

    if strcmp(opts.datasets.name, 'oxford5k')
    %     opts.datasets.image_path = fullfile(fileparts(root), 'datasets', 'Oxford5K', 'oxbuild_images');
        opts.datasets.image_path = [datapath, '/datasets/Oxford5K/oxbuild_images/'];  % config the images datasets orgin path
        opts.datasets.gt_path = [datapath, '/datasets/Oxford5K/gt_files_170407/'];    % config the images datasets orgin path
        opts.features.path = [datapath, '/features/oxford5k/fc7_vgg16/'];
        opts.features.query_path = [datapath, '/features/oxford5k/pool5_queries/'];      % config the query images feature save path
        opts.match.rank_path = [datapath, '/features/oxford5k/rank_file/'];             % config the optional path to save query image match ranked ouput
    end
    if strcmp(opts.datasets.name, 'roxford5k')
        opts.datasets.image_path = [datapath, '/datasets/Oxford5K/oxbuild_images/'];  % config the images datasets orgin path
        opts.datasets.gt_path = [datapath, '/datasets/roxford5k/'];    % config the images datasets orgin path
        opts.features.path = [datapath, '/features/roxford5k/pool5_vgg16/'];                    % config the images feature save path
        opts.features.query_path = [datapath, '/features/roxford5k/pool5_queries/'];      % config the query images feature save pathopts.match.rank_path = [datapath, '/features/roxford5k/rank_file/'];             % config the optional path to save query image match ranked ouput
    end
    if strcmp(opts.datasets.name, 'paris6k')
        opts.datasets.image_path = [datapath, '/datasets/Paris6K/paris_images/'];     % config the images datasets orgin path
        opts.datasets.gt_path = [datapath, '/datasets/Paris6K/gt_files_120310/'];     % config the images datasets orgin path
        opts.features.path = [datapath, '/features/paris6k/fc7_vgg16/'];
        opts.features.query_path = [datapath, '/features/paris6k/pool5_queries/'];       % config the query images feature save path
        opts.match.rank_path = [datapath, '/features/paris6k/rank_file/'];              % config the optional path to save query image match ranked ouput
    end
    if strcmp(opts.datasets.name, 'rparis6k')
        opts.datasets.image_path = [datapath, '/datasets/Paris6K/paris_images/'];     % config the images datasets orgin path
        opts.datasets.gt_path = [datapath, '/datasets/rparis6k/'];    % config the images datasets orgin path
        opts.features.path = [datapath, '/features/rparis6k/pool5_vgg16/'];                    % config the images feature save path
        opts.features.query_path = [datapath, '/features/rparis6k/pool5_queries/'];      % config the query images feature save path
        opts.match.rank_path = [datapath, '/features/rparis6k/rank_file/'];             % config the optional path to save query image match ranked ouput
    end
    if strcmp(opts.datasets.name, 'holidays_upright')
        opts.datasets.image_path = [datapath, '/datasets/Holidays_upright/images/'];                 % config the images datasets orgin path
        opts.datasets.gt_path = [datapath, '/datasets/Holidays_upright/images/'];                    % config the images datasets orgin path
        opts.features.path = [datapath, '/features/holidays_upright/pool5_vgg16/'];                   % config the images feature save path
        opts.features.query_path = [datapath, '/features/holidays_upright/pool5_queries/'];     % config the query images feature save path
        opts.match.rank_path = [datapath, '/features/holidays_upright/rank_file/'];           % config the optional path to save query image match ranked ouput
    end

    opts.features.query_crop = 0;        % config the query image extract form, value 1 is crop, value 0 is full image (not crop)
    if ~isfield(opts.features, 'net')
        opts.features.net = 'vgg16';    % config net model frame, one of [vgg16, caffe, matconvnet, matconvnet_dag]
    end
    opts.features.net_layer = 'pool5';              % config images feature extracted from which net layter
    if ~isfield(opts.features, 'dimension')
        opts.features.dimension = 128;              % config the images feature extracted dimension
    end
    if ~isfield(opts.features, 'cross_model')
        opts.features.cross_model = 'DOAH';        % config calculate cross model, one of [mhdf], mhdf: mid- and high deep feature
    end
    if ~isfield(opts.features, 'pipeline_model')
        opts.features.pipeline_model = 'none';      % config pipeline model such as Dimension reduction model, one of [none, norm, pca, pca_whitening, pca_whitening_self, pca_relja, pca_whitening_relja, pca_pairs]
    end

    if ~isfield(opts.match, 'qe_positive')
        opts.match.qe_positive = 0;      % config image retrieval query expansion positive top R, if do not use query expansion that put R=0
    end
    if ~isfield(opts.match, 'qe_negative')
        opts.match.qe_negative = 0;      % config image retrieval query expansion negative bottom R, if do not use that put R=0
    end
    if ~isfield(opts.match, 'metric')
        opts.match.metric = 2;           % config the metric (measure) method. option: L1 Manhattan distance:1, L2 Euclidean:2, Canberra distance:3, Correlation similarity:4, Cosine similarity:5, Histogram intersection:6, Inner product distance:7, Chebyshev distance:8,  
    end
    if ~isfield(opts.match, 'precisiontop')
        opts.match.precisiontop = 12;    % config the precision (including recall, auc) top N images.
    end
    if ~isfield(opts.match, 'queryratio')
        opts.match.queryratio = 0.1;             % config each query ratio of datasets
    end

    if ~isfield(opts.param, 'c_k1')
        opts.param.c_k1 = 1;        % config parameter for train model, pre-set
    end
    if ~isfield(opts.param, 'c_k2')
        opts.param.c_k2 = 1;
    end
    if ~isfield(opts.param, 'c_k3')
        opts.param.c_k3 = 1;
    end

    save('opts', 'opts');       % save and use for some module loading
end
