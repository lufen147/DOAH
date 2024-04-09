% Authors: F. Lu, G-H. Liu, 2021. 
%% load general config
addpath('../revisitop-master');
clear; tic; warning off;

opts.run.data_temp = './data_temp_DOAH/';
opts = im_config(opts);

% A_data = ["roxford5k", "rparis6k"];
A_data = ["Oxford5K", "Paris6K", "Holidays_upright"];
B_net = ["vgg16"];
% B_net = ["alexnet", "resnet101", "densenet201", "mobilenetv2", "efficientnetb0"];

%% calculating raw descriptors

for a_i = 1:size(A_data,2)
    for b_i = 1:size(B_net,2)
        opts.datasets.name = lower(A_data(a_i));
        opts.features.net = lower(B_net(b_i));
        
       % % set dataset parameter
        
        filePath = "../../Data/datasets/" + opts.datasets.name + "/";
        if opts.datasets.name == "oxford5k"
            filePath="../../Data/datasets/" + opts.datasets.name + "/oxbuild_images/";
        end
        if opts.datasets.name == "paris6k"
            filePath="../../Data/datasets/" + opts.datasets.name + "/paris_images/*/";
        end

        if ismember(opts.datasets.name, ["roxford5k", "rparis6k"])
            data_root = fullfile('../../data');
            cfg = configdataset(opts.datasets.name, fullfile(data_root, 'datasets/'));
            file_num_all = [cfg.nq, cfg.n];
        %     file_num_all = [cfg.nq];
        else
            filename = dir(filePath + "*.jpg");
            [file_num_all, temp] = size(filename);
        end

        % % load pre-train network
        net = importdata('../../data/networks/imagenet/vgg16.mat');
        if strcmp(opts.features.net, 'vgg16')
%             net = vgg16;
            layer1 = 'pool5';
            dim1 = 512;
        end

        % % aggregate and save feature

        for file_num = file_num_all
           % % set parameter
            cn1 = 6;
            cn2 = 3;
            cn3 = 3;
            CSC = cn1 * cn2 * cn3;
            CSO = 18;
            CSI = 64;
            Hcnum = CSC + CSO + CSI;
            CSS = 32;
            
            MidH = single(zeros(file_num, dim1));
            MF = MidH;
            SMF = MidH;
            OSMF = MidH;
            CatF = MidH;
            
            name_list = "";

           % % prepair to start aggregating
           disp([char(datetime), ' aggregating from ', char(opts.datasets.name), ' using ', char(opts.features.net), ' on (', num2str(file_num), '):       ']);
           
           for i = 1:file_num
               if ismember(opts.datasets.name, ["roxford5k", "rparis6k"])
                    if file_num <= 70
                        imdata = crop_qim(imread(cfg.qim_fname(cfg, i)), cfg.gnd(i).bbx);
                    else
                        imdata = imread(cfg.im_fname(cfg, i));
                    end 
               else
                   imdata = imread([filename(i).folder, '/', filename(i).name]);
               end

               % % pre-process the size of image
                if size(imdata,3)==1
                    imdata = cat(3,imdata,imdata,imdata);
                end
                img = single(imdata);

                if ismember(opts.datasets.name, ["holidays", "holidays_upright"])
                    img = imresize(img, 0.5);
                end

                [h, w, ~] = size(img);
                img1 = imresize(img, [112, 112]);

                if min(h, w) < 224
                    img_resize = imresize(img, [224, 224]);
                else
                    img_resize = img;
                end

                % % get images name list
                if ismember(opts.datasets.name, ["roxford5k", "rparis6k"])
                    if file_num <= 70
                        name = cfg.qimlist(i);
                    else
                        name = cfg.imlist(i);
                    end
                    name_list(i) = name{1};
                else
                    ssplit = strsplit(filename(i).name, {'.'});
                    name = ssplit(1);
                    name_list(i) = name{1};
                end

                % % get feature representation
                
                X = activations(net, img_resize, layer1, 'OutputAs', 'channels');
%                 X = activations(net, img_resize, layer1, 'OutputAs', 'channels', 'ExecutionEnvironment', 'cpu');
                
                [H1, H2, H3, H4, H5] = DOAH_model(img1, X, cn1, cn2, cn3, CSO, CSI, CSS);
                
                MidH(i,:) = H1;
                MF(i,:)   = H2;
                SMF(i,:)  = H3;
                OSMF(i,:) = H4;
                CatF(i,:) = H5;

                fprintf(1,'\b\b\b\b\b\b%6d', i);

            end
            % % gather various feature vector
            toc
            if file_num <= 70
                MHDF2.qname = name_list;
                MHDF2.qMidH = MidH;
                MHDF2.qMF = MF;
                MHDF2.qSMF = SMF;
                MHDF2.qOSMF = OSMF;
                MHDF2.qCatF = CatF;
            else
                MHDF2.name = name_list;
                MHDF2.MidH = MidH;
                MHDF2.MF = MF;
                MHDF2.SMF = SMF;
                MHDF2.OSMF = OSMF;
                MHDF2.CatF = CatF;
            end
        end
        
        % save data to .mat

        save([opts.run.data_temp, 'MHDF1_', opts.datasets.name{1}, '_', opts.features.net{1}], 'MHDF2');
    end
end
disp(datetime);

%% 3. post-process and testing evaluation

opts.run.data_temp = './data_temp_DOAH/';

% % Baseline
% C_dim = [512];
% opts.features.pipeline_model = 'norm';
% % Feature  = 'MF'; qFeature = 'qMF';
% isIDF = 0;

% % raw_DOAH
% C_dim = [512];
% opts.features.pipeline_model = 'norm';
% Feature  = 'OSMF'; qFeature = 'qOSMF';
% isIDF = 0;

% % pIDF_DOAH
% C_dim = [512];
% opts.features.pipeline_model = 'norm';
% Feature  = 'OSMF'; qFeature = 'qOSMF';
% isIDF = 1;

% % DOAH
% C_dim = [16, 32, 64, 128, 256, 512];
C_dim = [128, 512];
% opts.features.pipeline_model = 'pcasw';
% opts.features.pipeline_model = 'pcacw';
opts.features.pipeline_model = 'pcaaw';
Feature  = 'OSMF'; qFeature = 'qOSMF';
isIDF = 1;

for a_i = 1:size(A_data, 2)
    for b_i = 1:size(B_net, 2)
        for c_i = 1:size(C_dim, 2)
            opts.datasets.name = lower(A_data(a_i));
            opts.features.net = lower(B_net(b_i));
            dim = C_dim(c_i);
            if ismember(opts.datasets.name, ["roxford5k", "rparis6k"])
                file_num_all = [5000, 70];
            else
                file_num_all = 5000;
            end
            
            load([opts.run.data_temp, 'MHDF1_', opts.datasets.name{1}, '_', opts.features.net{1}]);
            
            for file_num = file_num_all
                if file_num <= 70
                    MHDF3.qname = MHDF2.qname;
                    MHDF = MHDF2.(qFeature);
                else
                    MHDF3.name = MHDF2.name;
                    MHDF = MHDF2.(Feature);
                end
                
                % % PCA self dataset
                train = MHDF2.(Feature);
                if ismember(opts.features.pipeline_model, ["pcacw"])
                    % % PCA cross dataset
                    if ismember(opts.datasets.name, ["rparis6k"])
                        MHDF4 = load([opts.run.data_temp, 'MHDF1_', 'roxford5k', '_', opts.features.net{1}]);
                        train = MHDF4.MHDF2.(Feature);
                    end
                    if ismember(opts.datasets.name, ["roxford5k"])
                        MHDF4 = load([opts.run.data_temp, 'MHDF1_', 'rparis6k', '_', opts.features.net{1}]);
                        train = MHDF4.MHDF2.(Feature);
                    end
                    if ismember(opts.datasets.name, ["oxford5k"])
                        MHDF4 = load([opts.run.data_temp, 'MHDF1_', 'paris6k', '_', opts.features.net{1}]);
                        train = MHDF4.MHDF2.(Feature);
                    end
                    if ismember(opts.datasets.name, ["paris6k", "holidays_upright"])
                        MHDF4 = load([opts.run.data_temp, 'MHDF1_', 'oxford5k', '_', opts.features.net{1}]);
                        train = MHDF4.MHDF2.(Feature);
                    end
                end
                if ismember(opts.features.pipeline_model, ["pcaaw"])
                    % % pca augmentation dataset
                    MHDF4 = load([opts.run.data_temp, 'MHDF1_', 'oxford5k', '_', opts.features.net{1}]);
                    train1 = MHDF4.MHDF2.(Feature);
                    MHDF4 = load([opts.run.data_temp, 'MHDF1_', 'paris6k', '_', opts.features.net{1}]);
                    train2 = MHDF4.MHDF2.(Feature);
                    MHDF4 = load([opts.run.data_temp, 'MHDF1_', 'holidays_upright', '_', opts.features.net{1}]);
                    train3 = MHDF4.MHDF2.(Feature);
                    train = [train1; train2; train3];
                end
                                
                if isIDF == 1
                    MHDF  = IDF(MHDF);
                    train = IDF(train);
                end
                if ismember(opts.features.pipeline_model, ["norm"]) 
                    MHDF  = normalize(MHDF, 2, 'norm');
                end
                if ismember(opts.features.pipeline_model, ["pcacw", "pcasw", "pcaaw"])
                    MHDF = PCA_whitening(MHDF, train, dim);
                end
                if file_num <= 70
                    MHDF3.qMHDF = MHDF;
                else
                    MHDF3.MHDF = MHDF;
                end
            end
            
            fdata = MHDF3;
%             save([opts.run.data_temp, 'MHDF3_', opts.datasets.name{1}], 'MHDF3');

            % % evaluation
            opts.features.dimension = size(MHDF, 2);
            opts.datasets.name = opts.datasets.name{1};
            opts.features.net = opts.features.net{1};
            if ismember(opts.datasets.name, ["roxford5k", "rparis6k"])
                opts.match.metric = 2;
                opts.match.qe_positive = 0;
                example_evaluate;
            else
                opts.match.metric = 2;
                opts.match.qe_positive = 0;
                opts.match.queryratio = 1;
                opts.match.precisiontop = 10;
                report_eval = im_evaluation(opts, fdata);
            end
        end
    end
end
disp(datetime);