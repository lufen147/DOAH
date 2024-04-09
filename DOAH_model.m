function  [DOAH1, DOAH2, DOAH3, DOAH4, DOAH5] = DOAH_model(img, X, cn1, cn2, cn3, CSO, CSI, CSS)
    % im: RGB image
    % X: deep map
    % DOAH: Deep aggregating histogram
    % Authors: F. Lu 2023. 
    
   %% color quantization
    
    % RGB to HSV color space
    hsv = single(rgb2hsv(uint8(img)));
    
    % color map
    colorMap = ColorQuantization(hsv, cn1, cn2, cn3);
    
    % V(HSV) orientation map
    edgeOri = EdgeOrientation(hsv(:,:,3));
    oriMap  = CommonQuantization(edgeOri, CSO);
    
    % V(HSV) intensity map
    intenMap = CommonQuantization(hsv(:,:,3), CSI); 

    %% deep orientation detection
    
    [hei, wid, ~] = size(img);
    oriEnergyFrequency = single(zeros(hei, wid, 4));
    for h = 2:hei-1
        for w = 2:wid-1
            energy = 1;
            %color
            if colorMap(h,w) == colorMap(h-1,w-1) && colorMap(h,w) == colorMap(h+1,w+1)
                oriEnergyFrequency(h,w,1) = oriEnergyFrequency(h,w,1) + energy;
            end
            if colorMap(h,w) == colorMap(h,w-1) && colorMap(h,w) == colorMap(h,w+1)
                oriEnergyFrequency(h,w,2) = oriEnergyFrequency(h,w,2) + energy;
            end
            if colorMap(h,w) == colorMap(h-1,w) && colorMap(h,w) == colorMap(h+1,w)
                oriEnergyFrequency(h,w,3) = oriEnergyFrequency(h,w,3) + energy;
            end
            if colorMap(h,w) == colorMap(h-1,w+1) && colorMap(h,w) == colorMap(h+1,w-1)
                oriEnergyFrequency(h,w,4) = oriEnergyFrequency(h,w,4) + energy;
            end
            
            %orientation
            if oriMap(h,w) == oriMap(h-1,w-1) && oriMap(h,w) == oriMap(h+1,w+1)
                oriEnergyFrequency(h,w,1) = oriEnergyFrequency(h,w,1) + energy;
            end
            if oriMap(h,w) == oriMap(h,w-1) && oriMap(h,w) == oriMap(h,w+1)
                oriEnergyFrequency(h,w,2) = oriEnergyFrequency(h,w,2) + energy;
            end
            if oriMap(h,w) == oriMap(h-1,w) && oriMap(h,w) == oriMap(h+1,w)
                oriEnergyFrequency(h,w,3) = oriEnergyFrequency(h,w,3) + energy;
            end
            if oriMap(h,w) == oriMap(h-1,w+1) && oriMap(h,w) == oriMap(h+1,w-1)
                oriEnergyFrequency(h,w,4) = oriEnergyFrequency(h,w,4) + energy;
            end
            
            %intensity
            if intenMap(h,w) == intenMap(h-1,w-1) && intenMap(h,w) == intenMap(h+1,w+1)
                oriEnergyFrequency(h,w,1) = oriEnergyFrequency(h,w,1) + energy;
            end
            if intenMap(h,w) == intenMap(h,w-1) && intenMap(h,w) == intenMap(h,w+1)
                oriEnergyFrequency(h,w,2) = oriEnergyFrequency(h,w,2) + energy;
            end
            if intenMap(h,w) == intenMap(h-1,w) && intenMap(h,w) == intenMap(h+1,w)
                oriEnergyFrequency(h,w,3) = oriEnergyFrequency(h,w,3) + energy;
            end
            if intenMap(h,w) == intenMap(h-1,w+1) && intenMap(h,w) == intenMap(h+1,w-1)
                oriEnergyFrequency(h,w,4) = oriEnergyFrequency(h,w,4) + energy;
            end
        end
    end
    oriEnergyFrequency = max(oriEnergyFrequency, [], 3);
    
    %% extract deep orientation features
    
    [hei2,wid2] = size(X, [1,2]);
    
    X_object = sum(X, 3);
    X_object = X_object ./ sum(X_object, 'all');
    
    X_objectEnergy = X .* X_object;
    
    X_object = imresize(X_object, [hei,wid]);
    X_object(X_object<0) = 0;
    oriEnergy = oriEnergyFrequency .* X_object;
    
    oriEnergy = imresize(oriEnergy, [hei2,wid2]);
    X_dosEnergy = X .* oriEnergy;  

    %% features aggregation representation
    
    histogram1 = sum(X, [1 2]);
    histogram2 = sum(X, [1 2]);
    histogram3 = sum(X_objectEnergy, [1 2]);
    histogram4 = sum(X_dosEnergy, [1 2]);
    histogram5 = histogram1 + histogram2 + histogram3;
    
    DOAH1 = histogram1;
    DOAH2 = histogram2;
    DOAH3 = histogram3;
    DOAH4 = histogram4;
    DOAH5 = histogram5;
    
end