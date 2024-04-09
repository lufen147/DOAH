function  edgeori = EdgeOrientation(data)
    % extract edge orientation.
    
    [~, ~, gv, gh]= edge(data, 'sobel');
    gvh = gv ./ gh;
    gvh(isnan(gvh)) = 0;
    theta = fix(90 + atan(gvh) .* 180 ./ pi);
    edgeori = theta ./ 180;
end
