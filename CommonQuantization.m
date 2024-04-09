function  Q = CommonQuantization(data, num)  
	% common quantization the input data into num feature.
    % input is [0,1] rang.
    
    Q = fix(data .* num) + 1;
    Q(Q>num) = num;
    Q(Q<1) = 1;
end
