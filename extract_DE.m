%% 

clear,clc

load_path = 'E:/数据集汇总/SEED/ExtractedFeatures1/session1/';
save_path = 'E:/数据集汇总/SEED/PR-PL-main/feature_LDS_de/';
label_path = 'E:/数据集汇总/SEED/ExtractedFeatures1/label';
load(label_path)

label_v = label;

file_list = dir(load_path);
file_list(1:2)=[];

for sub_i = 1:length(file_list)
    disp(['subject>',file_list(sub_i).name])
    S = load([load_path,file_list(sub_i).name]);
    DE = [];
    labelAll = [];
    for ii = 1:15
        eval(['data=','S.de_LDS',num2str(ii),';']);
        DE = cat(2,DE,data);
        labelAll = [labelAll;zeros(size(data,2),1)+label_v(ii)];
    end
    
    feature = DE;
    label = labelAll;
    
    feature = permute(feature,[2,1,3]);
    feature = reshape(feature,[],310);
    
    save([save_path,file_list(sub_i).name],'feature','label');
    
end
    