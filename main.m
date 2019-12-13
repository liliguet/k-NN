%Main index file 
clear;
close all;

% Calculation of time of execution
tic
%------------------------
load('dataset/liver.mat');
%load('dataset/wine.mat');
%load('dataset/climate.mat');
%load('dataset/auditrisk.mat');
%load('dataset/inno.mat');
%load('dataset/ad.mat');


%% Generating random permutation for division into test and train data
nrows = size(A, 1);
randrows = randperm(nrows);

%% KNN
X = A;
k = [2, 3, 4, 5]; %kappa
for K = 1 : 1 %K-NN
    for fold = 2 : 5
        for chunk = 1 : fold
            chunksize = floor(nrows/fold);
            x = (chunk - 1) * chunksize + 1;
            y = chunk * chunksize;
            testdata = X(randrows(x:y), :);
            if chunk == 1
                traindata = X(randrows(y + 1:end), :);
            elseif chunk == fold
                traindata = X(randrows(1 : x-1), :);
            else
                traindata = X(randrows(1, x-1:y+1, end), :);
            end
            currentacc_ED = knnclassifier_ED(traindata, testdata, K);
            currentacc_MD = knnclassifier_MD(traindata, testdata, K);
            currentacc_FD = knnclassifier_FD(traindata, testdata, K);
            currentacc_DMD = knnclassifier_DMD(traindata, testdata, K);
            s_ED(chunk) = currentacc_ED;
            s_MD(chunk) = currentacc_MD;
            s_FD(chunk) = currentacc_FD;
            s_DMD(chunk) = currentacc_DMD;
        end
        meanaccuracy_ED(fold - 1, K) = mean(s_ED);
        meanaccuracy_MD(fold - 1, K) = mean(s_MD);
        meanaccuracy_FD(fold - 1, K) = mean(s_FD);
        meanaccuracy_DMD(fold - 1, K) = mean(s_DMD);
        
        e_ED(fold - 1,K) = std(s_ED);   
        e_MD(fold - 1,K) = std(s_MD);
        e_FD(fold - 1,K) = std(s_FD);  
        e_DMD(fold - 1,K) = std(s_DMD);    
    end
%     subplot(3,3, K);
%     errorbar(k, out, e); 
end
toc
