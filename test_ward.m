addpath('./Liblinear');

rand_pre={'_1','1','_2','2','_3','3'};
train_nums=[500,1000,1500,2000,2500,3000];

for k=1:6
    load ./ward/rand_idx/o_label.mat  %label
    load(['./ward/rand_idx/',rand_pre{k},'rand.mat']);   %rand_idx
    train_num=train_nums(k);
    fprintf('========= train_num: %d =========\n\n',train_num)
    train_ind=rand_idx(1:train_num);
    test_ind=rand_idx(train_num+1:end);
    
    %% fc7 %%
    load ./ward/mat/ward_fc7.mat      %feats
    
    ftrain=feats(train_ind,:);
    TrnLabels=label(train_ind);
    ftest=feats(test_ind,:);
    TestLabels=label(test_ind);
    
    [tl ind_tl]=sort(TrnLabels);
    TrnLabels=tl;
    ftrain=ftrain(ind_tl,:);
    
    models = train(TrnLabels, sparse(ftrain), '-s 1 -q');
    
    [xLabel_est, accuracy, decision_values] = predict(TestLabels,sparse(ftest), models, '-q');
    fprintf('fc7 accuracy: %.2f\n\n',accuracy(1));
    dec3=decision_values;
    
    %% fc6 %%
    load ./ward/mat/ward_fc6.mat      %feats
    
    ftrain=feats(train_ind,:);
    TrnLabels=label(train_ind);
    ftest=feats(test_ind,:);
    TestLabels=label(test_ind);
    
    [tl ind_tl]=sort(TrnLabels);
    TrnLabels=tl;
    ftrain=ftrain(ind_tl,:);
    
    models = train(TrnLabels, sparse(ftrain), '-s 1 -q');
    
    [xLabel_est, accuracy, decision_values] = predict(TestLabels,sparse(ftest), models, '-q');
    fprintf('fc6 accuracy: %.2f\n\n',accuracy(1));
    dec2=decision_values;
    
    %% pool5 %%
    load ./ward/mat/ward_pool5.mat      %feats
    
    ftrain=feats(train_ind,:);
    TrnLabels=label(train_ind);
    ftest=feats(test_ind,:);
    TestLabels=label(test_ind);
    
    [tl ind_tl]=sort(TrnLabels);
    TrnLabels=tl;
    ftrain=ftrain(ind_tl,:);
    
    models = train(TrnLabels, sparse(ftrain), '-s 1 -q');
    
    [xLabel_est, accuracy, decision_values] = predict(TestLabels,sparse(ftest), models, '-q');
    fprintf('pool5 accuracy: %.2f\n\n',accuracy(1));
    dec1=decision_values;
    
    %% fusing pool5,fc6,fc7
    dec=0.7*dec1+0.2*dec2+0.1*dec3;
    [d i]=max(dec,[],2);
    i=i-1;
    cnt=length(find(i==TestLabels));
    fprintf('final accuracy: %.2f\n\n',cnt*1.0/length(TestLabels)*100);
    
end

