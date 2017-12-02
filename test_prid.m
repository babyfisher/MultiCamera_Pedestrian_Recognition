addpath('./Liblinear');

load ./prid/rand_ind_all.mat rand_ind
load ./prid/o_label.mat data_y
label=data_y;

for train_num=[2000,5000:5000:25000]
    fprintf('========= train_num: %d =========\n\n',train_num)
    train_ind=rand_ind(1:train_num);
    test_ind=rand_ind(train_num+1:end);
    %% fc7 %%
    load ./prid/mat/prid_fc7.mat      %feats
    
    ftrain=feats(train_ind,:);
    TrnLabels=label(train_ind);
    ftest=feats(test_ind,:);
    TestLabels=label(test_ind);
    
    tmp=[0:199]';
    tmp1=unique(TrnLabels);
    dif=setdiff(tmp,tmp1);
    for i=1:length(dif)
        idx=find(label==dif(i));
        ftrain=[ftrain;feats(idx(1),:)];
        TrnLabels=[TrnLabels;dif(i)];
    end
    
    [tl ind_tl]=sort(TrnLabels);
    TrnLabels=tl;
    ftrain=ftrain(ind_tl,:);
    
    models = train(TrnLabels, sparse(ftrain), '-s 1 -q');
    
    [xLabel_est, accuracy, decision_values] = predict(TestLabels,sparse(ftest), models, '-q');
    fprintf('fc7 accuracy: %.2f\n\n',accuracy(1));
    dec3=decision_values;
    
    %% fc6 %%
    load ./prid/mat/prid_fc6.mat      %feats
    
    ftrain=feats(train_ind,:);
    TrnLabels=label(train_ind);
    ftest=feats(test_ind,:);
    TestLabels=label(test_ind);
    
    tmp=[0:199]';
    tmp1=unique(TrnLabels);
    dif=setdiff(tmp,tmp1);
    for i=1:length(dif)
        idx=find(label==dif(i));
        ftrain=[ftrain;feats(idx(1),:)];
        TrnLabels=[TrnLabels;dif(i)];
    end
    
    [tl ind_tl]=sort(TrnLabels);
    TrnLabels=tl;
    ftrain=ftrain(ind_tl,:);
    
    models = train(TrnLabels, sparse(ftrain), '-s 1 -q');
    
    [xLabel_est, accuracy, decision_values] = predict(TestLabels,sparse(ftest), models, '-q');
    fprintf('fc6 accuracy: %.2f\n\n',accuracy(1));
    dec2=decision_values;
    
    %% pool5 %%
    load ./prid/mat/prid_pool5.mat      %feats
    
    ftrain=feats(train_ind,:);
    TrnLabels=label(train_ind);
    ftest=feats(test_ind,:);
    TestLabels=label(test_ind);
    
    tmp=[0:199]';
    tmp1=unique(TrnLabels);
    dif=setdiff(tmp,tmp1);
    for i=1:length(dif)
        idx=find(label==dif(i));
        ftrain=[ftrain;feats(idx(1),:)];
        TrnLabels=[TrnLabels;dif(i)];
    end
    
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

