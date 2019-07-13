function [test_data_tr,train_data_tr,test_data_ch,train_data_ch,label,Uro] =  Readout_subspace_rev(dataset,m,st,session,Trial_Label,trials,bins,etrials,dataset_e,m_e,st_e,Test,loc,ndim,V,rfin,expl,beta)
 %Test - 'Correct' or Error. For now keep it as 'Correct'
 %Test - 'Correct' or Error. For now keep it as 'Correct'
% ndim - No of dimensions to reduce the subspace to. This should be less
% than or equal to the number of neurons in the population
% V - uniform PCA components to project the data to minimize/optimize onto
rng('shuffle');
[training_label,train_trials,testing_label,test_trials] = MakeTrialSet(Trial_Label,trials,etrials,session,Test,loc);
% Initializing the training dataset with zeros
train_data = zeros(size(dataset,1),size(train_trials,2),size(dataset,3));
% Intializing the testing dataset with zeros
test_data = zeros(size(dataset,1),size(test_trials,2),size(dataset,3));
% Filling up train_data from dataset. All the values are z_scored using m,st to normalize the firing rate across neurons
for i_co = 1:size(dataset,1)
    
    train_data(i_co,:,:) = ((dataset(i_co,train_trials(i_co,:),:))-m(i_co))./st(i_co);
end
for i_co = 1:size(dataset,1)
    
    test_data(i_co,:,:) = ((dataset(i_co,test_trials(i_co,:),:))-m(i_co))./st(i_co);
end
% Computing the trial averaged sub-sampled responses for training and
% testing datasets
count = 1;label=[];
for i = loc
    ind_tr = find(training_label==i);
    ind_te = find(testing_label==i);
    for j = 1:50
        ind_sub = randsample(ind_tr,25);
        ind_sub_te = randsample(ind_te,25);
        train_data_m(:,count,:) = mean(train_data(:,ind_sub,:),2);
        test_data_m(:,count,:) = mean(test_data(:,ind_sub_te,:),2);
        count = count + 1;
        label = [label i];
    end
end

train_data_m(rfin,:,:)=[];
test_data_m(rfin,:,:)=[];
 n = dsearchn(expl,90);
% n = 61;
% Projecting the trial and time averaged training and testing data sets onto the
% principal components for different task epochs
% Delay 1
train_data_d1_ch = squeeze(mean(train_data_m(:,:,21:31),3))'*fliplr(V);
train_data_d1_ch=train_data_d1_ch(:,1:n);
% Delay 2 
train_data_d2_ch = squeeze(mean(train_data_m(:,:,45:55),3))'*fliplr(V);
train_data_d2_ch = train_data_d2_ch(:,1:n);
% Distractor presentation period
test_data_ch = reshape((reshape(test_data_m,size(test_data_m,1),[])'*fliplr(V))',[size(test_data_m,1) size(test_data_m,2) size(test_data_m,3)]);
test_data_ch = test_data_ch(1:n,:,:);
train_data_ch = reshape((reshape(train_data_m,size(train_data_m,1),[])'*fliplr(V))',[size(train_data_m,1) size(train_data_m,2) size(train_data_m,3)]);
train_data_ch = train_data_ch(1:n,:,:);
train_data_d1_ch_s = train_data_d1_ch - mean(train_data_d1_ch,1);

% % By shuffling the cluster labels for delay 2 to check if the minimization
% works for shuffled data
% new_lab = [7 6 4 1 5 2 3];
% label_tr=[];
% ex = 1:50;
% for w = 1:length(new_lab)
%     label_tr = [label_tr ex + (new_lab(w) - 1)*50];
% end
% train_data_d2_ch = train_data_d2_ch(label_tr,:);    

train_data_d2_ch_s = train_data_d2_ch - mean(train_data_d2_ch,1);
% Initializing U matrix for the optimization - U0 contains the initial
% value for Transformation matrix U and Pref
%  fval = 1.3;
% while fval > -0.45
U0 = rand(ndim,n);

% No bounds defined for the optimizaiton
A=[];
b = [];
Aeq = [];
beq=[];
lb=[];
ub=[];

% Computing the Fischer's LDA scatter for the info constraint
%[Sb_d1,Sw_d1,Sb_d2,Sw_d2] = fischers(train_data_d1_ch',train_data_d2_ch',label);
% Defining the optios for the minimization
options = optimoptions('fmincon','Algorithm','sqp','Display','Iter','SpecifyObjectiveGradient',true,'MaxFunctionEvaluations',10^10,'MaxIterations',10000,'StepTolerance',10^-6,'ConstraintTolerance',10^-2);
%

% Defining the equality and inequality constraints for hte optimization
 nlincon = @(U)constraints(U);
% Performing the optimization
 [U,fval,exitflag,output] = fmincon(@(U)objectivefn(U,train_data_d1_ch',train_data_d2_ch',train_data_d1_ch_s',train_data_d2_ch_s',label,beta),U0,A,b,Aeq,beq,lb,ub,nlincon,options);
 %end
% Extracting the Transformation matrix U
Uro = U;


for i = 1:size(dataset,3)
test_data_tr(:,:,i) = Uro*squeeze(test_data_ch(:,:,i));
train_data_tr(:,:,i) = Uro*squeeze(train_data_ch(:,:,i));
end


end
%% Function to create a matrix of training and testing labels
function [train_label,train_label_no,test_label,test_label_no] = MakeTrialSet(Trial_Label,trials,etrials,session,Test,loc)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function builds a matrix of trial indices and labels to build the
% training set. Trials from every session are split into training and
% testing pool. Further, two uniform distributions of 7 target labels was
% built to define the training and testing set's trial label. The length of
% these uniform distribution is defined by the number of trials used to
% train and test the decoder. For example, if the first trial label in the training set
% is target location 1, the function picks one trial for each neuron in the
% training pool with the target presented at target 1. Similarly, this is
% repeated for all the trial labels in the training and testing set.
% Inputs -
% Trial_Label - string - 'target' or 'distractor'
% trials - struct- passed on from the main function
% etrials - struct - passed on from the main function
% session - Nneurons x 2 matrix - passed on from the main function.
% Outputs -
% train_label - Ntraintrials x 1 matrix - uniform distribution of the 7
% trial_labels (target or distractor)
% train_label_no - Nneurons x Ntraintrials - indices of trials with
% trial label specified by train_label for all Nneurons.
% test_label and test_label_no are similar to train_label and
% train_label_no respectively but the trials to build test_label_no are
% chosen from the testing pool. Usually the length of train_label is set to
% 1500 and the length of test_label is set to 100.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deciphering the trial label to decode and assigning label the
% corresponding value, label = 1 when target is decoded and label = 2 when
% distractor is decoded.
if strcmp(Trial_Label,'Target') || strcmp(Trial_Label,'target')
    label=1;
elseif strcmp(Trial_Label,'Distractor') || strcmp(Trial_Label,'distractor')
    label=2;
end
% Dividing trials from each session into training and testing groups
for i_session = 1:length(trials)
    % Randomly picking 50% of the trials in a session to be under the
    % training pool
    
    train_num = randsample(length(trials(i_session).val),round(0.50*(length(trials(i_session).val))));
    % Assigning the other 50% to be the testing pool. While decoding error
    % trials it will be replaced with 1:length(etrials(i_session).val)
    if strcmp(Test,'Correct') || strcmp(Test,'correct')
        test_num = setdiff(1:length(trials(i_session).val),train_num);
    elseif strcmp(Test,'Error') || strcmp(Test,'error')
        test_num = 1:length(etrials(i_session).val);
    else
        test_num = setdiff(1:length(trials(i_session).val),train_num);
    end
    % Building the train_set with trial details for each session using
    % train_num
    train_set(i_session).val = trials(i_session).val(train_num);
    % Storing their original indices from trials
    train_set(i_session).orgind = train_num;
    % Storing the target labels of all the trials in train_set. Please note
    % that AssignTrialLabel function is specific for our dataset. This
    % function identifies the label (target or distractor location) for each 
    % trial. This needs to be modified if you are not using the dataset
    % used in Parthasarathy et al and subsequently the lines of code using
    % the output of AssignTrialLabel.
    train_set(i_session).tarlabel = AssignTrialLabel(train_set(i_session).val,label);
    % Similar variables for test_set
    if strcmp(Test,'Correct') || strcmp(Test,'correct')
        test_set(i_session).val = trials(i_session).val(test_num);
        test_set(i_session).orgind = test_num;
        test_set(i_session).tarlabel = AssignTrialLabel(test_set(i_session).val,label);
    elseif strcmp(Test,'Error') || strcmp(Test,'error')
        test_set(i_session).val = etrials(i_session).val;
        test_set(i_session).orgind = test_num;
        test_set(i_session).tarlabel = AssignTrialLabel(test_set(i_session).val,label);
    else
        test_set(i_session).val = trials(i_session).val(test_num);
        test_set(i_session).orgind = test_num;
        test_set(i_session).tarlabel = AssignTrialLabel(test_set(i_session).val,label);
    end
    train_num=[];test_num=[];
end
% Setting Ntraintrials
train_tr = 400;
% Setting Ntesttrials
test_tr = 400;
% count_sess stores the number of neurons recorded in each session
% Initializing count_sess
count_sess = zeros(length(trials),1);
for i_session = 1:length(trials)
    count_sess(i_session,1) = length(find(session(:,1)==i_session));
end
% Creates a uniform distribution of trial labels (based on Trial_Label)
% between 1 and 7 of length train_tr
if strcmp(Test,'Correct') || strcmp(Test,'correct')
    train_label = randsample(loc,train_tr,true);
    %Creates a uniform distribution of trial labels between 1 and 7 of length
    %test_tr
    test_label = randsample(loc,test_tr,true);
elseif strcmp(Test,'Error') || strcmp(Test,'error')
    train_label = randsample([2 3 5 6],train_tr,true);
    test_label = randsample([2 3 5 6],test_tr,true);
else
    train_label = randsample([2 3 5 6],train_tr,true);
    test_label = randsample([2 3 5 6],test_tr,true);
end
% id is a counter for the number of cells used in this analysis.
id = 1;
% i_session loops through the number of recorded sessions used in this
% analysis.
for i_session = 1:length(trials)
    % Checking if there are any neurons recorded in the session
    if count_sess(i_session,1)~=0
        % if there are neurons in that recorded session, loop through the
        % length of training set. For each value in train_label, find the
        % trials from the training pool for that recorded session
        % (represented as i_session) And repeat this for
        % every trial label in train_label
        for i_len = 1:train_tr
            % Initializing temporary variables
            train_label_tmp=[];ind=[];
            % Finding all the trials in the training pool with the i_len th
            % value of train_label.
            ind = find(train_set(i_session).tarlabel==train_label(i_len));
            % Sample with replacement from ind as many times as the number
            % of neurons in the recorded session (i_session)
%              train_label_tmp = (randsample(length(ind),count_sess(i_session),true));
             train_label_tmp = repmat(randsample(length(ind),1),1,count_sess(i_session));
            % Build train_label_no with the selected trials with
            % train_label_tmp. Note id is the cell counter and
            % id+count_sess(i_session,1) is the counter after adding all
            % the neurons recorded in i_session.
            train_label_no(id:id+count_sess(i_session,1)-1,i_len) = train_set(i_session).orgind(ind(train_label_tmp));
        end
        % Go through the same loop for test trials to build hte test_label_no
%         test_label = [];
%         for i_len = 1:length(test_set(i_session).val)
%             test_label_tmp = repmat(i_len,1,count_sess(i_session));
%             test_label_no(id:id+count_sess(i_session)-1,i_len) = test_set(i_session).orgind(test_label_tmp);
%             test_label = [test_label test_set(i_session).tarlabel(i_len)];
%         end
        for i_len = 1:length(test_label)
            ind=[];test_label_tmp=[];
            ind = find(test_set(i_session).tarlabel==test_label(i_len));
%             test_label_tmp = (randsample(length(ind),count_sess(i_session),true));
             test_label_tmp = repmat(randsample(length(ind),1),1,count_sess(i_session));
            test_label_no(id:id+count_sess(i_session)-1,i_len) = test_set(i_session).orgind(ind(test_label_tmp));
        end
    end
    id = id+count_sess(i_session);
end
clearvars -except train_label train_label_no test_label test_label_no
end
%% Constraints for the minimization
function [c,ceq] = constraints(x)
c=[];
% ceq = det(eye(size(x,1)) - x*x');
   ceq=[];
 end

%% Objective functions and gradients
function [fun,grad] = objectivefn(x,D1,D2,D1_s,D2_s,label,beta)
[Sb_d1,Sw_d1,Sb_d2,Sw_d2] = fischers(D1,D2,label);
Sb_d1_c = x*Sb_d1*x';
 Sw_d1_c = x*Sw_d1*x';
 Sb_d2_c = x*Sb_d2*x';
 Sw_d2_c = x*Sw_d2*x';
Dd1 = x*D1_s;
Dd2 = x*D2_s;
Dd = x*(D1-D2);

fun = -1*beta*norm(Dd1,'fro')/2 - beta*norm(Dd2,'fro')/2 + norm(Dd,'fro');
%fun = -1*beta*log(norm(Dd1,'fro')/(2*norm(D1-D2,'fro'))) - beta*log(norm(Dd2,'fro')/2*norm(D1-D2,'fro')) + log(norm(Dd,'fro')/norm(D1-D2,'fro'));
%- beta*log(norm(Sb_d1_c,'fro')/norm(Sw_d1_c,'fro')) - beta*log(norm(Sb_d2_c,'fro')/norm(Sw_d2_c,'fro'));
%fun =  log(norm(Dd,'fro')/norm(D1-D2,'fro'))- beta*log(norm(Sb_d1_c,'fro')/norm(Sw_d1_c,'fro')) - beta*log(norm(Sb_d2_c,'fro')/norm(Sw_d2_c,'fro'));

grad = ((-1*beta*x*D1_s*D1_s')) - ((beta*x*D2_s*D2_s')) + ((2*x*(D1-D2)*(D1-D2)'));
%- ((beta/norm(Sb_d1_c,'fro'))*4*x*Sb_d1*x'*x*Sb_d1) + ((beta/norm(Sw_d1_c,'fro'))*4*x*Sw_d1*x'*x*Sw_d1) - ((beta/norm(Sb_d2_c,'fro'))*4*x*Sb_d2*x'*x*Sb_d2) + ((beta/norm(Sw_d2_c,'fro'))*4*x*Sw_d2*x'*x*Sw_d2);
%grad =   ((x*(D1-D2)*(D1-D2)')/norm(Dd,'fro'))- ((beta/norm(Sb_d1_c,'fro'))*4*x*Sb_d1*x'*x*Sb_d1) + ((beta/norm(Sw_d1_c,'fro'))*4*x*Sw_d1*x'*x*Sw_d1) - ((beta/norm(Sb_d2_c,'fro'))*4*x*Sb_d2*x'*x*Sb_d2) + ((beta/norm(Sw_d2_c,'fro'))*4*x*Sw_d2*x'*x*Sw_d2);
end
%% Fischer's LDA
function [Sb_d1,Sw_d1,Sb_d2,Sw_d2] = fischers(D1,D2,label)

for i = 1:length(unique(label))
    ind_tr = find(label==i);
    mean_d1(:,i) = mean(D1(:,ind_tr),2);
    mean_d2(:,i) = mean(D2(:,ind_tr),2);
   
    cov_d1(:,:,i) = cov(D1(:,ind_tr)');
    cov_d2(:,:,i) = cov(D2(:,ind_tr)');
end
Sw_d1 = mean(cov_d1,3);
Sw_d2 = mean(cov_d2,3);
Sb_d1 = 0;Sb_d2 = 0;
for i = 1:length(unique(label))
    Sb_d1 = Sb_d1 + (mean_d1(:,i) - mean(mean_d1,2))*(mean_d1(:,i) - mean(mean_d1,2))';
    Sb_d2 = Sb_d2 + (mean_d2(:,i) - mean(mean_d2,2))*(mean_d2(:,i) - mean(mean_d2,2))';
end
Sb_d1 = Sb_d1/length(unique(label));
Sb_d2 = Sb_d2/length(unique(label));
end
