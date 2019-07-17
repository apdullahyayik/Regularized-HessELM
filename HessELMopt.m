function [TrainingTime, TestingTime, trmeasurements, temeasurements, mse] = HessELMopt(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction, onset, offset)


% Cao et. al [1] introduced implementation of in ELM with SVD method, 
% implementation of in ELM with hessenbergdecomposition method was introduced with this code (lines 122-157).

% [1] J. Cao, K. Zhang, M. Luo, C. Yin, and X. Lai, "Extreme learning machine and adaptive sparse representation for image classification", Neural networks, vol. 81, pp. 91102, 2016
% This function requies MSE.m and Perfcal.m


% please kindly refer 
% https://arxiv.org/abs/1907.05888 "Regularized HessELM and Inclined Entropy Measurement for 
% Congestive Heart Failure Prediction"


REGRESSION=0;
CLASSIFIER=1;


%%%%%%%%%%% Load training dataset
train_data=TrainingData_File;
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=TestingData_File;
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break;
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;
    
    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break;
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
    
end                                                 %   end if of Elm_Type

%%%%%%%%%%% Calculate weights & biases
start_time_train=cputime;

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid
        H = 1 ./ (1 + exp(-tempH));
        save H H
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
    case {'cos'}
        %%%%%%%% cosinus function
        H = cos(tempH);
    case {'tan'}
        %%%%%%%% cosinus function
        H = tan(tempH);
    case{'e2'}
        %%Eistein function 1
        H=E2(tempH);
        
    case ('tanh')
        H=(exp(tempH)-exp(-tempH))/(exp(tempH)+exp(-tempH));
        
    case('relu')
        H=tempH .* (tempH > 0);
        
        %%%%%%%% More activation functions can be added here
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H


%%%%%%%%%%%%%%%%%%%%%%%%%% Apdullah Yayık
lcandidate=onset:1:offset;
mse = zeros(1,length(lcandidate));
temeasurements=zeros(length(lcandidate),3);

OutputWeight_C = cell(1,length(lcandidate));
for cnum=1:length(lcandidate)
    OutputWeight_C{1,cnum}=zeros(NumberofHiddenNeurons, 2);
end
lamda=zeros(1,length(lcandidate));
H=H';
for ii=1:length(lcandidate)
    ii
    
    if ii==offset
        lamda(ii)=0;
    else
        lamda(ii)=exp(-lcandidate(ii));
    end
    %     [U, D, V]=svd(H);
    %     invsvdopt=(inv(V*D'*U'*U*D*V'+lamda(ii)*eye(size(H'*H,1))))*H'; % H*invsvdopt;
    % P=Q, U=A in alg 1 of paper
    [P, A]=hess(H*H');
      invhessopt=H'*P*inv(A+lamda(ii)*eye(size(A,1)))*P';
%     invhessopt=H'*P*((A\P')+(lamda(ii)*eye(size(A,1))));
%     invhessopt=H'*(((A*P')+(lamda(ii)*eye(size(A,1))))\P'); % fastest one 

    OutputWeight_C{1,ii}=invhessopt*T';
    y=(H*OutputWeight_C{1,ii})';
    mse(ii)=MSE(H,invhessopt,T,y);
    OutputWeight=OutputWeight_C{1,ii};
end
    OutputWeight=OutputWeight_C{1,mse==min(mse)};
    H=H';

 %%%%%%%%%%%%%%%% Apdullah Yayk
    
    end_time_train=cputime;
    TrainingTime=end_time_train-start_time_train  ;      %   Calculate CPU time (seconds) spent for training ELM
    
    %%%%%%%%%%% Calculate the training accuracy
    Y=(H'*OutputWeight)';
    
    
    if Elm_Type == REGRESSION
        TrainingAccuracy=sqrt(mse(T - Y)) ;              %   Calculate training accuracy (RMSE) for regression case
    end
    %     clear H;
    
      
    %%%%%%%%%%% Calculate the output of testing input
    start_time_test=cputime;
    tempH_test=InputWeight*TV.P;
    clear TV.P;             %   Release input of testing data
    ind=ones(1,NumberofTestingData);
    BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
    tempH_test=tempH_test + BiasMatrix;
    switch lower(ActivationFunction)
        case {'sig','sigmoid'}
            %%%%%%%% Sigmoid
            H_test = 1 ./ (1 + exp(-tempH_test));
        case {'sin','sine'}
            %%%%%%%% Sine
            H_test = sin(tempH_test);
        case {'hardlim'}
            %%%%%%%% Hard Limit
            H_test = hardlim(tempH_test);
        case {'tribas'}
            %%%%%%%% Triangular basis function
            H_test = tribas(tempH_test);
        case {'radbas'}
            %%%%%%%% Radial basis function
            H_test = radbas(tempH_test);
            
        case ('tanh')
            H_test=(exp(tempH_test)-exp(-tempH_test))/(exp(tempH_test)+exp(-tempH_test));
            
        case('relu')
            H_test=tempH_test .* (tempH_test > 0);
            %%%%%%%% More activation functions can be added here
    end
    TY=(H_test' * OutputWeight)';                       %   TY: the actual output of the testing data
    end_time_test=cputime;
    TestingTime=end_time_test-start_time_test;          %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
    
    if Elm_Type == REGRESSION
        TestingAccuracy=sqrt(mse(TV.T - TY))            %   Calculate testing accuracy (RMSE) for regression case
    end
    
    if Elm_Type == CLASSIFIER
        %%%%%%%%%% Calculate training & testing classification accuracy
        MissClassificationRate_Training=0;
        MissClassificationRate_Testing=0;
        
        for i = 1 : size(T, 2)
            [x, trlabel_index_expected(i)]=max(T(:,i));
            [x, trlabel_index_actual(i)]=max(Y(:,i));
            % if label_index_actual~=label_index_expected
            %    MissClassificationRate_Training=MissClassificationRate_Training+1;
            % end
        end
        
        trconfusionMatrix=crosstab(trlabel_index_expected, trlabel_index_actual);
        
        %size control
        if size(trconfusionMatrix)==[1 1];
            trconfusionMatrix(1,2)=0 ;
            trconfusionMatrix(2,1)=0 ;
            trconfusionMatrix(2,2)=0;
        elseif size(trconfusionMatrix)==[1 2];
            trconfusionMatrix(2,1)=0 ;
            trconfusionMatrix(2,2)=0;
        elseif size(trconfusionMatrix)==[2,1];
            trconfusionMatrix(1,2)=0 ;
            trconfusionMatrix(2,2)=0;
        end
        
        trmeasurements=perfCal(trconfusionMatrix);
        
        
        
        
        % TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)
        for i = 1 : size(TV.T, 2)
            [x, telabel_index_expected(i)]=max(TV.T(:,i));
            [x, telabel_index_actual(i)]=max(TY(:,i));
            %if label_index_actualx~=label_index_expectedx
            %   MissClassificationRate_Testing=MissClassificationRate_Testing+1;
            %end
        end
        % TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)
        
        
        teconfusionMatrix=crosstab(telabel_index_expected, telabel_index_actual);
        
        %size control
        if size(teconfusionMatrix)==[1 1];
            teconfusionMatrix(1,2)=0 ;
            teconfusionMatrix(2,1)=0 ;
            teconfusionMatrix(2,2)=0;
        elseif size(teconfusionMatrix)==[1 2];
            teconfusionMatrix(2,1)=0 ;
            teconfusionMatrix(2,2)=0;
        elseif size(teconfusionMatrix)==[2,1];
            teconfusionMatrix(1,2)=0 ;
            teconfusionMatrix(2,2)=0;
        end
        
        temeasurements=perfCal(teconfusionMatrix);
        
        
    end
    


