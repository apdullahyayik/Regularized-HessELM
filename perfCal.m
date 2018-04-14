function measurements=perfCal(confusionMatrix)
        


% This function calculates detailed accuracy by class (like in WEKA)
% from confusion matrix binary classes.
% 
% USAGE
% measurements=perfCal(confusionMatrix)
%
% Please, also see crosstab function in MATLAB library
% confusion matrix can be calculated using built-in crosstab function 
% confusionMatrix=crosstab(label_index_expected, label_index_actual);
%
% Apdullah YAYIK 27 January 2017, Ankara
% apdullahyayik@gmail.com






% size control  
if size(confusionMatrix)==[1 1];
         confusionMatrix(1,2)=0 ;
         confusionMatrix(2,1)=0 ;
         confusionMatrix(2,2)=0; 
     elseif size(confusionMatrix)==[1 2];
         confusionMatrix(2,1)=0 ;
         confusionMatrix(2,2)=0;
     elseif size(confusionMatrix)==[2,1];
         confusionMatrix(1,2)=0 ;
         confusionMatrix(2,2)=0;
 end


% TP FP
% FN TN
TN.f=confusionMatrix(2,2);
TP.f=confusionMatrix(1,1);
FN.f=confusionMatrix(2,1);
FP.f=confusionMatrix(1,2);

% TN FN
% FP TP

% TN.s=confusionMatrix(1,1);
% TP.s=confusionMatrix(2,2);
% FN.s=confusionMatrix(1,2);
% FP.s=confusionMatrix(2,1);



Accuracy.f=(TP.f+TN.f)/sum(confusionMatrix(:));
% Accuracy.s=(TP.s+TN.s)/sum(confusionMatrix(:));

TruePositiveRate.f=TP.f/(TP.f+FN.f); % TPR=Recall=Sensitivity=Hit Rate
% TruePositiveRate.s=TP.s/(TP.s+FN.s); % TPR=Recall=Sensitivity=Hit Rate

% FalsePositiveRate.f=FP.f/(FP.f+TN.f); % FPR=fall-out=1-TNR
% FalsePositiveRate.s=FP.s/(FP.s+TN.s); % FPR=fall-out=1-TNR

Specificity.f=TN.f/(TN.f+FP.f); % TNR=TrueNegativeRate
% Specificity.s=TN.s/(TN.s+FP.s); % TNR=TrueNegativeRate

Presicion.f=TP.f/(TP.f+FP.f); % PPV=positive predicted value
% Presicion.s=TP.s/(TP.s+FP.s); % PPV=positive predicted value

% NegativePredictiveValue=TN/(TN+FN);
% Prevelance=(FN+TP)/sum(confusionMatrix(:));
FMeasure.f=2*((Presicion.f*TruePositiveRate.f)/(Presicion.f+TruePositiveRate.f)); % harmonic mean of precision and recall (= balanced f-score)
% FMeasure.s=2*((Presicion.s*TruePositiveRate.s)/(Presicion.s+TruePositiveRate.s)); % harmonic mean of precision and recall (= balanced f-score)

MCC.f=(TP.f*TN.f-FP.f*FN.f)/sqrt((TP.f+FP.f)*(TP.f+FN.f)*(TN.f+FP.f)*(TN.f+FN.f)); % Matthews correlation coefficient
% MCC.s=(TP.s*TN.s-FP.s*FN.s)/sqrt((TP.s+FP.s)*(TP.s+FN.s)*(TN.s+FP.s)*(TN.s+FN.s)); % Matthews correlation coefficient

% disp('TruePositiveRate, Presicion, FMeasure, Specificity, MCC, Accuracy');
measurements=100*[Presicion.f, Specificity.f, Accuracy.f]; %...
             % TruePositiveRate.s,  Presicion.s, FMeasure.s, Specificity.s, MCC.s, Accuracy.s];
% measurements=100*[Accuracy.f]; %..
end