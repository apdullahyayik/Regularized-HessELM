function z=MSE(H,invhessopt,T,y)


% This function gives meanm squared error explained in 
% https://arxiv.org/abs/1907.05888 "Regularized HessELM and Inclined Entropy Measurement for 
% Congestive Heart Failure Prediction"
% Section 2.6 
% Algorithm 1 Computing weights with regularized HessELM 

%
% Apdullah YAYIK 27 January 2017, Turkey
% for questions: apdullahyayik@gmail.com


% 

R=sum(y-T);
HAT=H*invhessopt;
S = diag(eye(size(HAT, 2))-HAT);
z=mean(sum((R'./S')^2));

end




