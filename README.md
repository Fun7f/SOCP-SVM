# SOCP-SVM
Part code of SOCP-SVM
%输入数据集 正例正确率k1 负例正确率k2 输出AUC
function dataAUC = SocpSvm(Data_train,Data_test,Train_label,Test_label,K,C)

    [Train_x,Train_y] = size(Data_train);
    Train_A = [];
    Train_B = [];
    Train_A_x = 1;
    Train_B_x = 1;
    for i = 1:Train_x
        if Train_label(i,1) == 1
           Train_A(Train_A_x,:) = Data_train(i , 1 : Train_y);
           Train_A_x = Train_A_x + 1;
        else
            Train_B(Train_B_x,:) = Data_train(i , 1 :Train_y );
            Train_B_x = Train_B_x +1;
        end
    end
    %确定特征数
    [m,n] = size(Train_A);
    %计算正例平均值
    A_mean1 = mean(Train_A);
    %计算负例平均值
    A_mean2 = mean(Train_B);
    %计算正例协方差矩阵
    A_cov1 = cov(Train_A);
    %计算负例协方差矩阵
    A_cov2 = cov(Train_B);
    %正例协方差矩阵分解
    S1 = chol(A_cov1 + 10^(-10)*eye(n));
    %负例协方差矩阵分解
    S2 = chol(A_cov2 + 10^(-10)*eye(n));


    
    %计算K1(正例)  ?
    K1 = sqrt(K.k1/(1-K.k1));
    %计算K2（负例）
    K2 = sqrt(K.k2/(1-K.k2));
    
    %A_mean1正例平均值，A_mean2负例平均值
   %S1正例协方差矩阵分解，S2负例协方差矩阵分解   

     z=Train_A_x;%正例数                    
     f=Train_B_x;%负例数
     
     

     %b,+松弛,-松弛,t,w,u0,u1,v0,v1
                                                       %软soft socp 1
    A = [zeros(z,1)+1/K1  1/K1*eye(z)  zeros(z,f)    zeros(z,1)  zeros(z,1)+A_mean1/K1  zeros(z,1)-1  zeros(z,n)  zeros(z,1)   zeros(z,n);...       
         zeros(f,1)-1/K2  zeros(f,z)  1/K2*eye(f)    zeros(f,1)  zeros(f,1)-A_mean2/K2  zeros(f,1)    zeros(f,n)  zeros(f,1)-1 zeros(f,n);...
         zeros(n,1)       zeros(n,z)  zeros(n,f)     zeros(n,1)  (S1)'                  zeros(n,1)    -eye(n)     zeros(n,1)   zeros(n,n);...       
         zeros(n,1)       zeros(n,z)  zeros(n,f)     zeros(n,1)  (S2)'                  zeros(n,1)    zeros(n,n)  zeros(n,1)   -eye(n)];
    At = sparse(A);
    b = sparse([zeros(z,1)+1/K1;zeros(f,1)+1/K2;zeros(n,1);zeros(n,1)]);
    c=[0;zeros(z+f,1)+2^C.c1;1;zeros(n,1);0;zeros(n,1);0;zeros(n,1)];
    c=sparse(c);
    K.f = 1;
    K.l= z + f;
    K.q = [n+1 n+1 n+1];
    X = sedumi(At,b,c,K);
    w = X(z+f+3:z+f+n+2,1);
    b = X(1,1);
     
     


    
    [Dt_x,Dt_y] = size(Data_test);
    %将测试标签改为 +1 -1
    for j = 1:Dt_x
        if Test_label(j,1) ~= 1
            Test_label(j,1) = -1;
        end
    end


    

    
    
    Pre_result = [];
    for i = 1:Dt_x
        Pre_result(i,1) = sign(w'*Data_test(i,:)' + b);
        i = i +1; 
    end

    
%     dataAUC = AUC(Pre_result,Test_label);
%    dataAUC = Gmean(Pre_result,Test_label);
     dataAUC = Fmeasure(Pre_result,Test_label);
end
