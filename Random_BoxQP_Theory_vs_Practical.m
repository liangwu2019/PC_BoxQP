clear,close,clc
rng(2025);
dims = [100,200,300,400,500,600,700,800,900,1000];
num_exp = 20;

cr = 1e3;
epsilon = 1e-6;
Direct_Iters = zeros(1,length(dims));
Theory_Iters = zeros(1,length(dims));
Pratical_Iters_Avg = zeros(1,length(dims));
Pratical_Iters_Upper = zeros(1,length(dims));
Pratical_Iters_Low = zeros(1,length(dims));
pratical_iters = zeros(1,length(num_exp));
for i=1:length(dims)
    n = dims(i);
    Direct_Iters(i) = ceil(-0.5*log(2*n/epsilon)/log(sqrt(2*n)/(sqrt(2*n)+sqrt(2)-1)))+1;
    Theory_Iters(i)  = ceil(-0.5*log(2*n/epsilon)/log(1-0.2348/sqrt(2*n)))+1;
    %%
    for j=1:num_exp
        v = [cr;(cr-1)*rand(n-2,1)+1;1];
        U = orth(randn(n,n));
        H = U*diag(v)*U';
        H = (H+H')/2;
        h = 1e3 * (1-2*rand(n,1));
        [z_PC_BoxQP,pratical_iter,~] = PC_BoxQP_Matlab(H,h,epsilon);
        pratical_iters(j) = pratical_iter; 
    end
    Pratical_Iters_Avg(i) = mean(pratical_iters);
    Pratical_Iters_Upper(i) = max(pratical_iters);
    Pratical_Iters_Low(i) = min(pratical_iters);
end
%%
figure(1)
semilogy(dims,Pratical_Iters_Avg,'b-','LineWidth',3)
hold on
semilogy(dims,Theory_Iters,'LineWidth',3)
semilogy(dims,Direct_Iters,'LineWidth',3)
legend('Practical: Algorithm 1 ','Theoretical: Algorithm 1','Ref [1]');
xlabel('Problem dimension: n'),ylabel('Number of iterations')
grid on

figure(2)
x_fill = [dims, fliplr(dims)];
y_fill = [Pratical_Iters_Upper, fliplr(Pratical_Iters_Low)];
fill(x_fill, y_fill, [207  155  255]/255,'EdgeColor','none');
hold on
plot(dims,Pratical_Iters_Avg,'b-','LineWidth',3)
xlabel('Problem dimension: n'),ylabel('Number of iterations')
grid on