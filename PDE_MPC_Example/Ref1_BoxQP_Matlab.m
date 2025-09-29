function [z,run_time] = Ref1_BoxQP_Matlab(H,h,epsilon)
[nz,~] = size(h);
h_infinite_norm = max(abs(h));
z = zeros(nz,1);
run_time = 0;
tic
if h_infinite_norm<=epsilon
    return
else
    lambda = 1/sqrt(nz+1);
    eta = 1/((sqrt(2)+1)*sqrt(2*nz)+1);
    H_hat = 2*lambda*H/h_infinite_norm;
    gamma = 1 - lambda*h/h_infinite_norm;
    theta = 1 + lambda*h/h_infinite_norm;
    phi = ones(nz,1);
    psi = ones(nz,1);
    tau = 1/(1-eta);
    max_iter = ceil(-0.5*log(2*nz/epsilon)/log(sqrt(2*nz)/(sqrt(2*nz)+sqrt(2)-1))) + 1;
    for k=1:max_iter
        tau = (1-eta)*tau;
        temp_r1 = 2*(sqrt(gamma./phi)*tau-gamma);
        temp_r2 = 2*(sqrt(theta./psi).*tau-theta);
        delta_z = (H_hat+diag(gamma./phi)+diag(theta./psi))\(temp_r2-temp_r1);
        delta_gamma = temp_r1 + gamma./phi.*delta_z;
        delta_theta = temp_r2 - theta./psi.*delta_z;
        delta_phi = -delta_z;
        delta_psi = delta_z;
        z = z + delta_z;
        gamma = gamma + delta_gamma;
        theta = theta + delta_theta;
        phi = phi + delta_phi;
        psi = psi + delta_psi;
    end    
end
run_time = toc;