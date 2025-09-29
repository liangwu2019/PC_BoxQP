function [z,iter,max_iter] = PC_BoxQP_Matlab(H,h,epsilon)
n = length(h);
z = zeros(n,1);
norm_h = norm(h);
if norm_h<=epsilon
    return;
else
    lambda = 1/(4*sqrt(2)*norm_h);
    H = 2*lambda*H;
    gamma = 1.0 - lambda*h;
    theta = 1.0 + lambda*h;
    phi = ones(n,1);
    psi = ones(n,1);
    max_iter = ceil(-0.5*log(2*n/epsilon)/log(1-2^0.25/4/sqrt(2*n)))+1;
    for iter=1:max_iter
        %% Predictor Step
        duality_gap = gamma'*phi + theta'*psi;
        if duality_gap<=epsilon
            break;
        end
        gamma_divide_phi = gamma./phi;
        theta_divide_psi = theta./psi;
        L = chol(H+diag(gamma_divide_phi)+diag(theta_divide_psi),'lower');
        
        delta_z = L'\(L\(gamma-theta));
        delta_gamma = gamma_divide_phi.*delta_z - gamma;
        delta_theta = -theta_divide_psi.*delta_z - theta;
        delta_phi = -delta_z;
        delta_psi = delta_z;

        % Computing the step size in the Predictor Step
        mu = duality_gap/(2*n);
        delta_comp = [delta_gamma.*delta_phi; delta_theta.*delta_psi];
        delta_mu = sum(delta_comp)/(2*n);
        delta_comp = delta_comp - delta_mu;
        norm_delta_comp = norm(delta_comp);
        step_size = min(0.5, sqrt(mu/8/norm_delta_comp));

        % Updating with the adaptive step size
        z = z + step_size * delta_z;
        gamma = gamma + step_size * delta_gamma;
        theta = theta + step_size * delta_theta;
        phi = phi + step_size * delta_phi;
        psi = psi + step_size * delta_psi;

        %% Corrector Step
        gamma_divide_phi = gamma./phi;
        theta_divide_psi = theta./psi;
        L = chol(H+diag(gamma_divide_phi)+diag(theta_divide_psi),'lower');
        mu_c = (gamma'*phi + theta'*psi)/(2*n);
        
        delta_z = L'\(L\(mu_c./psi-mu_c./phi+gamma-theta));
        delta_gamma = gamma_divide_phi.*delta_z + mu_c./phi - gamma;
        delta_theta = -theta_divide_psi.*delta_z +mu_c./psi - theta;
        delta_phi = -delta_z;
        delta_psi = delta_z;

        % Updating with step size = 1
        z = z + delta_z;
        gamma = gamma + delta_gamma;
        theta = theta + delta_theta;
        phi = phi + delta_phi;
        psi = psi + delta_psi;
    end
end