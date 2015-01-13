function L = learn_L_fix_end_point(L, data_orig, data, varM, tar_set, imp_set, lambda)

fprintf('Begin optimization...\n');

MAX_ITER = 5000;

f_val = zeros(MAX_ITER,1);      % objective function value
n_cons = zeros(MAX_ITER,1);     % # violated constraints

step_size = 0.01;

best_f = 1e10;
best_L = [];

for iter = 1:MAX_ITER
    [f_val(iter), dL, n_cons(iter)] = lossfun_LMNN_end_point(L, data_orig, data, varM, lambda, tar_set, imp_set);
    
    if f_val(iter) < best_f
        best_f = f_val(iter);
        best_L = L;
    end
    
    best_f_val(iter) = best_f;
    
    if mod(iter,20) == 1
        fprintf('iter = %g \t step size = %f \t obj = %.4f \t ncons %g\n', iter, step_size, f_val(iter), n_cons(iter));
    end
    
    if iter > 200 && max(abs( diff( best_f_val(iter-50:iter) ) ) )< max(1e-8*abs(best_f_val(iter)),1e-3)
        fprintf('no more progress.. \niter = %g \t obj = %.4f\n', iter, f_val(iter));
        break;
    end
    
    % update L
    L = L - step_size * dL;
  
    if iter > 1
        if f_val(iter) < f_val(iter-1)
            step_size = step_size * 1.05;
        elseif f_val(iter) > f_val(iter-1)
            step_size = step_size * 0.5;
        end
        
        if step_size < 1e-10
            step_size = 1e-10;
        end
    end
end