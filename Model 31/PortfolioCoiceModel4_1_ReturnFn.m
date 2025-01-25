function F=PortfolioCoiceModel4_1_ReturnFn(savings,a,z1,z2,e1,e2,w,sigma,agej,Jr,pension,kappa_j)

F=-Inf;
if agej<Jr 
    c=w*kappa_j*z1*e1+a-savings; % Working ages
else 
    c=pension*z2*e2+a-savings; % Retirement ages
end

if c>0
    F=(c^(1-sigma))/(1-sigma); % The utility function
end

end
