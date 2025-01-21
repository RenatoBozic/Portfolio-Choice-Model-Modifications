function F=PortfolioCoiceModel1_ReturnFn(savings,riskyshare,a,z1,z2,z3,w,sigma,agej,Jr,pension,kappa_j)

F=-Inf;
if agej<Jr % If working age
    c=w*kappa_j*z1*z2+a-savings; % Add z here
else % Retirement
    c=pension*z3+a-savings;
end

if c>0
    F=(c^(1-sigma))/(1-sigma); % The utility function
end

end
