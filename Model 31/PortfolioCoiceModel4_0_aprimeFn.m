function aprime=PortfolioCoiceModel4_0_aprimeFn(savings,riskyshare,u, r)

aprime=(1+r)*(1-riskyshare)*savings+(1+r+u)*riskyshare*savings;

end
