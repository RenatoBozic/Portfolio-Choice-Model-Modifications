function aprime=PortfolioCoiceModel4_1_aprimeFn(riskyshare,savings,u, r)

aprime=(1+r)*(1-riskyshare)*savings+(1+r+u)*riskyshare*savings;

end
