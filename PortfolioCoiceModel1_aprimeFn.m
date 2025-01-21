function aprime=LifeCycleModel31_11A_Stock_disaster_ret1_aprimeFn(savings,riskyshare,u, r)

aprime=(1+r)*(1-riskyshare)*savings+(1+r+u)*riskyshare*savings;

end
