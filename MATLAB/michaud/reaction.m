% reaction.m
%
% compute the reaction term for the michaud model
%
function R = reaction(RT,RD,F, k0, k1, k2, k3, k4, alpha, beta)
    R = (k0*alpha*(k1*(RT.^3))./(1+k2*(RT.^2))).*RD - (k3 + k4*(1+beta)*F).*RT;
end