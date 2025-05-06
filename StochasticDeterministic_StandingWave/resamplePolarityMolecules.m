% Gaussian smear of discretely located particles
% Input: molecule distribution
% Output: molecule concentration at spatial intervals dxa
%
% Last updated: 1/31/2020
% Calina Copos

function [x,xConcentration,yConcentration] = resamplePolarityMolecules(posxxx,posyyy,K1,K2,L,Na)

% Gaussian smear the molecules
epsilon=0.01;                     % variance squared (default)
x=linspace(0,L,Na);
a0=sqrt(pi)/(sqrt(2*pi*epsilon)); % area of normal distribution

xsmear=zeros(length(x),K1);
for i=1:K1
    xsmear(:,i) = exp(-(x-posxxx(i)).^2/(2*epsilon))./(sqrt(2*pi*epsilon));
end

ysmear=zeros(length(x),K2);
for i=1:K2
    ysmear(:,i) = exp(-(x-posyyy(i)).^2/(2*epsilon))./(sqrt(2*pi*epsilon));
end

% normalize
xsmear = xsmear/a0;
ysmear = ysmear/a0;

xConcentration = sum(xsmear,2);
yConcentration = sum(ysmear,2);

end
