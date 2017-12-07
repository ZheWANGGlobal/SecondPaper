function equalRate = computeEqualRate(d,f)
if f(1) > f(end)
    d = d(end:-1:1);
    f = f(end:-1:1);
end

equalRate=-1;
for j=1:size(f,2)
    if (1 - f(j)) == d(j)
        equalRate=d(j);
        break
    elseif (1 - f(j)) < d(j)
        if j == 1
            equalRate = d(j)/(f(j)+d(j));
        elseif f(j) == f(j-1)
            equalRate = 1-f(j);
        elseif d(j) == d(j-1)
            equalRate = d(j);
        else
            m = (d(j)-d(j-1))/(f(j)-f(j-1));
            equalRate = (m-f(j-1)*m + d(j-1))/(1+m);
        end
        break
    end
end
if equalRate==-1
    equalRate=d(size(d,2));
end
