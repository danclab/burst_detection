function z=gaus2d(x, y, mx, my, sx, sy)
    z=exp(-((x - mx).^2. / (2. * sx^2.) + (y - my).^2. / (2. * sy^2.)));
end
