function o=overlap(a,b)
    o=(a(1)<=b(1) && b(1)<=a(2)) || (b(1)<=a(1) && a(1)<=b(2));
end