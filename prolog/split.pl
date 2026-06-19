split(X,0,[],X):-!.

split([H|Tail],N,[H|Y],M):-  split(Tail,N1,Y,M), N1 is N-1.  
