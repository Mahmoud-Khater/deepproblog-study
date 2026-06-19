range(X,X,[X]).

range(M,N,[M|R]) :- M\=N, K is M +1, range(K,N,R).

