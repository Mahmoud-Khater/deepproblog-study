list_s([],0).

list_s([_|Tail],N) :- list_s(Tail,N1),N is N1 +1.
