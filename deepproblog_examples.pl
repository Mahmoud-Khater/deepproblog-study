% DeepProbLog Examples
% Extracted from "Programming in DeepProbLog" paper
% Source: https://bitbucket.org/problog/deepproblog

% ============================================================================
% Example 1: MNIST Single-Digit Addition (T1)
% ============================================================================
% Neural predicate for digit classification
% nn(m_digit,[X],Y,[0,...,9]) :: digit(X,Y).

% Addition predicate using digit classification
addition(X,Y,Z) :- 
    digit(X,X2), 
    digit(Y,Y2), 
    Z is X2 + Y2.

% ============================================================================
% Example 2: MNIST Multi-Digit Addition (T2)
% ============================================================================
% Convert list of digit images to number
number([],Result,Result).
number([H|T],Acc,Result) :-
    digit(H,Nr),
    Acc2 is Nr + 10 * Acc,
    number(T,Acc2,Result).

number(X,Y) :- number(X,0,Y).

% Multi-digit addition
multi_addition(X,Y,Z) :- 
    number(X,X2), 
    number(Y,Y2), 
    Z is X2 + Y2.

% ============================================================================
% Example 3: ProbLog Alarm Bayesian Network
% ============================================================================
0.1::burglary.
0.5::at_home(mary).
0.2::earthquake.
0.4::at_home(john).

alarm :- earthquake.
alarm :- burglary.
calls(X) :- alarm, at_home(X).

% ============================================================================
% Example 4: Forth Addition (T5)
% ============================================================================
% Neural predicates for result and carry
% nn(m_result,[D1,D2,Carry],Y,[0,...,9])::result(D1,D2,Carry,Y).
% nn(m_carry,[D1,D2,Carry],Y,[0,1])::carry(D1,D2,Carry,Y).

hole(I1,I2,Carry,NewCarry,Result) :-
    result(I1,I2,Carry,Result),
    carry(I1,I2,Carry,NewCarry).

add([],[],[C],C,[]).
add([H1|T1],[H2|T2],C,Carry,[Digit|Res]) :-
    add(T1,T2,C,NewCarry,Res),
    hole(H1,H2,NewCarry,Carry,Digit).

forth_addition(L1,L2,C,[Carry|Res]) :- 
    add(L1,L2,C,Carry,Res).

% ============================================================================
% Example 5: Forth Sorting (T6)
% ============================================================================
% Neural predicate for swap decision
% nn(m_swap,[X,Y])::swap(X,Y).

hole(X,Y,X,Y) :- \+swap(X,Y).
hole(X,Y,Y,X) :- swap(X,Y).

bubble([X],[],X).
bubble([H1,H2|T],[X1|T1],X) :-
    hole(H1,H2,X1,X2),
    bubble([X2|T],T1,X).

bubblesort([],L,L).
bubblesort(L,L3,Sorted) :-
    bubble(L,L2,X),
    bubblesort(L2,[X|L3],Sorted).

forth_sort(L,L2) :- bubblesort(L,[],L2).

% ============================================================================
% Example 6: CLUTRR Family Relations (T11)
% ============================================================================
% Background knowledge for family relations
grandchild(X,Y) :- child(X,Z), child(Z,Y).
grandchild(X,Y) :- so(X,Z), grandchild(Z,Y).
grandchild(X,Y) :- grandchild(X,Z), sibling(Z,Y).

grandparent(X,Y) :- parent(X,Z), parent(Z,Y).
grandparent(X,Y) :- sibling(X,Z), grandparent(Z,Y).

child(X,Y) :- child(X,Z), sibling(Z,Y).
child(X,Y) :- so(X,Z), child(Z,Y).

parent(X,Y) :- sibling(X,Z), parent(Z,Y).
parent(X,Y) :- child(X,Z), grandparent(Z,Y).

sibling(X,Y) :- child(X,Z), uncle(Z,Y).
sibling(X,Y) :- parent(X,Z), child(Z,Y).
sibling(X,Y) :- sibling(X,Z), sibling(Z,Y).

child_in_law(X,Y) :- child(X,Z), so(Z,Y).
parent_in_law(X,Y) :- so(X,Z), parent(Z,Y).

nephew(X,Y) :- sibling(X,Z), child(Z,Y).
uncle(X,Y) :- parent(X,Z), sibling(Z,Y).

% ============================================================================
% Example 7: Coin Classification (T8)
% ============================================================================
% Neural predicates for coin classification
% nn(net1,[X],Y,[heads,tails])::coin1(X,Y).
% nn(net2,[X],Y,[heads,tails])::coin2(X,Y).

compare(X,X,same).
compare(X,Y,different) :- \+compare(X,Y,same).

coins(X,Comparison) :-
    coin1(X,C1),
    coin2(X,C2),
    compare(C1,C2,Comparison).

% ============================================================================
% Example 8: Noisy MNIST Addition (T4)
% ============================================================================
% nn(classifier,[X],Y,[0..9])::digit(X,Y).
% t(0.2)::noisy.
% 1/19::uniform(X,Y,0) ; ... ; 1/19::uniform(X,Y,18).

noisy_addition(X,Y,Z) :- noisy, uniform(X,Y,Z).
noisy_addition(X,Y,Z) :- \+noisy, digit(X,N1), digit(Y,N2), Z is N1 + N2.

% ============================================================================
% Notes:
% - Lines starting with % nn(...) are neural predicate declarations
% - These require the DeepProbLog framework to execute
% - Pure Prolog rules can be tested with standard Prolog interpreters
% - Probabilistic facts (e.g., 0.1::burglary) require ProbLog
% ============================================================================
