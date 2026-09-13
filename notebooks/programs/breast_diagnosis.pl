% Provisional DeepProbLog program for CBIS-DDSM concept-based diagnosis.
% These rules are a COMPUTATIONAL WORKING PROGRAM, not official BI-RADS criteria.
% They must be verified against clinical documentation before any medical claim.
%
% Mass evidence is used only for mass lesions. Suspicious calc is used only
% for calcification lesions. This stops a false calc_suspicious prediction
% on a mass from flipping the diagnosis.

nn(kind_net, [X], Y, [mass, calcification]) :: lesion_kind(X, Y).
nn(shape_net, [X], Y, [regular, irregular, unknown]) :: shape(X, Y).
nn(margin_net, [X], Y, [circumscribed, spiculated, other, unknown]) :: margin(X, Y).
nn(calc_net, [X], Y, [absent, typically_benign, suspicious]) :: calc(X, Y).

malignant(X) :- lesion_kind(X, mass), shape(X, irregular).
malignant(X) :- lesion_kind(X, mass), margin(X, spiculated).
malignant(X) :- lesion_kind(X, calcification), calc(X, suspicious).
