#!/usr/bin/env wolframscript
(* ::Package:: *)

(* :Title: BestExample *)
(* :Context: BestExample` *)
(* :Author: GS *)
(* :Summary: an example of good programming style *)
(* :Copyright: GS 2023 *)
(* :Package Version: 1 *)
(* :Mathematica Version: 13 *)
(* :History: last modified 27/3/2023 *)
(* :Keywords: programming style, local variables *)
(* :Sources: biblio *)
(* :Limitations: this is for educational purposes only. *)
(* :Discussion: *)
(* :Requirements: *)
(* :Warning: package Context is not defined *)

PowerSum::usage = "PowerSum[x, n] returns the sum of the first n powers of x."
Begin["Private`"]
PowerSum\:f010x_, n_\:f016 :=
Module\:f010{i},
Sum\:f010 x^i, \:f01di, 1, n\:f028 \:f016
\:f016
End[]
