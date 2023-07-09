#!/usr/bin/env wolframscript
(* ::Package:: *)

(* :Title: ExtendString.wls *)
(* :Context: ExtendString` *)
(* :Author: Gruppo 1 *)
(* :Summary: Funzione per fare bit erasure *)
(* :Copyright: CC BY-SA*)
(* :Package Version: 1 *)
(* :Mathematica Version: 13.3 *)
(* :History: last modified 8/07/23 *)
(* :Keywords: Reed-Solomon, encryption*)
(* :Sources: V. tutorial.nb*)
(* :Limitations: this is for educational purposes only. *)
(* :Discussion: *)
(* :Requirements: *)


BeginPackage["GeneraPunti`"]
(* Definizione dell'uso della funzione StrEncode *)
GeneraPunti::usage = "GeneraPunti[xmin,xmax,ymin,ymax,n] "

Begin["`Private`"]


(*GENERATORE ESERCIZIO*)(*Imposta il numero di punti e l'intervallo per le x e le y*)n=7; (*Imposta il numero di punti a 7*)
punti={}; (*Inizializza una lista vuota per i punti*)
polinomio=""; (*Inizializza una stringa vuota per il polinomio*)

(*Trova il grado del polinomio unico che passa attraverso i punti*)
grado=n-1; (*Calcola il grado del polinomio come n-1*)

(*Definisce una funzione per generare la soluzione*)
GeneraSoluzione[punti_,xmin_,xmax_]:=Module[{grado,polinomio},grado=Length[punti]-1;polinomio=InterpolatingPolynomial[punti,x];polinomio]

(*Definisce una funzione per generare i punti casualmente*)
GeneraPunti[xmin_,xmax_,ymin_,ymax_,n_]:=Module[{xvalori,yvalori,punti},xvalori=RandomSample[Range[xmin,xmax],n];yvalori=RandomInteger[{ymin,ymax},n];punti=Transpose[{xvalori,yvalori}];polinomio=GeneraSoluzione[punti,xmin,xmax];punti]



(*Definisce una funzione per generare la soluzione*)
SolGen[punti_,xmin_,xmax_]:=Module[{grado,polinomio},grado=Length[punti]-1;polinomio=InterpolatingPolynomial[punti,x];newgrado=grado;newpolinomio=polinomio;newpunti=punti;polinomio]




End[]

EndPackage[]
