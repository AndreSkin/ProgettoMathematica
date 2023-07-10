#!/usr/bin/env wolframscript
(* ::Package:: *)

(* :Title: ExtendString.wls *)
(* :Context: ExtendString` *)
(* :Author: Gruppo 1 *)
(* :Summary: Funzioni per generare i punti di un polinomio e il polinomio stesso*)
(* :Copyright: CC BY-SA*)
(* :Package Version: 1 *)
(* :Mathematica Version: 13.3 *)
(* :History: last modified 10/07/23 *)
(* :Keywords: Reed-Solomon, encryption*)
(* :Sources: V. tutorial.nb*)
(* :Limitations: this is for educational purposes only. *)
(* :Discussion: *)
(* :Requirements: *)


BeginPackage["GeneraPunti`"]


GeneraPunti::usage = "GeneraPunti[xmin, xmax, ymin, ymax, n] prende in input cinque interi che rappresentano rispettivamente il valore minimo e massimo dell'asse x, 
il valore minimo e massimo dell'asse y e il numero di punti da generare. 
La funzione genera una lista di n punti casuali all'interno del rettangolo definito dai valori xmin, xmax, ymin e ymax e restituisce la lista di punti."

SolGen::usage = "SolGen[punti, xmin, xmax] prende in input una lista di punti e due interi che rappresentano rispettivamente il valore minimo e massimo dell'asse x. 
La funzione calcola il grado del polinomio interpolante che passa attraverso i punti e restituisce una lista contenente il grado del polinomio,
il polinomio interpolante e la lista di punti."

x;

Begin["`Private`"]


(*GENERATORE ESERCIZIO*)(*Imposta il numero di punti e l'intervallo per le x e le y*)n=7; (*Imposta il numero di punti a 7*)
punti={}; (*Inizializza una lista vuota per i punti*)
polinomio=""; (*Inizializza una stringa vuota per il polinomio*)

(*Trova il grado del polinomio unico che passa attraverso i punti*)
grado=n-1; (*Calcola il grado del polinomio come n-1*)

(*Definisce una funzione per generare la soluzione*)
GeneraSoluzione[punti_List,xmin_Integer,xmax_Integer]:=Module[{grado,polinomio},
grado=Length[punti]-1;
polinomio=InterpolatingPolynomial[punti,x];
polinomio
]

(*Definisce una funzione per generare i punti casualmente*)
GeneraPunti[xmin_Integer,xmax_Integer,ymin_Integer,ymax_Integer,n_Integer]:=Module[{xvalori,yvalori,punti},
xvalori=RandomSample[Range[xmin,xmax],n];
yvalori=RandomInteger[{ymin,ymax},n];
punti=Transpose[{xvalori,yvalori}];
polinomio=GeneraSoluzione[punti,xmin,xmax];
punti
]



(*Definisce una funzione per generare la soluzione*)
SolGen[punti_List,xmin_Integer,xmax_Integer]:=Module[{grado,polinomio},grado=Length[punti]-1;
polinomio=InterpolatingPolynomial[punti,x];
newgrado=grado;
newpolinomio=polinomio;
newpunti=punti;
polinomio;
myList = {newgrado, newpolinomio, newpunti};
Return[myList];
]



End[]

EndPackage[]
