#!/usr/bin/env wolframscript
(* ::Package:: *)

(* :Title: BitErasure.wls *)
(* :Context: Erasure` *)
(* :Author: Gruppo 1 *)
(* :Summary: Funzione per fare bit erasure *)
(* :Copyright: CC BY-SA*)
(* :Package Version: 1 *)
(* :Mathematica Version: 13.3 *)
(* :History: last modified 10/07/23 *)
(* :Keywords: Reed-Solomon, encryption*)
(* :Sources: V. tutorial.nb*)
(* :Limitations: this is for educational purposes only. *)
(* :Discussion: *)
(* :Requirements: *)


BeginPackage["BitErasure`"]
(* Definizione dell'uso della funzione StrEncode *)
DeleteBit::usage = "DeleteBit[string] prende in input una stringa e restituisce una versione modificata della stessa da cui \[EGrave] stato eliminato un bit.
La funzione converte la stringa in una sequenza di codici ASCII e successivamente in una rappresentazione binaria.
Poi, genera un indice casuale e sostituisce l'elemento all'indice casuale con il simbolo '_'."

Begin["`Private`"]


(*UTF-8*)(*Variabile per indicare se mostrare il risultato*)ShowResult=False;

(*Funzione per stampare il risultato*)
PrintResult[originalList_List,modifiedList_List]:=Module[{},
	Column[{StringJoin["La rappresentazione binaria di questa stringa \[EGrave]:\n ",
	(*Unisce gli elementi di originalList in una stringa separati da virgole*)
	StringJoin[ToString/@Riffle[originalList,", "]]],
	(*Unisce gli elementi di modifiedList in una stringa separati da virgole*)
	StringJoin["Ora succede qualche catastrofe e otteniamo:\n ",
	StringJoin[ToString/@Riffle[modifiedList,", "]]]}
	] 
	(*Se ShowResult \[EGrave] falso,non viene mostrato alcun risultato*)]

(*Funzione per eliminare un bit*)
DeleteBit[string_String]:=Module[{str=string, bitString, originalList, modifiedList, length, randomIndex},
	(*Controlla se la stringa non \[EGrave] vuota*)
	If[StringLength[str]>0,
	(*Converte la stringa in una sequenza di codici ASCII e successivamente in una rappresentazione binaria*)
	bitString=IntegerDigits[ToCharacterCode[str],2,8];
	(*Appiattisce la lista di bit*)
	originalList=Flatten[bitString];
	(*Copia la lista originale neloriginalStringla lista modificata*)
	modifiedList=originalList;
	(*Calcola la lunghezza della lista modificata*)
	length=Length[modifiedList];
	(*Genera un indice casuale tra 1 e la lunghezza della lista*)
	randomIndex=RandomInteger[{1,length}];
	(*Sostituisce l'elemento all'indice casuale con il simbolo "_"*)
	modifiedList=ReplacePart[modifiedList,randomIndex->"_"];
	
	 If[Length[originalList]>0,
	PrintResult[originalList, modifiedList]] (*Imposta ShowResult su vero per mostrare il risultato*)]
	]



End[]

EndPackage[]
