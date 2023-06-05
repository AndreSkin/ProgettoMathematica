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

BeginPackage["DEC`"]

MsgDecode::usage = "Decode[x] prende in input una stringa x e la decodifica."

Begin["Private"]

MsgDecode[input_List] :=
DynamicModule[{i},
(* Estrae i valori dalla lista di input *)
corruptedString = input[[1]];
errorsPos = input[[2]];
strLen = input[[3]];
maxErrors = input[[4]];

(* Converte la stringa corrotta in una lista di valori ASCII *)
asciiList = ToCharacterCode[corruptedString];

(* Converte la stringa corrotta in una lista di caratteri *)
charList = Characters[corruptedString];

(* Calcola la posizione dei caratteri non corrotti *)
charsPos = Complement[Range[strLen], errorsPos];

(* Costruisce una lista di punti con etichette per il grafico *)
points = Transpose[{charsPos, asciiList}];
labeledPoints = Transpose[{points, charList}];
labeledPointsList = Labeled[#[[1]], #[[2]], Above] & /@ labeledPoints;

(* Ricostruisce il polinomio interpolante *)
reconstructedPoly = Expand[InterpolatingPolynomial[points, x]];

(* Richiama la funzione per ricostruire la stringa originale *)
ReconstructString[reconstructedPoly, strLen, maxErrors];

(* Richiama la funzione per visualizzare il polinomio ricostruito *)
PlotReconstructedPoly[reconstructedPoly, labeledPointsList, strLen]
]



(* Definizione della funzione PlotReconstructedPoly *)
PlotReconstructedPoly[reconstructedPoly_, labeledPointsList_List, strLen_Integer] :=
Module[{i},
(* Mostra un grafico dei punti e del polinomio ricostruito *)
Column[
{
	ListPlot[labeledPointsList, PlotStyle -> {PointSize[Medium], Blue}, AxesLabel -> {"x", "ASCII Value"}, ImageSize -> Medium],
	Show[
		Plot[reconstructedPoly, {x, 1, strLen}, PlotStyle -> Red, AxesLabel -> {"x", "ASCII Value"}, ImageSize -> Medium],
		ListPlot[labeledPointsList, PlotStyle -> {PointSize[Medium], Blue}, AxesLabel -> {"x", "ASCII Value"}, ImageSize -> Medium]
	]
}
]
]


(* Definizione della funzione ReconstructString *)
ReconstructString[reconstructedPoly_, strLen_Integer, maxErrors_Integer] :=
Module[{i},
(* Calcola i punti ricostruiti *)
reconstructedPoints = Table[{i, reconstructedPoly /. x -> i}, {i, strLen - maxErrors}];
(* Ottiene la lista di valori ASCII dai punti ricostruiti *)
reconstructedAsciiList = Round[Last /@ reconstructedPoints];

(* Converte la lista di valori ASCII in una stringa ricostruita *)
reconstructedString = FromCharacterCode[reconstructedAsciiList];

(* Stampa la stringa ricostruita *)
Print[reconstructedString];
]


End[]

EndPackage[]

