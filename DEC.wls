#!/usr/bin/env wolframscript
(* ::Package:: *)

(* :Title: DEC.wls *)
(* :Context: DEC` *)
(* :Author: Gruppo 1 *)
(* :Summary: Funzioni di decodifica di messaggi corrotti per mezzo di Reed-Solomon *)
(* :Copyright: CC BY-SA*)
(* :Package Version: 1 *)
(* :Mathematica Version: 13.2 *)
(* :History: last modified 16/06/23 *)
(* :Keywords: Reed-Solomon, decryption*)
(* :Sources: V. notebook.nb*)
(* :Limitations: this is for educational purposes only. *)
(* :Discussion: *)
(* :Requirements: Package ENC.wls *)

BeginPackage["DEC`"]

MsgDecode::usage = "Decode[x] prende in input un messaggio x generato da ENC.wls e lo decodifica."

Begin["Private`"]

MsgDecode[input_List] := DynamicModule[{i},
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
    Grid[{
      {ListPlot[labeledPointsList, PlotStyle -> {PointSize[Medium], Blue},
        AxesLabel -> {"x", "ASCII Value"}, ImageSize -> {1000,400}], SpanFromLeft},
      {Show[Plot[reconstructedPoly, {x, 1, strLen}, PlotStyle -> Red,
          AxesLabel -> {"x", "ASCII Value"}, ImageSize -> {1000,400}],
        ListPlot[labeledPointsList, PlotStyle -> {PointSize[Medium], Blue},
          AxesLabel -> {"x", "ASCII Value"}, ImageSize -> {1000,400}]], SpanFromLeft}
    }, Spacings -> {5,5}]
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

(* Stampa la stringa ricostruita AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA *)
(*Print["La stringa ricostruita \[EGrave]: " <> ToString[reconstructedString]];*)
]


End[]

EndPackage[]

