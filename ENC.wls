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

BeginPackage["ENC`"]

(* Definizione dell'uso della funzione StrEncode *)
StrEncode::usage = "Encode[x] prende in input una stringa x e la espande."

(* Definizione dell'uso della variabile ComputedValues *)
ComputedValues::usage = "ComputedValues \[EGrave] una variabile pubblica che contiene i valori computati da StrEncode."

Begin["`Private`"]

(* Definizione della funzione StrEncode *)
StrEncode[inputString_String, maxErrors_Integer] :=
DynamicModule[{i},
    (* Converte la stringa in una lista di caratteri ASCII *)
    asciiList = ToCharacterCode[inputString];
    
    (* Costruisce una lista di punti usando gli indici dei caratteri ASCII e i loro valori ASCII *)
    points = Transpose[{Range[Length[asciiList]], asciiList}];
    
    (* Calcola un polinomio interpolante che passa attraverso i punti *)
    polynomial = Expand[InterpolatingPolynomial[points, x]];
    
    (* Espande il polinomio aggiungendo punti aggiuntivi *)
    expandedPoints = Table[{i, polynomial /. x -> i}, {i, Length[points] + maxErrors}];
    
    (* Converte la stringa in una lista di caratteri *)
    charList = Characters[inputString];
    
    (* Memorizza la stringa originale *)
    str = inputString;
    
    (* Calcola la derivata del polinomio al punto finale *)
    fDev = D[polynomial, x];
    fDevSign = Sign[fDev /. x -> asciiList[[-1]]];
    
    (* Verifica se la derivata \[EGrave] negativa *)
    If[fDevSign < 0,
        (* Aggiunge la lettera 'z' alla stringa originale *)
        str = inputString <> "z";
        
        (* Converte la nuova stringa in una lista di caratteri ASCII *)
        asciiList = ToCharacterCode[str];
        
        (* Costruisce una lista di punti usando gli indici dei caratteri ASCII e i loro valori ASCII *)
        points = Transpose[{Range[Length[asciiList]], asciiList}];
        
        (* Converte la stringa in una lista di caratteri *)
        charList = Characters[str];
        
        (* Costruisce una lista di punti con etichette per il grafico *)
        labeledPoints = Transpose[{points, charList}];
        labeledPointsList = Labeled[#[[1]], #[[2]], Above] & /@ labeledPoints;
        
        (* Ricalcola il polinomio interpolante *)
        polynomial = Expand[InterpolatingPolynomial[points, x]]
    ];
    
    (* Espande il polinomio aggiungendo punti aggiuntivi *)
    expandedPoints = Table[{i, polynomial /. x -> i}, {i, Length[points] + maxErrors}];
    
    (* Mostra i risultati in una colonna *)
    Column[
        {
            PlotString[polynomial, asciiList, charList, points],
            ExpandString[polynomial, expandedPoints],
            StringCorruption[expandedPoints, maxErrors]
        }
    ]
]



(* Definizione della funzione PlotString *)
PlotString[polynomial_, asciiList_List, charlist_List, points_List] :=
Module[{i},
    (* Costruisce una lista di punti con etichette per il grafico *)
    labeledPoints = Transpose[{points, charlist}];
    labeledPointsList = Labeled[#[[1]], #[[2]], Above] & /@ labeledPoints;
    
    (* Mostra un grafico dei punti con le etichette *)
    Column[
        {
            ListPlot[labeledPointsList, PlotStyle -> {PointSize[Medium], Blue}, AxesLabel -> {"x", "ASCII Value"}, ImageSize -> Medium],
            PlotPoly[polynomial, asciiList, points, labeledPointsList]
        }
    ]
]


(* Definizione della funzione PlotPoly *)
PlotPoly[polynomial_, asciiList_List, points_List, labeledPointsList_List] :=
Module[{i},
    (* Mostra il polinomio e i punti su un grafico *)
    Show[
        Plot[polynomial, {x, 1, Length[asciiList]}, PlotStyle -> Red, AxesLabel -> {"x", "ASCII Value"}, ImageSize -> Medium],
        ListPlot[labeledPointsList, PlotStyle -> {PointSize[Medium], Blue}]
    ]
]


(* Definizione della funzione ExpandString *)
ExpandString[polynomial_, expandedPoints_List] :=
Module[{i},
    (* Ottiene la lista dei valori ASCII dagli expandedPoints *)
    asciiListExpanded = Round[Last /@ expandedPoints];
    
    (* Converte la lista di valori ASCII in una stringa *)
    stringExpanded = FromCharacterCode[asciiListExpanded];
    
    (* Converte la stringa espansa in una lista di caratteri *)
    charListExpanded = FromCharacterCode /@ asciiListExpanded;
    
    (* Costruisce una lista di punti espansi con etichette per il grafico *)
    labeledPointsExpanded = Transpose[{expandedPoints, charListExpanded}];
    labeledPointsExpandedList = Labeled[#[[1]], #[[2]], Above] & /@ labeledPointsExpanded;
    
    (* Mostra il polinomio espanso e i punti su un grafico *)
    Show[
        Plot[polynomial, {x, 1, Length[expandedPoints]}, PlotStyle -> Red, AxesLabel -> {"x", "ASCII Value"}, ImageSize -> Medium],
        ListPlot[labeledPointsExpandedList, PlotStyle -> {PointSize[Medium], Blue}]
    ]
]


(* Definizione della funzione StringCorruption *)
StringCorruption[expandedPoints_List, maxErrors_Integer] :=
Module[{i},
    (* Ottiene la lista dei valori ASCII dagli expandedPoints *)
    asciiListExpanded = Round[Last /@ expandedPoints];
    
    (* Converte la lista di valori ASCII in una stringa *)
    stringExpanded = FromCharacterCode[asciiListExpanded];
    
    (* Genera un numero casuale di errori da introdurre nella stringa *)
    nErrors = RandomInteger[{1, maxErrors}];
    
    (* Genera una lista casuale di posizioni in cui introdurre gli errori *)
    errorsPos = RandomSample[Range[StringLength[stringExpanded]], nErrors];
    
    (* Converte la stringa espansa in una lista di caratteri *)
    myStringAsList = Characters[stringExpanded];
    
    (* Rimuove i caratteri alle posizioni degli errori *)
    myStringAsList = Delete[myStringAsList, List /@ errorsPos];
    
    (* Ricostruisce la stringa corrotta *)
    corruptedString = StringJoin[myStringAsList];
    
    (* Calcola la lunghezza della stringa espansa *)
    StrLen = StringLength[stringExpanded];
    
    (* Memorizza i valori calcolati nella variabile ComputedValues *)
    ComputedValues = {corruptedString, errorsPos, StrLen, maxErrors};
    
    (* Stampa i valori calcolati *)
    Print[ComputedValues]
]


End[]

EndPackage[]

