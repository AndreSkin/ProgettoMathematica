#!/usr/bin/env wolframscript
(* ::Package:: *)

(* :Title: ENC.wls *)
(* :Context: ENC` *)
(* :Author: Gruppo 1 *)
(* :Summary: Funzioni di espansione e corruzione di messaggi per mezzo di Reed-Solomon *)
(* :Copyright: CC BY-SA*)
(* :Package Version: 1 *)
(* :Mathematica Version: 13.2 *)
(* :History: last modified 09/08/23 *)
(* :Keywords: Reed-Solomon, encryption*)
(* :Sources: V. notebook.nb*)
(* :Limitations: this is for educational purposes only. *)
(* :Discussion: *)
(* :Requirements: Package DEC.wls *)

BeginPackage["ENC`"]

(* Definizione dell'uso della funzione StrEncode *)
StrEncode::usage = "StrEncode[inputString, maxErrors] prende in input una stringa e un intero che rappresenta il numero massimo di errori consentiti. 
La funzione calcola un polinomio interpolante che passa attraverso i punti rappresentati dai valori ASCII dei caratteri della stringa. 
Se la derivata del polinomio al punto finale \[EGrave] negativa, aggiunge la lettera 'z' alla stringa originale e ricalcola il polinomio interpolante.
Espande il polinomio aggiungendo punti extra e mostra i risultati in una colonna."

(* Definizione dell'uso della variabile ComputedValues *)
ComputedValues::usage = "ComputedValues \[EGrave] una variabile pubblica che contiene i valori computati da StrEncode."

CreaInterfaccia::usage="Crea un\[CloseCurlyQuote]interfaccia utente per chiamare la funzione StrEncode con gli argomenti specificati dall\[CloseCurlyQuote]utente. 
L\[CloseCurlyQuote]interfaccia consiste in un campo di input per l\[CloseCurlyQuote]input dell\[CloseCurlyQuote]utente, uno slider per selezionare il valore massimo degli errori 
e un bottone per chiamare la funzione StrEncode e mostrare l\[CloseCurlyQuote]output."


Begin["Private`"]

(* Inizializza le variabili locali al modulo *)
CreaInterfaccia[] := DynamicModule[{UpperBound = 12, baseInput = "ciao", output = "", maxErrors = 1, randomSeed = 1, seedMessage = ""},
  (* Crea una colonna di elementi dell'interfaccia *)
  Column[{
    (* Crea una riga con un'etichetta e un campo di input per l'input dell'utente per impostare il RandomSeed *)
    Row[{"Random Seed: ", InputField[Dynamic[randomSeed], Number, BaseStyle -> {FontSize -> 24}, ImageSize -> {200, 50}], 
    Button["Imposta Seed", SeedRandom[If[NumericQ[randomSeed], randomSeed, 1]]; 
    seedMessage = If[NumericQ[randomSeed], StringJoin["Seed impostato a ", ToString[randomSeed]], "Seed assente: il seed di default \[EGrave] 1"], BaseStyle -> {FontSize -> 24}, ImageSize -> {200, 50}]}],
    Dynamic[seedMessage],
    
    Row[{"Stringa di Input: ", InputField[Dynamic[baseInput], String, BaseStyle -> {FontSize -> 24}, ImageSize -> {200, 50}]}],
    (* Crea una riga con un'etichetta, uno slider e un elemento dinamico per mostrare il valore corrente dello slider *)
    Row[{"Massimo Errori: ", Slider[Dynamic[maxErrors], {1, Dynamic[Max[1, UpperBound - StringLength[StringTrim[baseInput]]]], 1}], Dynamic[maxErrors]}],
    (* Crea un elemento dinamico per mostrare l'output della funzione StrEncode *)
    Dynamic[output],
    (* Crea un bottone che chiama la funzione StrEncode quando viene premuto e mostra l'output *)
    Button["Espandi e corrompi stringa",
      If[StringLength[StringTrim[baseInput]] < 2,
        output = "Per favore inserisci contenente almeno due caratteri",
        output = StrEncode[StringTake[StringTrim[baseInput], Min[StringLength[StringTrim[baseInput]], UpperBound]], maxErrors];
        maxErrors = 1
      ],
      BaseStyle -> {FontSize -> 24}, ImageSize -> {400, 100}
    ]
  }]
]




(* Definizione della funzione StrEncode *)
StrEncode[inputString_String, maxErrors_Integer] :=
DynamicModule[{i, asciiList, points, polynomial, expandedPoints, charList, str, fDev, fDevSign, labeledPoints, labeledPointsList},
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
        Row[{
            PlotString[polynomial, asciiList, charList, points],
            ExpandString[polynomial, expandedPoints]}],
            StringCorruption[expandedPoints, maxErrors]
        }
    ]
]




(* Definizione della funzione PlotString *)
PlotString[polynomial_, asciiList_List, charlist_List, points_List] :=
Module[{i, labeledPoints, labeledPointsList},
    (* Costruisce una lista di punti con etichette per il grafico *)
    labeledPoints = Transpose[{points, charlist}];
    labeledPointsList = Labeled[#[[1]], #[[2]], Above] & /@ labeledPoints;
    
    (* Mostra un grafico dei punti con le etichette *)
    Row[
        {
            ListPlot[labeledPointsList, PlotStyle -> {PointSize[Medium], Blue}, AxesLabel -> {"x", "ASCII Value"}, ImageSize -> Large, BaseStyle -> {FontSize -> 24}],
            PlotPoly[polynomial, asciiList, points, labeledPointsList]
        }
    ]
]



(* Definizione della funzione PlotPoly *)
PlotPoly[polynomial_, asciiList_List, points_List, labeledPointsList_List] :=
Module[{},
    (* Mostra il polinomio e i punti su un grafico *)
    Show[
        Plot[polynomial, {x, 1, Length[asciiList]}, PlotStyle -> Red, AxesLabel -> {"x", "ASCII Value"}, ImageSize -> Large, BaseStyle -> {FontSize -> 24}],
        ListPlot[labeledPointsList, PlotStyle -> {PointSize[Medium], Blue}]
    ]
]




(* Definizione della funzione ExpandString *)
ExpandString[polynomial_, expandedPoints_List] :=
Module[{asciiListExpanded, stringExpanded, charListExpanded, labeledPointsExpanded, labeledPointsExpandedList},
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
        Plot[polynomial, {x, 1, Length[expandedPoints]}, PlotStyle -> Red, AxesLabel -> {"x", "ASCII Value"}, ImageSize -> Large, BaseStyle -> {FontSize -> 24}],
        ListPlot[labeledPointsExpandedList, PlotStyle -> {PointSize[Medium], Blue}]
    ]
]




(* Definizione della funzione StringCorruption *)
StringCorruption[expandedPoints_List, maxErrors_Integer] :=
Module[{asciiListExpanded, stringExpanded, nErrors, errorsPos,myStringAsList,corruptedString,StrLen},
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
    
    Column[{
    StringJoin["Stringa corrotta: ", corruptedString],
    StringJoin["Lunghezza della stringa espansa: ", ToString[StrLen]]}]
]



End[]

EndPackage[]

