#!/usr/bin/env wolframscript
(* ::Package:: *)

(* :Title: DEC.wls *)
(* :Context: DEC` *)
(* :Author: Gruppo 1 *)
(* :Summary: Funzioni di decodifica di messaggi corrotti per mezzo di Reed-Solomon *)
(* :Copyright: CC BY-SA*)
(* :Package Version: 1 *)
(* :Mathematica Version: 13.2 *)
(* :History: last modified 09/08/23 *)
(* :Keywords: Reed-Solomon, decryption*)
(* :Sources: V. notebook.nb*)
(* :Limitations: this is for educational purposes only. *)
(* :Discussion: *)
(* :Requirements: Package ENC.wls *)

BeginPackage["DEC`"]

(*MsgDecode::usage = "MsgDecode[input] prende in input una lista contenente una stringa corrotta, 
una lista di posizioni degli errori, la lunghezza della stringa originale e il numero massimo di errori consentiti. 
La funzione ricostruisce il polinomio interpolante utilizzando i caratteri non corrotti e lo utilizza per ricostruire la stringa originale. 
Restituisce una lista contenente la stringa ricostruita e il grafico del polinomio ricostruito."*)

DecodeInterfaccia::usage="Funzione che crea un\[CloseCurlyQuote]interfaccia utente per chiamare la funzione MsgDecode. 
L\[CloseCurlyQuote]interfaccia consiste in un bottone per decodificare la stringa e due elementi dinamici per mostrare il messaggio di errore 
o la stringa decodificata e i grafici restituiti dalla funzione MsgDecode. 
La funzione accetta come input una lista di valori calcolati da MsgEncode e utilizza le variabili locali al modulo per gestire lo stato dell\[CloseCurlyQuote]interfaccia."


Begin["Private`"]

(* Inizializza le variabili locali al modulo *)
DecodeInterfaccia[ComputedValues_] := DynamicModule[{output = "", decodedString = "", grafici = {}, err = ""},
  (* Crea una colonna di elementi dell'interfaccia *)
  Column[{
    (* Crea un bottone per decodificare la stringa *)
    Button["Decodifica stringa",
      (* Verifica se l'input \[EGrave] valido *)
      If[Not[ListQ[ComputedValues]],
        (* Se l'input non \[EGrave] valido, mostra un messaggio di errore *)
        err = "Corrompi una stringa nella casella di input precedente prima di provare a decodificare";
        ,
        (* Se l'input \[EGrave] valido, chiama la funzione MsgDecode e mostra l'output *)
        err = "";
        output = MsgDecode[ComputedValues];
        decodedString = "La stringa decodificata \[EGrave]: " <> output[[1]];
        grafici = output[[2]];
      ],
      BaseStyle -> {FontSize -> 24}, ImageSize -> {400, 100}
    ],
    (* Mostra il messaggio di errore o la stringa decodificata *)
    Dynamic[err],
    Dynamic[decodedString],
    (* Mostra i grafici restituiti dalla funzione MsgDecode *)
    Dynamic[If[grafici === {}, "", grafici]]
  }]
]



MsgDecode[input_List] := 
Module[{corruptedString, errorsPos, strLen, maxErrors, asciiList, charList, charsPos,points,labeledPoints, labeledPointsList, reconstructedPoly,recStr,grafici, toReturn},    
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
    recStr = ReconstructString[reconstructedPoly, strLen, maxErrors];

    (* Richiama la funzione per visualizzare il polinomio ricostruito *)
    grafici = PlotReconstructedPoly[reconstructedPoly, labeledPointsList, strLen];
    
    toReturn = {recStr, grafici};
    Return[toReturn];
]


(* Definizione della funzione PlotReconstructedPoly *)
PlotReconstructedPoly[reconstructedPoly_, labeledPointsList_List, strLen_Integer] :=
Return[
Module[{},
    (* Mostra un grafico dei punti e del polinomio ricostruito *)
    Row[{
      ListPlot[labeledPointsList, PlotStyle -> {PointSize[Large], Blue},
        AxesLabel -> {"x", "ASCII Value"}, ImageSize -> {600, 400}, 
        LabelStyle -> {FontSize -> 18}],
        
      Show[Plot[reconstructedPoly, {x, 1, strLen}, PlotStyle -> Red,
          AxesLabel -> {"x", "ASCII Value"}, ImageSize -> {600, 400}, 
          LabelStyle -> {FontSize -> 18}],
          
        ListPlot[labeledPointsList, PlotStyle -> {PointSize[Large], Blue},
          AxesLabel -> {"x", "ASCII Value"}, ImageSize -> {600, 400}, 
          LabelStyle -> {FontSize -> 18}]]
    }]
]]




(* Definizione della funzione ReconstructString *)
ReconstructString[reconstructedPoly_, strLen_Integer, maxErrors_Integer] :=
Module[{reconstructedPoints,reconstructedAsciiList,reconstructedString  },
(* Calcola i punti ricostruiti *)
reconstructedPoints = Table[{i, reconstructedPoly /. x -> i}, {i, strLen - maxErrors}];
(* Ottiene la lista di valori ASCII dai punti ricostruiti *)
reconstructedAsciiList = Round[Last /@ reconstructedPoints];

(* Converte la lista di valori ASCII in una stringa ricostruita *)
reconstructedString = FromCharacterCode[reconstructedAsciiList];

Return[reconstructedString];
]


End[]

EndPackage[]

