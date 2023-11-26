#!/usr/bin/env wolframscript
(* ::Package:: *)

(* :Title: ENCandDEC.wls *)
(* :Context: ENCandDEC` *)
(* :Author: Gruppo 1 *)
(* :Summary: Funzioni di espansione e corruzione di messaggi per mezzo di Reed-Solomon + funzioni di decodifica di messaggi corrotti per mezzo di Reed-Solomon *)
(* :Copyright: CC BY-SA*)
(* :Package Version: 2.0 *)
(* :Mathematica Version: 13.3 *)
(* :History: last modified 26/11/23 *)
(* :Keywords: Reed-Solomon, encryption*)
(* :Sources: V. notebook.nb*)
(* :Limitations: this is for educational purposes only. *)
(* :Discussion: *)
(* :Requirements: None *)

BeginPackage["ENCandDEC`"]

(* Definizione dell'uso della funzione StrEncode *)
StrEncode::usage = "StrEncode[inputString, maxErrors] prende in input una stringa e un intero che rappresenta il numero massimo di errori consentiti. 
La funzione calcola un polinomio interpolante che passa attraverso i punti rappresentati dai valori ASCII dei caratteri della stringa. 
Se la derivata del polinomio al punto finale \[EGrave] negativa, aggiunge la lettera '\[AGrave]' alla stringa originale e ricalcola il polinomio interpolante.
Espande il polinomio aggiungendo punti extra e mostra i risultati in una colonna."

CreaInterfaccia::usage="Crea un\[CloseCurlyQuote]interfaccia utente per chiamare la funzione StrEncode con gli argomenti specificati dall\[CloseCurlyQuote]utente. 
L\[CloseCurlyQuote]interfaccia consiste in un campo di input per l\[CloseCurlyQuote]input dell\[CloseCurlyQuote]utente, uno slider per selezionare il valore massimo degli errori 
e un bottone per chiamare la funzione StrEncode e mostrare l\[CloseCurlyQuote]output."

DecodeInterfaccia::usage="Funzione che crea un\[CloseCurlyQuote]interfaccia utente per chiamare la funzione MsgDecode. 
L\[CloseCurlyQuote]interfaccia consiste in un bottone per decodificare la stringa e due elementi dinamici per mostrare il messaggio di errore 
o la stringa decodificata e i grafici restituiti dalla funzione MsgDecode. 
La funzione accetta come input una lista di valori calcolati da MsgEncode e utilizza le variabili locali al modulo per gestire lo stato dell\[CloseCurlyQuote]interfaccia."


Begin["Private`"]


CreaInterfaccia[] := DynamicModule[{UpperBound = 12, baseInput = "ciao", output = "", maxErrors = 1, randomSeed = 1, seedMessage = "", showDecodeButton = False},
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
        (*ComputedValues = output;*)
        maxErrors = 1;
        showDecodeButton = True;
      ],
      BaseStyle -> {FontSize -> 24}, ImageSize -> {400, 100}
    ],
    (* Mostra la seconda interfaccia solo se showDecodeButton \[EGrave] True *)
    Dynamic[If[showDecodeButton, DecodeInterfaccia[], "" ]],
    Dynamic[If[showDecodeButton, 
    Button["(Pulizia)", 
        baseInput = "ciao"; output = ""; randomSeed = 1; seedMessage = ""; showDecodeButton = False;,
        BaseStyle -> {FontSize -> 24}, ImageSize -> {400, 100}
    ], 
    ""
]]

  }]
]


(* Inizializza le variabili locali al modulo *)
DecodeInterfaccia[] := DynamicModule[{output = "", decodedString = "", grafici = {}, err = ""},
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
        output = MsgDecode[];
        grafici = output[[2]];
        If[output[[3]] === True, decodedString = "La stringa decodificataa \[EGrave]: " <> ToString[StringDrop[output[[1]],-1]],
        decodedString = "La stringa decodificata \[EGrave]: " <> ToString[output[[1]]]]
      ],
      BaseStyle -> {FontSize -> 24}, ImageSize -> {400, 100}
    ],
    (* Mostra il messaggio di errore o la stringa decodificata *)
    Dynamic[err],
    Dynamic[If[decodedString === "","", decodedString]],
    (* Mostra i grafici restituiti dalla funzione MsgDecode *)
    Dynamic[If[grafici === {}, "", grafici]]
  }]
]


(* Definizione della funzione StrEncode *)
StrEncode[inputString_String, maxErrors_Integer] :=
Module[{i, asciiList, points, polynomial, expandedPoints, charList, str, fDev, fDevSign, labeledPoints, labeledPointsList, isCharacterAdded,len},
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
    
    len = Length[asciiList];
    (* Calcola la derivata del polinomio al punto finale *)
    fDev = D[polynomial, x];
    fDevSign = fDev /. x -> len;
    
    (* Verifica se la derivata \[EGrave] negativa *)
    If[fDevSign < 0,
        isCharacterAdded = True;
        (* Aggiunge la lettera 'z' alla stringa originale *)
        str = inputString <> "\[AGrave]";
        
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
        (*PlotString[fDev, asciiList, charList, points],*)
            PlotString[polynomial, asciiList, charList, points],
            ExpandString[polynomial, expandedPoints]
            }],
            StringCorruption[expandedPoints, maxErrors, isCharacterAdded]   
        }
    ]
]


(* Definizione della funzione PlotString (primo grafico di ENC) *)
PlotString[polynomial_, asciiList_List, charlist_List, points_List] :=
Module[{i, labeledPoints, labeledPointsList},
    (* Costruisce una lista di punti con etichette per il grafico *)
    labeledPoints = Transpose[{points, charlist}];
    labeledPointsList = Labeled[#[[1]], #[[2]], Above] & /@ labeledPoints;
    
    (* Mostra un grafico dei punti con le etichette *)
    Row[
        {
            ListPlot[labeledPointsList, PlotStyle -> {PointSize[Medium], Blue}, AxesLabel -> {"x", "ASCII Value"}, ImageSize -> Large, BaseStyle -> {FontSize -> 24}, Ticks -> {Range[0, 10, 1], Automatic}],
            PlotPoly[polynomial, asciiList, points, labeledPointsList]
        }
    ]
]



(* Definizione della funzione PlotPoly (secondo grafico di ENC)*)
PlotPoly[polynomial_, asciiList_List, points_List, labeledPointsList_List] :=
Module[{},
    (* Mostra il polinomio e i punti su un grafico *)
    Show[
        Plot[polynomial, {x, 1, Length[asciiList]}, PlotStyle -> Red, AxesLabel -> {"x", "ASCII Value"}, ImageSize -> Large, BaseStyle -> {FontSize -> 24},Ticks -> {Range[0, Length[asciiList], 1], Automatic}],
        ListPlot[labeledPointsList, PlotStyle -> {PointSize[Medium], Blue},Ticks -> {Range[0, Length[labeledPointsList], 1], Automatic}]
    ]
]


(* Definizione della funzione ExpandString (terzo grafico di ENC) *)
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
        Plot[polynomial, {x, 1, Length[expandedPoints]}, PlotStyle -> Red, AxesLabel -> {"x", "ASCII Value"},PlotRange -> {{0, Length[expandedPoints] + 1}, {Min[expandedPoints[[All,2]]], Max[expandedPoints[[All,2]]] + Floor[Max[expandedPoints[[All,2]]]/3]}}, ImageSize -> Large, BaseStyle -> {FontSize -> 24}],
        ListPlot[labeledPointsExpandedList, PlotStyle -> {PointSize[Medium], Blue}]
    ]
]



(* Definizione della funzione StringCorruption *)
StringCorruption[expandedPoints_List, maxErrors_Integer,isCharacterAdded_] :=
Module[{asciiListExpanded, stringExpanded, nErrors, errorsPos,myStringAsList,corruptedString,StrLen, intervallo, valoriDaEscludere, posizioneAaccentata, toDelete},
    (* Ottiene la lista dei valori ASCII dagli expandedPoints *)
    asciiListExpanded = Round[Last /@ expandedPoints];
    
    (* Converte la lista di valori ASCII in una stringa *)
    stringExpanded = FromCharacterCode[asciiListExpanded];
    
      (* Converte la stringa espansa in una lista di caratteri *)
    myStringAsList = Characters[stringExpanded];
    intervallo = {1, maxErrors};
    
    valoriDaEscludere = {};
    
	(* Trova la posizione del carattere '\[AGrave]' nella lista *)
	posizioneAaccentata = Position[myStringAsList, "\[AGrave]"];
	
	(* Stampa il risultato *)
	If[posizioneAaccentata =!= {}, 
	  valoriDaEscludere = Last[posizioneAaccentata];
	];    
    (* Genera un numero casuale di errori da introdurre nella stringa *)
    nErrors = RandomInteger[{1, maxErrors}];
    
    (* Genera una lista casuale di posizioni in cui introdurre gli errori *)
    errorsPos = RandomSample[Range[StringLength[stringExpanded]], nErrors];
   
	If[valoriDaEscludere =!= {}, 
	   toDelete=Position[errorsPos, valoriDaEscludere[[-1]]];
    errorsPos = Delete[errorsPos, toDelete];
	];
    
    (* Rimuove i caratteri alle posizioni degli errori *)
    myStringAsList = Delete[myStringAsList, List /@ errorsPos];
    
    (* Ricostruisce la stringa corrotta *)
    corruptedString = StringJoin[myStringAsList];
    
    (* Calcola la lunghezza della stringa espansa *)
    StrLen = StringLength[stringExpanded];
    
    (* Memorizza i valori calcolati nella variabile ComputedValues *)
    ComputedValues = {corruptedString, errorsPos, StrLen, maxErrors, isCharacterAdded};
  
    Column[{
    StringJoin["Stringa corrotta: ", corruptedString],
    StringJoin["Lunghezza della stringa espansa: ", ToString[StrLen]]}]
]


MsgDecode[] := 
Module[{corruptedString, errorsPos, strLen, maxErrors, asciiList, charList, charsPos,points,labeledPoints, labeledPointsList, reconstructedPoly,recStr,grafici, toReturn, isCharacterAdded, ReconstructStringResult, reconstructedPointsList, labeledReconstructedPoints, charReconstructedList, labeledReconstructedPointsList},    
    (* Estrae i valori dalla lista di input *)
    corruptedString = ComputedValues[[1]];
    errorsPos = ComputedValues[[2]];
    strLen = ComputedValues[[3]];
    maxErrors = ComputedValues[[4]];
	isCharacterAdded = ComputedValues[[5]];
    (* Converte la stringa corrotta in una lista di valori ASCII *)
    asciiList = ToCharacterCode[corruptedString];

    (* Converte la stringa corrotta in una lista di caratteri *)
    charList = Characters[corruptedString];

    (* Calcola la posizione dei caratteri non corrotti *)
    charsPos = Complement[Range[strLen], errorsPos];

	If[isCharacterAdded, charList[[Last[Position[charList, "\[AGrave]"]]]] = " "];

    (* Costruisce una lista di punti con etichette per il grafico *)
    points = Transpose[{charsPos, asciiList}]; 
    labeledPoints = Transpose[{points, charList}];
    labeledPointsList = Labeled[#[[1]], #[[2]], Above] & /@ labeledPoints;
	
    (* Ricostruisce il polinomio interpolante *)
    reconstructedPoly = Expand[InterpolatingPolynomial[points, x]];

    (* Richiama la funzione per ricostruire la stringa originale *)
    ReconstructStringResult = ReconstructString[reconstructedPoly, strLen, maxErrors];
    
    recStr = ReconstructStringResult[[1]];
    
    reconstructedPointsList = ReconstructStringResult[[2]];
    
    charReconstructedList = Characters[recStr]; 
    
    labeledReconstructedPoints = Transpose[{reconstructedPointsList, charReconstructedList}];
    labeledReconstructedPointsList = Labeled[#[[1]], #[[2]], Above] & /@ labeledReconstructedPoints;
    
    (* Richiama la funzione per visualizzare il polinomio ricostruito *)
    grafici = PlotReconstructedPoly[reconstructedPoly, labeledPointsList, strLen, labeledReconstructedPointsList, recStr];
    
    toReturn = {recStr, grafici, isCharacterAdded};
    Return[toReturn];
]


(* Definizione della funzione PlotReconstructedPoly *)
PlotReconstructedPoly[reconstructedPoly_, labeledPointsList_List, strLen_Integer, reconstructedPointsList_, recStr_] :=
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
          LabelStyle -> {FontSize -> 18}]],
          
       Show[Plot[reconstructedPoly, {x, 1, StringLength[recStr]}, PlotStyle -> Red,
          AxesLabel -> {"x", "ASCII Value"}, ImageSize -> {600, 400}, 
          LabelStyle -> {FontSize -> 18}],
          
        ListPlot[reconstructedPointsList, PlotStyle -> {PointSize[Large], Blue},
          AxesLabel -> {"x", "ASCII Value"}, ImageSize -> {600, 400}, 
          LabelStyle -> {FontSize -> 18}]]
    }]
]]


(* Definizione della funzione ReconstructString *)
ReconstructString[reconstructedPoly_, strLen_Integer, maxErrors_Integer] :=
Module[{reconstructedPoints,reconstructedAsciiList,reconstructedString, Result  },
(* Calcola i punti ricostruiti *)
reconstructedPoints = Table[{i, reconstructedPoly /. x -> i}, {i, strLen - maxErrors}];
(* Ottiene la lista di valori ASCII dai punti ricostruiti *)
reconstructedAsciiList = Round[Last /@ reconstructedPoints];

(* Converte la lista di valori ASCII in una stringa ricostruita *)
reconstructedString = FromCharacterCode[reconstructedAsciiList];

Result = {reconstructedString, reconstructedPoints};

Return[Result];
]


End[]

EndPackage[]
