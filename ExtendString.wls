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


BeginPackage["ExtendString`"]
(* Definizione dell'uso della funzione StrEncode *)
ExtendString::usage = "ExtendString[stringa,fattore] "

Begin["`Private`"]




(*Dichiarazione delle variabili*)
originalString="";  (*Variabile per memorizzare la stringa originale*)
stringaEstesa="";  (*Variabile per memorizzare la stringa estesa*)
repet=0;  (*Variabile per memorizzare il fattore di ripetizione*)
corruptedStringRip="";  (*Variabile per memorizzare la stringa corrotta*)
numCharToDelete=0;  (*Variabile per memorizzare il numero di caratteri da eliminare*)

(*Variabile per gestire la visualizzazione del risultato*)

(*Definizione della funzione stamp*)
Stamp[]:=Module[{},Column[
Print["aoo1"];
If[StringLength[corruptedStringRip] != 0,
Print["aoo2"];
      StringJoin["La stringa estesa \[EGrave]: ", stringaEstesa], ""],
If[StringLength[corruptedStringRip] != 0,
      StringJoin["La stringa corrotta \[EGrave]: ", corruptedStringRip], ""]
(*Restituisce un messaggio in base alla risposta data dall'utente*)
(*Dynamic[err],*)
]]


(*Definizione della funzione extendString*)
ExtendString[stringa_,fattoreRipetizione_]:=Module[{str=stringa,rep=fattoreRipetizione},
(*Salva la stringa originale nella variabile originalString*)
originalString=str;
If[fattoreRipetizione >=0,
stringaEstesa=StringJoin[Riffle[StringRepeat[#,fattoreRipetizione]&/@Characters[stringa],""]];
repet=fattoreRipetizione;
corruptedStringRip=stringaEstesa;
generateCorruptedString[];
, Print["Il fattore di ripetizione deve essere maggiore di 0"]]
];

(*Definizione della funzione generateCorruptedString*)
generateCorruptedString[]:=Module[{lunghezzaStringa,toDelete},

(*Ottiene la lunghezza della stringa estesa*)
lunghezzaStringa=StringLength[stringaEstesa];

(*Genera un numero casuale di caratteri da eliminare tra 1 e la lunghezza della stringa*)
numCharToDelete=RandomInteger[{1,lunghezzaStringa}];
(*Genera una lista casuale di posizioni di caratteri da eliminare*)
toDelete=RandomSample[Range[1,lunghezzaStringa],numCharToDelete];

(*Crea la stringa corrotta sostituendo i caratteri alle posizioni specificate con "_"*)
corruptedStringRip=StringReplacePart[stringaEstesa,ConstantArray["_",numCharToDelete],Thread[{toDelete,toDelete}]];
Print["aoo0"];
Stamp[]
];  


  

(*PrintErrors[numErrors_]:= DynamicModule[{}, Dynamic[
Dynamic[
If[quizVisible,Column[{"\nQuanti errori sono avvenuti?",InputField[Dynamic[numErrors],Number],
If[numErrors==numCharToDelete,
"Corretto!",
"Sbagliato! Confronta attentamente la stringa estesa e quella contenente gli errori"]
}]
]]]*)


End[]

EndPackage[]
