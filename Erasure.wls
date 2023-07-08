#!/usr/bin/env wolframscript
(* ::Package:: *)

(* :Title: ENC.wls *)
(* :Context: Erasure` *)
(* :Author: Gruppo 1 *)
(* :Summary: Funzione per fare bit flip *)
(* :Copyright: CC BY-SA*)
(* :Package Version: 1 *)
(* :Mathematica Version: 13.3 *)
(* :History: last modified 8/07/23 *)
(* :Keywords: Reed-Solomon, encryption*)
(* :Sources: V. tutorial.nb*)
(* :Limitations: this is for educational purposes only. *)
(* :Discussion: *)
(* :Requirements: *)


BeginPackage["Erasure`"]
(* Definizione dell'uso della funzione StrEncode *)
CorruptBit::usage = "CorruptBit[stringa] effettua bit flip di un bit casuale della stringa data"

Begin["`Private`"]


(*Funzione per stampare il risultato*)
StampaRisposta[lista_, newlista_, FlippedString_] :=
  Module[{},
      (* Unisce gli elementi di lista in una stringa separati da virgole *)
      Column[{StringJoin["La rappresentazione binaria di questa stringa \[EGrave]:\n ", StringJoin[ToString /@ Riffle[lista, ", "]]],
        (* Unisce gli elementi di newlista in una stringa separati da virgole *)
        StringJoin["Ora succede qualche catastrofe e otteniamo:\n ", StringJoin[ToString /@ Riffle[newlista, ", "]]],
        (* Mostra la stringa FlippedString *)
        StringJoin["Dopo la catastrofe la stringa \[EGrave]: ", FlippedString]
      }]
  ]

(* Funzione per corrompere un bit *)
(* Controlla se la stringa non \[EGrave] vuota *)
CorruptBit[string_String] :=
  Module[{str = string, bitStringa, lista, newlista, lunghezza, numeroCasuale, binaryList, binaryString, FlippedString},
    If[StringLength[str] > 0,
      (* Converte la stringa in una sequenza di codici ASCII e successivamente in una rappresentazione binaria *)
      bitStringa = IntegerDigits[ToCharacterCode[str], 2, 8];
      (* Appiattisce la lista di bit *)
      lista = Flatten[bitStringa];
      (* Copia la lista originale nella lista modificata *)
      newlista = lista;
      (* Calcola la lunghezza della lista modificata *)
      lunghezza = Length[newlista];
      (* Genera un numero casuale tra 1 e la lunghezza della lista *)
      numeroCasuale = RandomInteger[{1, lunghezza}];
      (* Inverte un bit casuale nella lista modificata *)
      newlista = ReplacePart[newlista, numeroCasuale -> 1 - lista[[numeroCasuale]]];
      (* Imposta Mostra su vero per mostrare il risultato *)
      If[Length[newlista] > 0,
        binaryList = newlista;
        binaryString = StringJoin[ToString /@ binaryList];
        (* Ricrea una stringa a partire dalla lista di bit modificata *)
        FlippedString = FromCharacterCode[FromDigits[#, 2] & /@ StringPartition[binaryString, 8]];
      ];
      If[Length[lista]>0,
      StampaRisposta[lista, newlista, FlippedString]]
    ]
  ]



End[]

EndPackage[]
