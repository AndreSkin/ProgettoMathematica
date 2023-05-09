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

StrEncode::usage = "Encode[x] prende in input una stringa x e la espande."
ComputedValues::usage = "ComputedValues \[EGrave] una variabile pubblica che contiene i valori computati da StrEncode."

Begin["`Private`"]

StrEncode[inputString_String, maxErrors_Integer]:=
DynamicModule[{i},
asciiList =ToCharacterCode[inputString];
points = Transpose[{Range[Length[asciiList]],asciiList}];
polynomial = Expand[InterpolatingPolynomial[points, x]];
expandedPoints=Table[{i,polynomial/. x->i},{i,Length[points]+maxErrors}];
charList=Characters[inputString];

str = inputString;

fDev = D[polynomial, x];
fDevSign = Sign[fDev /. x -> asciiList[[-1]]];
If[fDevSign < 0, 
(str = inputString <> "x"; 
asciiList =ToCharacterCode[str]; 
points = Transpose[{Range[Length[asciiList]],asciiList}];
charList=Characters[str];
labeledPoints=Transpose[{points,charList}];
labeledPointsList=Labeled[#[[1]],#[[2]],Above]&/@labeledPoints;
polynomial = Expand[InterpolatingPolynomial[points, x]]);
expandedPoints=Table[{i,polynomial/. x->i},{i,Length[points]+maxErrors}];
,Print["All good"]];

Column[
	{
		PlotString[polynomial,asciiList, charList, points],
		ExpandString[polynomial,expandedPoints],
		StringCorruption[expandedPoints,maxErrors];
	}
]

]



PlotString[polynomial_,asciiList_List, charlist_List, points_List]:=
Module[{i},
	labeledPoints=Transpose[{points,charlist}];
	labeledPointsList=Labeled[#[[1]],#[[2]],Above]&/@labeledPoints;
	Column[
		{
			ListPlot[labeledPointsList,PlotStyle->{PointSize[Medium],Blue},AxesLabel->{"x","ASCII Value"},ImageSize->Medium],
			PlotPoly[polynomial,asciiList, points, labeledPointsList]
		}
	]
]


PlotPoly[polynomial_,asciiList_List, points_List, labeledPointsList_List]:=
Module[{i},
	Show[
		Plot[polynomial,{x,1,Length[asciiList]},PlotStyle->Red,AxesLabel->{"x","ASCII Value"},ImageSize->Medium],
		ListPlot[labeledPointsList,PlotStyle->{PointSize[Medium], Blue}]
	]
]


ExpandString[polynomial_,expandedPoints_List]:=
Module[{i},
asciiListExpanded=Round[Last/@expandedPoints];
stringExpanded = FromCharacterCode[asciiListExpanded];
charListExpanded=FromCharacterCode/@asciiListExpanded;
labeledPointsExpanded=Transpose[{expandedPoints,charListExpanded}];
labeledPointsExpandedList=Labeled[#[[1]],#[[2]],Above]&/@labeledPointsExpanded;

Show[
	Plot[polynomial,{x,1,Length[expandedPoints]},PlotStyle->Red,AxesLabel->{"x","ASCII Value"},ImageSize->Medium],
	ListPlot[labeledPointsExpandedList,PlotStyle->{PointSize[Medium],Blue}]
]

]


StringCorruption[expandedPoints_List,maxErrors_Integer]:=
Module[{i},
	asciiListExpanded=Round[Last/@expandedPoints];
	stringExpanded = FromCharacterCode[asciiListExpanded];
	nErrors = RandomInteger[{1, maxErrors}];
	errorsPos = RandomSample[Range[StringLength[stringExpanded]], nErrors];
	
	myStringAsList = Characters[stringExpanded];
	myStringAsList = Delete[myStringAsList, List /@ errorsPos];
	corruptedString = StringJoin[myStringAsList];
	
	(*msg_received =  [corruptedString, errorsPos, strLen, maxErrors];*)
	StrLen = StringLength[stringExpanded];
	ComputedValues = {corruptedString, errorsPos,StrLen, maxErrors};
	Print[ComputedValues]
]


End[]

EndPackage[]

