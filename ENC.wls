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

Begin["`Private`"]

StrEncode[inputString_, maxErrors_]:=
DynamicModule[{i},
asciiList =ToCharacterCode[inputString];
points = Transpose[{Range[Length[asciiList]],asciiList}];
polynomial = Expand[InterpolatingPolynomial[points, x]];
expandedPoints=Table[{i,polynomial/. x->i},{i,Length[points]+maxErrors}];

StringCorruption[expandedPoints,maxErrors];
Column[
	{
		PlotString[asciiList, inputString, points],
		ExpandString[expandedPoints]
	}
]

]



PlotString[asciiList_, inputString_, points_]:=
Module[{i},
	charList=Characters[inputString];
	labeledPoints=Transpose[{points,charList}];
	labeledPointsList=Labeled[#[[1]],#[[2]],Above]&/@labeledPoints;
	Column[
		{
			ListPlot[labeledPointsList,PlotStyle->{PointSize[Medium],Blue},AxesLabel->{"x","ASCII Value"},ImageSize->Medium],
			PlotPoly[asciiList, points, labeledPointsList]
		}
	]
]


PlotPoly[asciiList_, points_, labeledPointsList_]:=
Module[{i},
	Show[
		Plot[polynomial,{x,1,Length[asciiList]},PlotStyle->Red,AxesLabel->{"x","ASCII Value"},ImageSize->Medium],
		ListPlot[labeledPointsList,PlotStyle->{PointSize[Medium], Blue}]
	]
]


ExpandString[expandedPoints_]:=
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


StringCorruption[expandedPoints_,maxErrors_]:=
Module[{i},
	asciiListExpanded=Round[Last/@expandedPoints];
	stringExpanded = FromCharacterCode[asciiListExpanded];
	nErrors = RandomInteger[{1, maxErrors}];
	errorsPos = RandomSample[Range[StringLength[stringExpanded]], nErrors];
	
	myStringAsList = Characters[stringExpanded];
	myStringAsList = Delete[myStringAsList, List /@ errorsPos];
	corruptedString = StringJoin[myStringAsList];
	
	strLen = StringLength[stringExpanded];
	Print[{nErrors, errorsPos, corruptedString}]
]


End[]

EndPackage[]

