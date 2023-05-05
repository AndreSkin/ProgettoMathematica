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

Begin["`Private`"]

MsgDecode[input_List]:=
DynamicModule[{i},
(*input =  {corruptedString, errorsPos, strLen, maxErrors};*)
corruptedString = input[[1]];
errorsPos = input[[2]];
strLen = input[[3]];
maxErrors = input[[4]];

asciiList =ToCharacterCode[corruptedString];
charList = Characters[corruptedString];
charsPos=Complement[Range[strLen], errorsPos];
points = Transpose[{charsPos,asciiList}];
labeledPoints=Transpose[{points,charList}];
labeledPointsList=Labeled[#[[1]],#[[2]],Above]&/@labeledPoints;
reconstructedPoly = Expand[InterpolatingPolynomial[points,x]];

ReconstructString[reconstructedPoly, strLen, maxErrors];
PlotReconstructedPoly[reconstructedPoly,labeledPointsList,strLen]
]



PlotReconstructedPoly[reconstructedPoly_,labeledPointsList_List,strLen_Integer]:=
Module[{i},
	Column[
		{
			ListPlot[labeledPointsList,PlotStyle->{PointSize[Medium],Blue},AxesLabel->{"x","ASCII Value"},ImageSize->Medium],
			Show[
				Plot[reconstructedPoly,{x,1,strLen},PlotStyle->Red,AxesLabel->{"x","ASCII Value"},ImageSize->Medium],
				ListPlot[labeledPointsList,PlotStyle->{PointSize[Medium],Blue},AxesLabel->{"x","ASCII Value"},ImageSize->Medium]
			]
		}
	]
]


ReconstructString[reconstructedPoly_, strLen_Integer, maxErrors_Integer]:=
Module[{i},
	reconstructedPoints = Table[{i,reconstructedPoly/. x->i},{i,strLen-maxErrors}];
	reconstructedAsciiList=Round[Last/@reconstructedPoints];
	
	reconstructedString = FromCharacterCode[reconstructedAsciiList];
	Print[reconstructedString];
	]


End[]

EndPackage[]

