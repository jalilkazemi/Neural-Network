package com.jalil.neuralnet.regression;

import com.jalil.munge.*;
import com.jalil.neuralnet.*;
import java.util.regex.*;

public class NNEstimateFactory implements EstimateFactory<NNParameters, NNEstimate> {

	public NNEstimate createEstimate(int nfeature, NNParameters parameters) {
		NNEstimate estimate = new NNEstimate(nfeature, parameters.hiddenLayerSize);
		return estimate;
	}

	public NNEstimate valueOf(String estimateString) {
		NNEstimate estimate = new NNEstimate();
		Pattern layer_p = Pattern.compile("\\blayer:\\s*\\d+");
		Matcher layer_m = layer_p.matcher(estimateString);
		estimate.nlayer = 0;
		while(layer_m.find())
			estimate.nlayer++;
		estimate.nrow = new int[estimate.nlayer];
		estimate.ncol = new int[estimate.nlayer];
		estimate.theta = new double[estimate.nlayer][][];
		Pattern theta_p = Pattern.compile("\\btheta:\\s*\\{\\s*([^\\{\\}]+)\\s*\\}");
		Matcher theta_m = theta_p.matcher(estimateString);
		for (int l = 0; l < estimate.nlayer; l++) {
			if(!theta_m.find())
				throw new IllegalArgumentException("Failed to find layer " + (l + 1));
			String[] rowFields = theta_m.group(1).split(";");
			estimate.nrow[l] = rowFields.length;
			estimate.theta[l] = new double[estimate.nrow[l]][];
			for (int r = 0; r < estimate.nrow[l]; r++) {
			 	String[] colFields = rowFields[r].split(",");
			 	if((r > 0) && (colFields.length != estimate.ncol[l]))
			 		throw new IllegalArgumentException("Matrix column length varies across rows.");
			 	estimate.ncol[l] = colFields.length;
			 	estimate.theta[l][r] = new double[estimate.ncol[l]];
			 	for (int c = 0; c < estimate.ncol[l]; c++) {
			 		estimate.theta[l][r][c] = Double.parseDouble(colFields[c]);
			 	}
			}
		}
		estimate.nfeature = estimate.ncol[0] - 1;
		estimate.hiddenLayerSize = new int[estimate.nlayer - 1];
		for (int l = 0; l < estimate.nlayer - 1; l++) {
			estimate.hiddenLayerSize[l] = estimate.nrow[l];
		}
		for (int l = 1; l < estimate.nlayer; l++) {
			if(estimate.ncol[l] != estimate.nrow[l - 1] + 1)
				throw new IllegalArgumentException("The theta matricies doesn't match in dimension.");
		}
		if(estimate.nrow[estimate.nlayer - 1] != 1)
			throw new IllegalArgumentException("Last layer is of size more than one.");
		
		return estimate;
	}
		
}