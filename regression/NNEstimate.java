package com.jalil.neuralnet.regression;

import com.jalil.munge.*;
import java.util.regex.*;

public class NNEstimate implements Estimate {
	protected int nfeature;
	protected int nlayer;
	protected int[] nrow, ncol;
	protected int[] hiddenLayerSize;
	public double[][][] theta;

	protected NNEstimate() {}

	protected NNEstimate(int nfeature, int... hiddenLayerSize) {
		this.nfeature = nfeature;
		this.hiddenLayerSize = hiddenLayerSize;
		nlayer = hiddenLayerSize.length + 1;
		nrow = new int[nlayer];
		ncol = new int[nlayer];
		theta = new double[nlayer][][];
		theta[0] = new double[hiddenLayerSize[0]][nfeature + 1];
		for (int l = 1; l < nlayer - 1; l++) {
			theta[l] = new double[hiddenLayerSize[l]][hiddenLayerSize[l - 1] + 1];
		}
		theta[nlayer - 1] = new double[1][hiddenLayerSize[nlayer - 2] + 1];	
		for (int l = 0; l < nlayer; l++) {
			nrow[l] = theta[l].length;
			ncol[l] = theta[l][0].length;
		}
	}

	public NNEstimate clone() {
		NNEstimate clonedEstimate = new NNEstimate(nfeature, hiddenLayerSize);
		int nlayer = hiddenLayerSize.length + 1;
		for (int l = 0; l < nlayer; l++) {
			int nrow = this.theta[l].length, ncol = this.theta[l][0].length;
			for (int r = 0; r < nrow; r++)
				for (int c = 0; c < ncol; c++)
					clonedEstimate.theta[l][r][c] = this.theta[l][r][c];
		}

		return clonedEstimate;
	}

	public String toString() {
		StringBuilder output = new StringBuilder();
		output.append("[\n");
		for (int l = 0; l < nlayer; l++) {
			output.append("{\n");
			output.append("layer: " + (l + 1) + "\n");
			output.append(", ");
			output.append("theta: ");
			output.append("{\n");
			for (int r = 0; r < nrow[l]; r++) {
				for (int c = 0; c < ncol[l]; c++) {
					output.append(theta[l][r][c] + ",");
				}
				output.deleteCharAt(output.length() - 1);
				output.append(";");
			}
			output.append("\n}");
			output.append("\n}\n");	
			output.append(", ");
		}
		output.delete(output.length() - 2, output.length());
		output.append("\n]\n");

		return output.toString();
	}

}