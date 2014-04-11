package com.jalil.neuralnet;

import com.jalil.munge.*;

public class NNParameters implements Parameters {

	public double randomInitBound;
	public double lambda;
	public int[] hiddenLayerSize;

	public NNParameters(double randomInitBound, double lambda, int ... hiddenLayerSize) {
		this.randomInitBound = randomInitBound;
		this.lambda = lambda;
		this.hiddenLayerSize = hiddenLayerSize;
	}

	public String toString() {
		StringBuilder output = new StringBuilder();
		output.append("( randomInitBound: " + randomInitBound);
		output.append(", lambda: " + lambda);
		output.append(", hidden layer size:");
		for (int size : hiddenLayerSize) {
			output.append(" " + size); 
		}
		output.append(" )");
		return output.toString();
	}
}