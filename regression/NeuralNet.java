package com.jalil.neuralnet.regression;

import com.jalil.munge.*;
import com.jalil.neuralnet.*;
import com.jalil.math.*;
import java.io.*;
import java.util.*;

/*
* Caution : No more than 50,000,000 entry (# of examples times # of features)
* Therefore it is smart to raise the quality of features not the number of them
* Set java heap space in the command line: java -Xmx2g ...
*/

public class NeuralNet implements Regression<NNParameters, NNEstimate> {

	class Intermediate {
		public int nlayer;
		public int[] nrow, ncol;
		public double[][] actions;
		public double[][] errors;
		public double[][][] gradient;

		public Intermediate(int nfeature, int[] hiddenLayerSize) {
			nlayer = hiddenLayerSize.length + 1;
			nrow = new int[nlayer];
			ncol = new int[nlayer];
			actions = new double[nlayer][];
			errors = new double[nlayer][];;
			gradient = new double[nlayer][][];
			nrow[0] = hiddenLayerSize[0];
			ncol[0] = nfeature + 1;
			for (int l = 1; l < nlayer - 1; l++) {
				nrow[l] = hiddenLayerSize[l];
				ncol[l] = hiddenLayerSize[l - 1] + 1;
			}
			nrow[nlayer - 1] = 1;
			ncol[nlayer - 1] = hiddenLayerSize[nlayer - 2] + 1;		
			for (int l = 0; l < nlayer; l++) {
				actions[l] = new double[nrow[l]];
				errors[l] = new double[nrow[l]];
				gradient[l] = new double[nrow[l]][ncol[l]];
			}
		}

		public Intermediate(NNEstimate estimate) {
			nlayer = estimate.theta.length;
			nrow = new int[nlayer];
			ncol = new int[nlayer];
			actions = new double[nlayer][];
			errors = new double[nlayer][];;
			gradient = new double[nlayer][][];
			for (int l = 0; l < nlayer; l++) {
				nrow[l] = estimate.theta[l].length;
				ncol[l] = estimate.theta[l][0].length;
				actions[l] = new double[nrow[l]];
				errors[l] = new double[nrow[l]];
				gradient[l] = new double[nrow[l]][ncol[l]];
			}
		}

		public void reset() {
			for (int l = 0; l < nlayer; l++)
				for (int r = 0; r < nrow[l]; r++) {
					actions[l][r] = 0;
					errors[l][r] = 0;
					for (int c = 0; c < ncol[l]; c++)
						gradient[l][r][c] = 0;
				}
		}

	}

	private double[] yTrain;
	private double[][] xTrain;
	private double[] yCrossValid;
	private double[][] xCrossValid;
	private NNParameters pars;
	//private NNEstimate estimate;
	//double[][][] fittedTheta;
	//int[] hiddenLayerSize;

	private int MAXITER = 50;
	private double ALPHA = 0.01; // 0.0003 for 28*28 image and 0.001 for 14*14 (grayscale)
	private final double THRESHOLD = 0.00001;

	public void importData(double[] yTrain, double[][] xTrain, double[] yCrossValid, double[][] xCrossValid) {
		this.yTrain = yTrain;
		this.xTrain = xTrain;
		this.yCrossValid = yCrossValid;
		this.xCrossValid = xCrossValid;
	}

	public void setOptimPars(int maxiter, double alpha) {
		this.MAXITER = maxiter;
		this.ALPHA = alpha;
	}

	public void setParameters(NNParameters pars) {
		this.pars = pars;
	}

	public double predict(NNEstimate estimate, double[] xTest) {
		Intermediate intermediate = new Intermediate(estimate);
		int nlayer = intermediate.nlayer;
		actions(intermediate, estimate, xTest);
		return intermediate.actions[nlayer - 1][0];
	}

	public double[] predictDerivative(NNEstimate estimate, double[] xTest) {
		Intermediate intermediate = new Intermediate(estimate);
		int nlayer = intermediate.nlayer;
		actions(intermediate, estimate, xTest);
		int n = xTest.length;
		double[] derivatives = new double[n];
		double[][] derivActions = new double[nlayer][];
		for (int l = 0; l < nlayer; l++) {
			derivActions[l] = new double[intermediate.nrow[l]];
		}
		for (int j = 0; j < n; j++) {
			for (int r = 0; r < intermediate.nrow[0]; r++) {
				derivActions[0][r] = intermediate.actions[0][r] * (1 - intermediate.actions[0][r]) * estimate.theta[0][r][j + 1];
			}			
			for (int l = 1; l < nlayer - 1; l++) {
				for (int r = 0; r < intermediate.nrow[l]; r++) {
					derivActions[l][r] = 0;
					for (int c = 1; c < intermediate.ncol[l]; c++)
						derivActions[l][r] += estimate.theta[l][r][c] * derivActions[l - 1][c - 1];
					derivActions[l][r] *=  intermediate.actions[l][r] * (1 - intermediate.actions[l][r]);
				}
			}
			derivActions[nlayer - 1][0] = 0;
			for (int c = 1; c < intermediate.ncol[nlayer - 1]; c++) {
				derivActions[nlayer - 1][0] += estimate.theta[nlayer - 1][0][c] * derivActions[nlayer - 2][c - 1];
			}
			derivatives[j] = derivActions[nlayer - 1][0];
		}
			
		return derivatives;
	}

	public NNEstimate stochasticLearn(boolean silent) {
		if((xTrain == null) || (yTrain == null) || (xCrossValid == null) || (yCrossValid == null))
			throw new IllegalArgumentException("Training data has no example.");
		System.out.println("learning ...");
		int n = xTrain[0].length;
		int mTrain = yTrain.length;
		int mCrossValid = yCrossValid.length;
		NNEstimateFactory factory = new NNEstimateFactory();
		NNEstimate estimate = factory.createEstimate(n, pars);
		Intermediate intermediate = new Intermediate(n, pars.hiddenLayerSize);
		int nlayer = intermediate.nlayer;
		for (int l = 0; l < nlayer; l++) {
			//double randomInitBound = Math.sqrt(6) / (Math.sqrt(intermediate.nrow[l] + intermediate.ncol[l] - 1));
			for (int r = 0; r < intermediate.nrow[l]; r++)
				for (int c = 0; c < intermediate.ncol[l]; c++)
					estimate.theta[l][r][c] = Math.random() * 2 * pars.randomInitBound - pars.randomInitBound;
		}
		double trainCost = trainCost(intermediate, estimate, 0, mTrain);
		double crossValidCost = crossValidCost(intermediate, estimate, 0, mCrossValid);
		NNEstimate bestEstimate = estimate.clone();
		double bestCrossValidCost = crossValidCost;		
		double swap;
		for (int i = 0; i < MAXITER; i++) {
			System.out.println("Cost of epoch\t" + i + "\t= " + trainCost);
			System.out.println("Evaluating the hypothesis ...");
			System.out.println("Cost on cross-validation set: " + crossValidCost);				
			System.out.println();
			for (int example = 0; example < mTrain; example++) {
				update(intermediate, estimate, example, example + 1);
			}
			swap = trainCost;
			trainCost = trainCost(intermediate, estimate, 0, mTrain);
			crossValidCost = crossValidCost(intermediate, estimate, 0, mCrossValid);
			if(crossValidCost < bestCrossValidCost) {
				bestEstimate = estimate.clone();
				bestCrossValidCost = crossValidCost;
			}
			if(Math.abs(trainCost - swap) < THRESHOLD * (trainCost + THRESHOLD)) {
				System.out.println("Threshold has reached.");
				break;
			}
		}	
		//return bestEstimate;		
		return estimate;
	}

	public double generalizationError(NNEstimate estimate) {
		int mCrossValid = yCrossValid.length;
		Intermediate intermediate = new Intermediate(estimate);
		return crossValidCost(intermediate, estimate, 0, mCrossValid);
	}

	@SuppressWarnings("unchecked")
	public NNEstimateFactory getEstimateFactory() {
		return new NNEstimateFactory();
	}

	public NNEstimate batchLearn(boolean silent) { return null;}

	private double sigmoid(double z) {return 1/(1+Math.exp(-z));}

	private void actions(Intermediate intermediate, NNEstimate estimate, double[] xTest) {
		int nlayer = intermediate.nlayer;
		double crossprod;
		for (int r = 0; r < intermediate.nrow[0]; r++) {
			crossprod = estimate.theta[0][r][0];
			for (int c = 1; c < intermediate.ncol[0]; c++) {
				crossprod += xTest[c - 1] * estimate.theta[0][r][c];
			}
			intermediate.actions[0][r] = sigmoid(crossprod);
		}
		for (int l = 1; l < nlayer - 1; l++) {
			for (int r = 0; r < intermediate.nrow[l]; r++) {
				crossprod = estimate.theta[l][r][0];
				for (int c = 1; c < intermediate.ncol[l]; c++)
					crossprod += intermediate.actions[l - 1][c - 1] * estimate.theta[l][r][c];
				intermediate.actions[l][r] = sigmoid(crossprod);
			}
		}
		crossprod = estimate.theta[nlayer - 1][0][0];;
		for (int c = 1; c < intermediate.ncol[nlayer - 1]; c++) {
			crossprod += intermediate.actions[nlayer - 2][c - 1] * estimate.theta[nlayer - 1][0][c];
		}
		intermediate.actions[nlayer - 1][0] = crossprod;		
	}

	private double trainCost(Intermediate intermediate, NNEstimate estimate, int fromExample, int toExample) {
		int nlayer = intermediate.nlayer;
		int mTrain = yTrain.length;
		double cost = 0;
		for (int example = fromExample; example < toExample; example++) {
			actions(intermediate, estimate, xTrain[example]); 
			cost += cost(intermediate, yTrain[example]);
		}
		cost /= (2*(toExample - fromExample));
		double penalty = 0;
		for (int l = 0; l < nlayer; l++) {
			for (int r = 0; r < intermediate.nrow[l]; r++)
				for (int c = 1; c < intermediate.ncol[l]; c++)
					penalty += estimate.theta[l][r][c]*estimate.theta[l][r][c];
		}
		penalty *= pars.lambda / (2*(toExample - fromExample));

		return cost + penalty;
	}
	
	private double crossValidCost(Intermediate intermediate, NNEstimate estimate, int fromExample, int toExample) {
		int nlayer = intermediate.nlayer;
		int mCrossValid = yCrossValid.length;
		double cost = 0;
		for (int example = fromExample; example < toExample; example++) {
			actions(intermediate, estimate, xCrossValid[example]); 
			cost += cost(intermediate, yCrossValid[example]);
		}
		cost /= (2*(toExample - fromExample));

		return cost;
	}

	private double cost(Intermediate intermediate, double yTest) {
		int nlayer = intermediate.nlayer;
		return (yTest - intermediate.actions[nlayer - 1][0])*(yTest - intermediate.actions[nlayer - 1][0]);
	}

	private void gradient(Intermediate intermediate, NNEstimate estimate, double[] xTest, double yTest) {
		int nlayer = estimate.theta.length;
		intermediate.errors[nlayer - 1][0] = intermediate.actions[nlayer - 1][0] - yTest;
		intermediate.gradient[nlayer - 1][0][0] += intermediate.errors[nlayer - 1][0];
		for (int c = 1; c < intermediate.ncol[nlayer - 1]; c++) {
			intermediate.gradient[nlayer - 1][0][c] += intermediate.errors[nlayer - 1][0] * intermediate.actions[nlayer - 2][c - 1];
		}
		for (int l = nlayer - 2; l > 0; l--) {
			for (int r = 0; r < intermediate.nrow[l]; r++) {
				intermediate.errors[l][r] = 0;
				for (int nextr = 0; nextr < intermediate.nrow[l + 1]; nextr++) {
					intermediate.errors[l][r] += intermediate.errors[l + 1][nextr] * estimate.theta[l + 1][nextr][r + 1];
				}
				intermediate.errors[l][r] *= intermediate.actions[l][r] * (1 - intermediate.actions[l][r]);
				intermediate.gradient[l][r][0] += intermediate.errors[l][r];
				for (int c = 1; c < intermediate.ncol[l]; c++) {
					intermediate.gradient[l][r][c] += intermediate.errors[l][r] * intermediate.actions[l - 1][c - 1];
				}
			}
		}
		for (int r = 0; r < intermediate.nrow[0]; r++) {
			intermediate.errors[0][r] = 0;
			for (int nextr = 0; nextr < intermediate.nrow[1]; nextr++) {
				intermediate.errors[0][r] += intermediate.errors[1][nextr] * estimate.theta[1][nextr][r + 1];
			}
			intermediate.errors[0][r] *= intermediate.actions[0][r] * (1 - intermediate.actions[0][r]);
			intermediate.gradient[0][r][0] += intermediate.errors[0][r];
			for (int c = 1; c < intermediate.ncol[0]; c++) {
				intermediate.gradient[0][r][c] += intermediate.errors[0][r] * xTest[c - 1];
			}
		}
	}

	private void update(Intermediate intermediate, NNEstimate estimate, int fromExample, int toExample) {
		int nlayer = intermediate.nlayer;
		/*if((xTrain == null) || (yTrain == null) || (xCrossValid == null) || (yCrossValid == null))
			throw new IllegalArgumentException("Training data has no example.");
		if(xTest[0].length + 1 != estimate.theta[0][0].length)
			throw new IllegalArgumentException("Parameter size is not compatible with the length of the example.");
		if(nlayer == 1)
			throw new IllegalArgumentException("Use Logit instead if there is no hidden layer.");
		if(estimate.theta[nlayer - 1].length != 1)
			throw new IllegalArgumentException("More than one node in the output layer.");
		for (int l = 1; l < nlayer; l++) {
			if(estimate.theta[l][0].length != estimate.theta[l - 1].length + 1)
				throw new IllegalArgumentException("The dimensions of theta is not valid." + (l + 1));
		}*/
		int mTrain = yTrain.length;
		double[] xExample;
		double yExample;
		intermediate.reset();
		for (int i = fromExample; i < toExample; i++) {
			xExample = xTrain[i];
			yExample = yTrain[i];
			actions(intermediate, estimate, xExample);
			gradient(intermediate, estimate, xExample, yExample);
		}
		for (int l = 0; l < nlayer; l++)
			for (int r = 0; r < intermediate.nrow[l]; r++) {
				intermediate.gradient[l][r][0] /= (toExample - fromExample);
				estimate.theta[l][r][0] -= ALPHA * intermediate.gradient[l][r][0];
				for (int c = 1; c < intermediate.ncol[l]; c++) {
					intermediate.gradient[l][r][c] /= (toExample - fromExample);
					estimate.theta[l][r][c] -= ALPHA * (intermediate.gradient[l][r][c] + pars.lambda / (toExample - fromExample) * estimate.theta[l][r][c]);
				}
			}
	}
	
}
