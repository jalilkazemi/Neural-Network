package com.jalil.neuralnet;
import com.jalil.math.*;
import java.io.*;
import java.util.*;

/*
* Caution : No more than 50,000,000 entry (# of examples times # of features)
* Therefore it is smart to raise the quality of features not the number of them
* Set java heap space in the command line: java -Xmx2g ...
*/

class NNCostGrad {
	public double cost; 
	public double[][][] gradient;
	public NNCostGrad(double cost, double[][][] gradient) {
		this.cost = cost;
		this.gradient = gradient;
	}
}

public class NeuralNet {

	List<Integer> yLst;
	List<double[]> XLst;
	int[] order;
	int[] y;
	double[][] X;
	double[][] Xpoly;
	double[][][] fittedTheta;
	Set<Integer> classes;
	double[] mu, sig;
	boolean onlyForPrediction;
	double[][] kernelBlur;

	final int MAXITER = 100;
	final double ALPHA = 0.01; // 0.0003 for 28*28 image and 0.001 for 14*14 (grayscale)
	final double THRESHOLD = 0.0001;

	public NeuralNet() {
		onlyForPrediction = true;
		setBlurKernel(1.0);
	}

	public NeuralNet(String trainingFile) {
		System.out.println("Loading train set ...");
		onlyForPrediction = false;
		setBlurKernel(1.0);
		BufferedReader in = null;
		yLst = new LinkedList<Integer>();
		XLst = new LinkedList<double[]>();
		classes = new TreeSet<Integer>();
		int featureSize;
		try {
			in = new BufferedReader(new FileReader(trainingFile));
			String scan = in.readLine();
			if(scan == null)
				throw new IllegalArgumentException("File is empty.");
			featureSize = scan.split(",").length - 1;
			int count = 0;
			while((scan = in.readLine()) != null) {
				count++;
				if(count / 1000 * 1000 == count)
					System.out.println(count + "`s example");
				//if(count > 10000)
				//	break;
				String[] fields = scan.split(",");
				yLst.add(Integer.parseInt(fields[0]));
				double[] x = new double[featureSize];
				for(int i = 0; i < featureSize; i++)
					x[i] = Double.parseDouble(fields[i+1]);
				XLst.add(x);
			}
		} catch(IOException e) {
			e.printStackTrace();
		} finally {
			try{in.close();} catch (IOException e) {e.printStackTrace();}
		}
		
		for (int oneClass : yLst) {
			classes.add(oneClass);
		}
	}

	private void setBlurKernel(double sig) {
		int l = 1;
		double sum = 0;
		kernelBlur = new double[2 * l + 1][2 * l + 1];
		for (int i = 0; i < 2 * l + 1; i++) {
			for (int j = 0; j < 2 * l + 1; j++) {
				kernelBlur[i][j] = Math.exp(-((i - l)*(i - l) + (j - l)*(j - l)) / (2 * sig * sig));
				sum += kernelBlur[i][j];
			}
		}
		for (int i = 0; i < 2 * l + 1; i++) {
			for (int j = 0; j < 2 * l + 1; j++) {
				kernelBlur[i][j] = kernelBlur[i][j] / sum;
			}
		}		
	}

	public void partitionData(boolean reorder) {
		if(onlyForPrediction)
			throw new RuntimeException("No training data is loaded.");
		System.out.println("Partitioning started ...");
		int m = yLst.size();
		if(reorder) {
			Double[] sample = new Double[m];
			for (int i = 0; i < m; i++)
				sample[i] = Math.random();
			order = MyMath.sort(sample);			
		} else {
			order = new int[m];
			for (int i = 0; i < m; i++)
				order[i] = i;
		}
	}

	public void extractFeature(boolean preprocess) {
		if(onlyForPrediction)
			throw new RuntimeException("No training data is loaded.");
		System.out.println("Features Extraction started ...");
		int m = yLst.size();
		y = new int[m];
		X = new double[m][];
		for(int i = 0; i < m; i++)
			y[i] = yLst.get(order[i]);
		if(preprocess) {
			for(int i = 0; i < m; i++) {
				double[] x = XLst.get(order[i]);
				X[i] = extractFeature(x);
			}			
		} else {
			for(int i = 0; i < m; i++)
				X[i] = XLst.get(order[i]);
		}
	}

	private double[] extractFeature(double[] x) {
		// Turns 28 by 28 images to 14 by 14.
		double[] xext = new double[196];
		int index = 0;
		for (int i = 0; i < 784; i += 2 * 28) {
			for (int j = 0; j < 28; j += 2) {
				for (int k = 0; k < 2; k++)
					for (int l = 0; l < 2 * 28; l += 28)
						xext[index] += x[i + j + k + l];
				xext[index] /= 4.0;
				index++;
			}
		}
		return xext;
	}

	public double[] blur(double[] x) {
		int l = kernelBlur.length / 2;
		double[] blurred = new double[x.length];
		for (int i = l; i < 14 - l; i++)
			for (int j = l; j < 14 - l; j++)
				for (int di = -l; di < l; di++)
					for (int dj = -l; dj < l; dj++)
						blurred[i * 14 + j] += x[(i + di) * 14 + (j + dj)] * kernelBlur[l + di][l + dj];
		return blurred;
	}

	public void polyFeatureNoRep(int power) {
		if(onlyForPrediction)
			throw new RuntimeException("No training data is loaded.");
		System.out.println("Feature polynomial mapping started ...");
		if(power == 1) {
			Xpoly = X;
			return;
		}
		int m = y.length;
		int n = X[0].length;
		//double[] Xpower = new double[m * n * power];
		int nterm = 0;
		for(int i = 1; i <= power; i++)
			nterm += MyMath.choose(n, i);
		Xpoly = new double[m][nterm];
		int term = 0;
		for(int p = 1; p <= power; p++) {
			System.out.println("Power " + p);
			int[] ijk = new int[p];
			for(int i = 0; i < p; i++)
				ijk[i] = i;
			while(ijk[0] < n - p + 1) {
				for(int i = 0; i < m; i++) {
					Xpoly[i][term] = 1;
					for(int j = 0; j < p; j++)
						Xpoly[i][term] *= X[i][ijk[j]];
				}
				term++;
				int i = p - 1;
				ijk[i]++;
				while(ijk[i] > n - p + i) {
					i--;
					if(i < 0)
						break;
					ijk[i]++;
				} 
				if(i >= 0) {
					for (int j = i + 1; j < p; j++) 
						ijk[j] = ijk[j - 1] + 1;
				}
			}
		}
	}

	public void normalizeFeature() {
		if(onlyForPrediction)
			throw new RuntimeException("No training data is loaded.");
		System.out.println("Feature normalization started ...");
		int n = Xpoly[0].length;
		int m = (int) Math.floor(0.7 * y.length);
		int M = y.length;
		mu = new double[n];
		sig = new double[n];
		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m; i++)
				mu[j] += Xpoly[i][j];
			mu[j] /= m;
			for (int i = 0; i < m; i++) {
				Xpoly[i][j] -= mu[j];
				sig[j] += Xpoly[i][j] * Xpoly[i][j];
			}
			sig[j] = Math.sqrt(sig[j] / m);
			for (int i = m + 1; i < M; i++)
				Xpoly[i][j] -= mu[j];
			if(sig[j] > 0) {
				for (int i = 0; i < M; i++)
					Xpoly[i][j] /= sig[j];
			}
		}
	}

	private double sigmoid(double z) {return 1/(1+Math.exp(-z));}

	public double cost(double lambda, double[][] ... theta) {
		if(onlyForPrediction)
			throw new RuntimeException("No training data is loaded.");
		if(Xpoly[0].length + 1 != theta[0][0].length)
			throw new IllegalArgumentException("Parameter size is not compatible with train examples length.");
		int nclass = classes.size();
		int nlayer = theta.length;
		if(theta[nlayer - 1].length != nclass)
			throw new IllegalArgumentException("Parameter size is not compatible with the number of classes in layer " + (nlayer + 1));
		for (int l = 1; l < nlayer; l++) {
			if(theta[l][0].length != theta[l - 1].length + 1)
				throw new IllegalArgumentException("Parameter size is not compatible with layer size in layer " + (l + 1));
		}
		int m = (int) Math.floor(0.7 * y.length);
		double[][] a = new double[nlayer][];
		for (int l = 0; l < nlayer; l++) {
			a[l] = new double[theta[l].length];
		}
		double totalcost = 0;
		double crossprod;
		for (int i = 0; i < m; i++) {
			int nrow = theta[0].length, ncol = Xpoly[i].length + 1;
			for (int r = 0; r < nrow; r++) {
				crossprod = theta[0][r][0];
				for (int c = 1; c < ncol; c++) {
					crossprod += Xpoly[i][c - 1] * theta[0][r][c];
				}
				a[0][r] = sigmoid(crossprod);
			}
			for (int l = 1; l < nlayer; l++) {
				nrow = theta[l].length; ncol = theta[l][0].length;
				for (int r = 0; r < nrow; r++) {
					crossprod = theta[l][r][0];;
					for (int c = 1; c < ncol; c++) {
						crossprod += a[l - 1][c - 1] * theta[l][r][c];
					}
					a[l][r] = sigmoid(crossprod);
				}
			}
			int cls = 0;
			for (int oneClass : classes) {
				if(y[i] == oneClass)
					break;
				cls++;
			}
			for (int k = 0; k < cls; k++) {
				totalcost -= Math.log(1 - a[nlayer - 1][k]);
			}
			totalcost -= Math.log(a[nlayer - 1][cls]);
			for (int k = cls + 1; k < nclass; k++) {
				totalcost -= Math.log(1 - a[nlayer - 1][k]);
			}
		}
		totalcost /= m;
		double penalty = 0;
		for (int l = 0; l < nlayer; l++) {
			int nrow = theta[l].length, ncol = theta[l][0].length;
			for (int r = 0; r < nrow; r++) {
				for (int c = 1; c < ncol; c++) {
					penalty += theta[l][r][c] * theta[l][r][c];
				}
			}
		}
		penalty *= lambda/(2*m);
		return totalcost + penalty;
	}

	public NNCostGrad costGradient(double lambda, double[][] ... theta) {
		if(onlyForPrediction)
			throw new RuntimeException("No training data is loaded.");
		if(Xpoly[0].length + 1 != theta[0][0].length)
			throw new IllegalArgumentException("Parameter size is not compatible with train examples length.");
		int nclass = classes.size();
		int nlayer = theta.length;
		if(nlayer == 1)
			throw new IllegalArgumentException("Use Logit instead if there is no hidden layer.");
		if(theta[nlayer - 1].length != nclass)
			throw new IllegalArgumentException("Parameter size is not compatible with the number of classes in layer " + (nlayer + 1));
		for (int l = 1; l < nlayer; l++) {
			if(theta[l][0].length != theta[l - 1].length + 1)
				throw new IllegalArgumentException("Parameter size is not compatible with layer size in layer " + (l + 1));
		}
		int m = (int) Math.floor(0.7 * y.length);
		double[][] a = new double[nlayer][];
		double[][] error = new double[nlayer][];;
		double[][][] gradient = new double[nlayer][][];
		for (int l = 0; l < nlayer; l++) {
			int nrow = theta[l].length, ncol = theta[l][0].length;
			a[l] = new double[nrow];
			error[l] = new double[nrow];
			gradient[l] = new double[nrow][ncol];
		}
		double totalcost = 0;
		double crossprod;
		for (int i = 0; i < m; i++) {
			/******* Feed Forward (calculation of activations and cost) ********/
			int nrow = theta[0].length, ncol = Xpoly[i].length + 1;
			for (int r = 0; r < nrow; r++) {
				crossprod = theta[0][r][0];
				for (int c = 1; c < ncol; c++) {
					crossprod += Xpoly[i][c - 1] * theta[0][r][c];
				}
				a[0][r] = sigmoid(crossprod);
			}
			for (int l = 1; l < nlayer; l++) {
				nrow = theta[l].length; ncol = theta[l][0].length;
				for (int r = 0; r < nrow; r++) {
					crossprod = theta[l][r][0];;
					for (int c = 1; c < ncol; c++) {
						crossprod += a[l - 1][c - 1] * theta[l][r][c];
					}
					a[l][r] = sigmoid(crossprod);
				}
			}
			int cls = 0;
			for (int oneClass : classes) {
				if(y[i] == oneClass)
					break;
				cls++;
			}
			for (int k = 0; k < cls; k++) {
				totalcost -= Math.log(1 - a[nlayer - 1][k]);
			}
			totalcost -= Math.log(a[nlayer - 1][cls]);
			for (int k = cls + 1; k < nclass; k++) {
				totalcost -= Math.log(1 - a[nlayer - 1][k]);
			}
			/******* Back Propogation (calculation of errors and gradient) ********/
			nrow = nclass; ncol = theta[nlayer - 1][0].length;
			for (int r = 0; r < nrow; r++) {
				error[nlayer - 1][r] = a[nlayer - 1][r] - (r == cls ? 1 : 0);
				gradient[nlayer - 1][r][0] += error[nlayer - 1][r];
				for (int c = 1; c < ncol; c++) {
					gradient[nlayer - 1][r][c] += error[nlayer - 1][r] * a[nlayer - 2][c - 1];
				}
			}
			for (int l = nlayer - 2; l > 0; l--) {
				nrow = theta[l].length; ncol = theta[l][0].length;
				int nnextrow = theta[l + 1].length;
				for (int r = 0; r < nrow; r++) {
					error[l][r] = 0;
					for (int nextr = 0; nextr < nnextrow; nextr++) {
						error[l][r] += error[l + 1][nextr] * theta[l + 1][nextr][r + 1];
					}
					error[l][r] *= a[l][r] * (1 - a[l][r]);
					gradient[l][r][0] += error[l][r];
					for (int c = 1; c < ncol; c++) {
						gradient[l][r][c] += error[l][r] * a[l - 1][c - 1];
					}
				}
			}
			nrow = theta[0].length; ncol = Xpoly[i].length + 1;
			int nnextrow = theta[1].length;
			for (int r = 0; r < nrow; r++) {
				error[0][r] = 0;
				for (int nextr = 0; nextr < nnextrow; nextr++) {
					error[0][r] += error[1][nextr] * theta[1][nextr][r + 1];
				}
				error[0][r] *= a[0][r] * (1 - a[0][r]);
				gradient[0][r][0] += error[0][r];
				for (int c = 1; c < ncol; c++) {
					gradient[0][r][c] += error[0][r] * Xpoly[i][c - 1];
				}
			}
		}
		totalcost /= m;
		for (int l = 0; l < nlayer; l++) {
			int nrow = theta[l].length, ncol = theta[l][0].length;
			for (int r = 0; r < nrow; r++) {
				gradient[l][r][0] /= m;
				for (int c = 1; c < ncol; c++) {
					gradient[l][r][c] /= m;
					gradient[l][r][c] += lambda/m * theta[l][r][c];
				}
			}
		}

		double penalty = 0;
		for (int l = 0; l < nlayer; l++) {
			int nrow = theta[l].length, ncol = theta[l][0].length;
			for (int r = 0; r < nrow; r++) {
				for (int c = 1; c < ncol; c++) {
					penalty += theta[l][r][c] * theta[l][r][c];
				}
			}
		}
		penalty *= lambda/(2*m);
		return new NNCostGrad(totalcost + penalty, gradient);
	}

	public NNCostGrad costGradientTest(double lambda, double[][] ... theta) {
		if(onlyForPrediction)
			throw new RuntimeException("No training data is loaded.");
		if(Xpoly[0].length + 1 != theta[0][0].length)
			throw new IllegalArgumentException("Parameter size is not compatible with train examples length.");
		int nclass = classes.size();
		int nlayer = theta.length;
		if(nlayer == 1)
			throw new IllegalArgumentException("Use Logit instead if there is no hidden layer.");
		if(theta[nlayer - 1].length != nclass)
			throw new IllegalArgumentException("Parameter size is not compatible with the number of classes in layer " + (nlayer + 1));
		for (int l = 1; l < nlayer; l++) {
			if(theta[l][0].length != theta[l - 1].length + 1)
				throw new IllegalArgumentException("Parameter size is not compatible with layer size in layer " + (l + 1));
		}
		double[][][] gradient = new double[nlayer][][];
		double eps = 0.0001;
		double[][][] rtheta = new double[nlayer][][];
		double[][][] ltheta = new double[nlayer][][];
		for (int l = 0; l < nlayer; l++) {
			int nrow = theta[l].length, ncol = theta[l][0].length;
			rtheta[l] = new double[nrow][ncol];
			ltheta[l] = new double[nrow][ncol];
			for (int r = 0; r < nrow; r++) {
				for (int c = 1; c < ncol; c++) {
					rtheta[l][r][c] = theta[l][r][c];
					ltheta[l][r][c] = theta[l][r][c];
				}
			}
		}

		for (int l = 0; l < nlayer; l++) {
			int nrow = theta[l].length, ncol = theta[l][0].length;
			gradient[l] = new double[nrow][ncol];
			System.out.println("layer " + l);
			for (int r = 0; r < nrow; r++) {
				System.out.println("node " + r);
				for (int c = 0; c < ncol; c++) {
					System.out.println("parameter " + c);
					rtheta[l][r][c] += eps;
					ltheta[l][r][c] -= eps;
					gradient[l][r][c] = (cost(lambda, rtheta) - cost(lambda, ltheta)) / (2*eps);
					rtheta[l][r][c] = theta[l][r][c];
					ltheta[l][r][c] = theta[l][r][c];
				}
			}
		}
		return new NNCostGrad(cost(lambda, theta), gradient);
	}

	public void epoch(double lambda, double[][] ... theta) {
		if(onlyForPrediction)
			throw new RuntimeException("No training data is loaded.");
		if(Xpoly[0].length + 1 != theta[0][0].length)
			throw new IllegalArgumentException("Parameter size is not compatible with train examples length.");
		int nclass = classes.size();
		int nlayer = theta.length;
		if(nlayer == 1)
			throw new IllegalArgumentException("Use Logit instead if there is no hidden layer.");
		if(theta[nlayer - 1].length != nclass)
			throw new IllegalArgumentException("Parameter size is not compatible with the number of classes in layer " + (nlayer + 1));
		for (int l = 1; l < nlayer; l++) {
			if(theta[l][0].length != theta[l - 1].length + 1)
				throw new IllegalArgumentException("Parameter size is not compatible with layer size in layer " + (l + 1));
		}
		int m = (int) Math.floor(0.7 * y.length);
		double[][] a = new double[nlayer][];
		double[][] error = new double[nlayer][];;
		double[][][] gradient = new double[nlayer][][];
		for (int l = 0; l < nlayer; l++) {
			int nrow = theta[l].length, ncol = theta[l][0].length;
			a[l] = new double[nrow];
			error[l] = new double[nrow];
			gradient[l] = new double[nrow][ncol];
		}
		double crossprod;
		for (int i = 0; i < m; i++) {
			/******* Feed Forward (calculation of activations and cost) ********/
			int nrow = theta[0].length, ncol = Xpoly[i].length + 1;
			for (int r = 0; r < nrow; r++) {
				crossprod = theta[0][r][0];
				for (int c = 1; c < ncol; c++) {
					crossprod += Xpoly[i][c - 1] * theta[0][r][c];
				}
				a[0][r] = sigmoid(crossprod);
			}
			for (int l = 1; l < nlayer; l++) {
				nrow = theta[l].length; ncol = theta[l][0].length;
				for (int r = 0; r < nrow; r++) {
					crossprod = theta[l][r][0];;
					for (int c = 1; c < ncol; c++) {
						crossprod += a[l - 1][c - 1] * theta[l][r][c];
					}
					a[l][r] = sigmoid(crossprod);
				}
			}
			int cls = 0;
			for (int oneClass : classes) {
				if(y[i] == oneClass)
					break;
				cls++;
			}
			/******* Back Propogation (calculation of errors and gradient) ********/
			nrow = nclass; ncol = theta[nlayer - 1][0].length;
			for (int r = 0; r < nrow; r++) {
				error[nlayer - 1][r] = a[nlayer - 1][r] - (r == cls ? 1 : 0);
				gradient[nlayer - 1][r][0] = error[nlayer - 1][r];
				for (int c = 1; c < ncol; c++) {
					gradient[nlayer - 1][r][c] = error[nlayer - 1][r] * a[nlayer - 2][c - 1];
				}
			}
			for (int l = nlayer - 2; l > 0; l--) {
				nrow = theta[l].length; ncol = theta[l][0].length;
				int nnextrow = theta[l + 1].length;
				for (int r = 0; r < nrow; r++) {
					error[l][r] = 0;
					for (int nextr = 0; nextr < nnextrow; nextr++) {
						error[l][r] += error[l + 1][nextr] * theta[l + 1][nextr][r + 1];
					}
					error[l][r] *= a[l][r] * (1 - a[l][r]);
					gradient[l][r][0] = error[l][r];
					for (int c = 1; c < ncol; c++) {
						gradient[l][r][c] = error[l][r] * a[l - 1][c - 1];
					}
				}
			}
			nrow = theta[0].length; ncol = Xpoly[i].length + 1;
			int nnextrow = theta[1].length;
			for (int r = 0; r < nrow; r++) {
				error[0][r] = 0;
				for (int nextr = 0; nextr < nnextrow; nextr++) {
					error[0][r] += error[1][nextr] * theta[1][nextr][r + 1];
				}
				error[0][r] *= a[0][r] * (1 - a[0][r]);
				gradient[0][r][0] = error[0][r];
				for (int c = 1; c < ncol; c++) {
					gradient[0][r][c] = error[0][r] * Xpoly[i][c - 1];
				}
			}
			for (int l = 0; l < nlayer; l++) {
				nrow = theta[l].length; ncol = theta[l][0].length;
				for (int r = 0; r < nrow; r++) {
					//theta[l][r][0] -= ALPHA / (i + 1) * gradient[l][r][0];
					theta[l][r][0] -= ALPHA * gradient[l][r][0];
					for (int c = 1; c < ncol; c++) {
						gradient[l][r][c] += lambda * theta[l][r][c];
						//theta[l][r][c] -= ALPHA / (i + 1) * gradient[l][r][c];
						theta[l][r][c] -= ALPHA * gradient[l][r][c];
					}
				}
			}
		}
	}

	public Double[] onlineLearn(double lambda, int ... hiddenLayerSize) {
		if(onlyForPrediction)
			throw new RuntimeException("No training data is loaded.");
		System.out.println("learning ...");
		int nlayer = hiddenLayerSize.length + 1;
		int nclass = classes.size();
		fittedTheta = new double[nlayer][][];
		fittedTheta[0] = new double[hiddenLayerSize[0]][Xpoly[0].length + 1];
		for (int l = 1; l < nlayer - 1; l++) {
			fittedTheta[l] = new double[hiddenLayerSize[l]][hiddenLayerSize[l - 1] + 1];
		}
		fittedTheta[nlayer - 1] = new double[nclass][hiddenLayerSize[nlayer - 2] + 1];
		for (int l = 0; l < nlayer; l++) {
			int nrow = fittedTheta[l].length, ncol = fittedTheta[l][0].length;
			double bound = Math.sqrt(6) / (Math.sqrt(nrow + ncol - 1));
			for (int r = 0; r < nrow; r++) {
				for (int c = 0; c < ncol; c++) {
					fittedTheta[l][r][c] = Math.random() * 2 * bound - bound;
				}
			}
		}
		int m = (int) Math.floor(0.7 * y.length) + 1;
		int M = y.length;
		double cost = cost(lambda, fittedTheta);
		List<Double> validAccuracys = new LinkedList<Double>();
		double validAccuracy;
		double swap;
		for (int i = 0; i < MAXITER; i++) {
			System.out.println("Cost of epoch\t" + i + "\t= " + cost);
			System.out.println("Evaluating the hypothesis ...");
			validAccuracy = 0;
			for (int j = m; j < M; j++) {
				if(y[j] != predict(Xpoly[j]))
					validAccuracy++;
			}
			validAccuracy = 1 - validAccuracy / (M - m);
			System.out.println("accuracy on valid set is " + validAccuracy);
			validAccuracys.add(validAccuracy);
			System.out.println();
			swap = cost;
			epoch(lambda, fittedTheta);
			cost = cost(lambda, fittedTheta);
			if(Math.abs(cost - swap) < THRESHOLD * (cost + THRESHOLD)) {
				System.out.println("Threshold has reached.");
				break;
			}
		}	
		return validAccuracys.toArray(new Double[0]);
	}

	public Double[] batchLearn(double lambda, int ... hiddenLayerSize) {
		if(onlyForPrediction)
			throw new RuntimeException("No training data is loaded.");
		System.out.println("learning ...");
		int nlayer = hiddenLayerSize.length + 1;
		int nclass = classes.size();
		fittedTheta = new double[nlayer][][];
		fittedTheta[0] = new double[hiddenLayerSize[0]][Xpoly[0].length + 1];
		for (int l = 1; l < nlayer - 1; l++) {
			fittedTheta[l] = new double[hiddenLayerSize[l]][hiddenLayerSize[l - 1] + 1];
		}
		fittedTheta[nlayer - 1] = new double[nclass][hiddenLayerSize[nlayer - 2] + 1];
		for (int l = 0; l < nlayer; l++) {
			int nrow = fittedTheta[l].length, ncol = fittedTheta[l][0].length;
			double bound = Math.sqrt(6) / (Math.sqrt(nrow + ncol - 1));
			for (int r = 0; r < nrow; r++) {
				for (int c = 0; c < ncol; c++) {
					fittedTheta[l][r][c] = Math.random() * 2 * bound - bound;
				}
			}
		}
		int m = (int) Math.floor(0.7 * y.length) + 1;
		int M = y.length;
		NNCostGrad cgr = costGradient(lambda, fittedTheta);
		List<Double> validAccuracys = new LinkedList<Double>();
		double validAccuracy;
		double swap;
		for (int i = 0; i < MAXITER; i++) {
			System.out.println("Cost\t" + i + "\t= " + cgr.cost);
			if(i % 5 == 0) {
				System.out.println("Evaluating the hypothesis ...");
				validAccuracy = 0;
				for (int j = m; j < M; j++) {
					if(y[j] != predict(Xpoly[j]))
						validAccuracy++;
				}
				validAccuracy = 1 - validAccuracy / (M - m);
				System.out.println("Accuracy on validation set : " + validAccuracy);				
				validAccuracys.add(validAccuracy);
			}
			for (int l = 0; l < nlayer; l++) {
				int nrow = fittedTheta[l].length, ncol = fittedTheta[l][0].length;
				for (int r = 0; r < nrow; r++) {
					for (int c = 0; c < ncol; c++) {
						fittedTheta[l][r][c] -= ALPHA * cgr.gradient[l][r][c];
					}
				}
			}
			swap = cgr.cost;
			cgr = costGradient(lambda, fittedTheta);
			if(Math.abs(cgr.cost - swap) < THRESHOLD * (cgr.cost + THRESHOLD)) {
				System.out.println("Threshold has reached.");
				break;
			}
		}	
		return validAccuracys.toArray(new Double[0]);
	}

	public void errorAnalysis(String errorFile) {
		if(fittedTheta == null)
			throw new RuntimeException("The hypothesis is not set.");
		System.out.println("Error analysis started ...");
		int n = XLst.get(0).length;
		int m = (int) Math.floor(0.7 * y.length) + 1;
		int M = y.length;
		double error = 0;
		BufferedWriter err = null;
		try {
			err = new BufferedWriter(new FileWriter(errorFile));
			err.write("Predicted,Actual");
			for(int j = 0; j < n; j++)
				err.write(",pixel" + (j + 1));
			err.newLine();
			for (int i = m; i < M; i++) {
				int predictedClass = predict(Xpoly[i]);
				int actualClass = y[i];
				if(actualClass != predictedClass) {
					error++;
					err.write(predictedClass + ",");
					err.write(actualClass + "");
					for(int j = 0; j < n; j++)
						err.write("," + XLst.get(order[i])[j]);
					err.newLine();
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {err.close();} catch (IOException e) {e.printStackTrace();}
		}
	}

	public void saveEstimates(String estimateFile) {
		if(fittedTheta == null)
			throw new RuntimeException("The hypothesis is not set.");
		System.out.println("Saving estimates started ...");
		int nlayer = fittedTheta.length;
		int n = fittedTheta[0][0].length - 1;
		BufferedWriter est = null;
		try {
			est = new BufferedWriter(new FileWriter(estimateFile));
			Iterator<Integer> oneClassIter = classes.iterator();
			est.write(oneClassIter.next() + "");
			while(oneClassIter.hasNext()) {
				est.write("," + oneClassIter.next());
			}
			est.newLine(); est.newLine();

			est.write(nlayer + "," + n);
			for (int l = 0; l < nlayer; l++) {
				est.write("," + fittedTheta[l].length);
			}
			est.newLine(); est.newLine();

			est.write(mu[0] + "");
			for (int j = 1; j < n; j++) {
				est.write("," + mu[j]);
			}
			est.newLine();
			est.write(sig[0] + "");
			for (int j = 1; j < n; j++) {
				est.write("," + sig[j]);
			}
			est.newLine(); est.newLine();

			for (int l = 0; l < nlayer; l++) {
				int nrow = fittedTheta[l].length, ncol = fittedTheta[l][0].length;
				for (int r = 0; r < nrow; r++) {
					est.write(fittedTheta[l][r][0] + "");
					for (int c = 1; c < ncol; c++) {
						est.write("," + fittedTheta[l][r][c]);
					}
					est.newLine();
				}
				est.newLine();
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {est.close();} catch (IOException e) {e.printStackTrace();}
		}
	}

	public void loadEstimates(String estimateFile) {
		System.out.println("Loading estimates started ...");
		BufferedReader est = null;
		try {
			est = new BufferedReader(new FileReader(estimateFile));
			classes = new TreeSet<Integer>();
			String[] fields = est.readLine().split(",");
			for (int k = 0; k < fields.length; k++) {
				classes.add(Integer.parseInt(fields[k]));
			}
			est.readLine();

			fields = est.readLine().split(",");
			int nlayer = Integer.parseInt(fields[0]);
			if(fields.length != nlayer + 2)
				throw new IllegalArgumentException("File format is not correct in header line.");
			int n = Integer.parseInt(fields[1]);
			fittedTheta = new double[nlayer][][];
			for (int l = 0; l < nlayer; l++) {
				fittedTheta[l] = new double[Integer.parseInt(fields[l + 2])][Integer.parseInt(fields[l + 1]) + 1];
			}
			if(fittedTheta[nlayer - 1].length != classes.size())
				throw new IllegalArgumentException("File format is not correct in header line.");
			est.readLine();

			fields = est.readLine().split(",");
			if(fields.length != n)
				throw new IllegalArgumentException("File format is not correct in mu line.");
			mu = new double[n];
			for (int j = 0; j < n; j++) {
				mu[j] = Double.parseDouble(fields[j]);
			}
			fields = est.readLine().split(",");
			if(fields.length != n)
				throw new IllegalArgumentException("File format is not correct in sig line.");
			sig = new double[n];
			for (int j = 0; j < n; j++) {
				sig[j] = Double.parseDouble(fields[j]);
			}
			est.readLine();

			for (int l = 0; l < nlayer; l++) {
				int nrow = fittedTheta[l].length, ncol = fittedTheta[l][0].length;
				for (int r = 0; r < nrow; r++) {
					fields = est.readLine().split(",");
					if(fields.length != ncol)
						throw new IllegalArgumentException("File format is not correct in " + (l + 1) + "th theta matrix.");
					for (int c = 0; c < ncol; c++) {
						fittedTheta[l][r][c] = Double.parseDouble(fields[c]);
					}
				}
				est.readLine();
			}
		} catch(IOException e) {
			e.printStackTrace();
		} finally {
			try{est.close();} catch (IOException e) {e.printStackTrace();}
		}
	}

	public int predict(double[] x) {
		if(fittedTheta == null)
			throw new RuntimeException("The hypothesis is not set.");
		if(x.length + 1 != fittedTheta[0][0].length)
			throw new IllegalArgumentException("The test example length is not compatible with the hypothesis parameter.");
		int nlayer = fittedTheta.length;
		double[][] a = new double[nlayer][];
		double crossprod;
		a[0] = new double[fittedTheta[0].length];
		int nrow = fittedTheta[0].length, ncol = x.length + 1;
		for (int r = 0; r < nrow; r++) {
			crossprod = fittedTheta[0][r][0];
			for (int c = 1; c < ncol; c++) {
				crossprod += x[c - 1] * fittedTheta[0][r][c];
			}
			a[0][r] = sigmoid(crossprod);
		}
		for (int l = 1; l < nlayer; l++) {
			a[l] = new double[fittedTheta[l].length];
			nrow = fittedTheta[l].length; ncol = fittedTheta[l][0].length;
			for (int r = 0; r < nrow; r++) {
				crossprod = fittedTheta[l][r][0];;
				for (int c = 1; c < ncol; c++) {
					crossprod += a[l - 1][c - 1] * fittedTheta[l][r][c];
				}
				a[l][r] = sigmoid(crossprod);
			}
		}
		int argmax = -1;
		double max = 0;
		int k = 0;
		for (int oneClass : classes) {
			if(a[nlayer - 1][k] > max) {
				max = a[nlayer - 1][k];
				argmax = oneClass;
			}
			k++;
		}
		return argmax;
	}

	public void predict(String testFile, String predictionFile, boolean preprocess) {
		System.out.println("Prediction started ...");
		BufferedReader in = null;
		BufferedWriter out = null;
		try {
			in = new BufferedReader(new FileReader(testFile));
			out = new BufferedWriter(new FileWriter(predictionFile));
			String scan = in.readLine();
			if(scan == null)
				throw new IllegalArgumentException("File is empty.");
			int n = scan.split(",").length;
			int nres = mu.length;
			double[] x = new double[n];
			double[] xres;
			int i = 1;
			out.write("ImageId,Label");
			out.newLine();
			while((scan = in.readLine()) != null) {
				String[] fields = scan.split(",");
				for (int j = 0; j < n; j++) {
					x[j] = Double.parseDouble(fields[j]);
				}
				if(preprocess)
					xres = extractFeature(x);
				else
					xres = x;
				for (int j = 0; j < nres; j++) {
					xres[j] -= mu[j];
					if(sig[j] > 0)
						xres[j] /= sig[j];
				}
				out.write(i + "," + predict(xres));
				out.newLine();
				i++;
			}

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {in.close(); out.close();} catch (IOException e) {e.printStackTrace();}
		}
	}

	public static void main(String[] args) {
		boolean toLearn = true;
		if(toLearn) {
			NeuralNet lr = new NeuralNet("./train_pca.csv");
			lr.partitionData(false);
			lr.extractFeature(false);
			lr.polyFeatureNoRep(1);
			lr.normalizeFeature();

			/*********** Tests cost function and gradient 
			//lr.loadEstimates("./ex4weights.csv");
			//double[][][] theta_initial = lr.fittedTheta;
			//System.out.println(lr.cost(0.0, theta_initial));
			double[][][] theta_initial = new double[2][][];
			theta_initial[0] = new double[2][401];
			theta_initial[1] = new double[10][3];
			for (int l = 0; l < 2; l++) {
				int nrow = theta_initial[l].length, ncol = theta_initial[l][0].length;
				for (int r = 0; r < nrow; r++) {
					for (int c = 0; c < ncol; c++) {
						theta_initial[l][r][c] = Math.random();
					}
				}
			}
			System.out.println("gradient calculation ...");
			double[][][] gradient = lr.costGradient(0.0, theta_initial).gradient;
			System.out.println("numerical gradient calculation ...");
			double[][][] gradientTest = lr.costGradientTest(0.0, theta_initial).gradient;
			//double[][][] gradientTest = lr.costGradientTest(0.0, theta_initial).gradient;
			double diff = 0;
			for (int l = 0; l < 2; l++) {
				int nrow = theta_initial[l].length, ncol = theta_initial[l][0].length;
				for (int r = 0; r < nrow; r++) {
					for (int c = 0; c < ncol; c++) {
						System.out.println(gradient[l][r][c] + "\t" + gradientTest[l][r][c]);
						diff = Math.max(diff, Math.abs(gradient[l][r][c] - gradientTest[l][r][c]));
					}
				}
			}
			System.out.println("diff = " + diff);
			System.out.println(lr.cost(0.0, theta_initial));
			*/
			/*********** Tests learning algorithm
			double[] theta = lr.learn(0.0, 0);
			for (int j = 0; j < theta.length; j++) {
				System.out.println(theta[j]);
			}
			double[] gr = lr.gradient(theta, 0.0, 0);
			double cost = lr.cost(theta, 0.0, 0);
			double norm = 0;
			for (int j = 0; j < theta.length; j++) {
				norm = Math.max(norm, Math.abs(gr[j]));
			}
			System.out.println("cost = " + cost + ", |gradeint| = " + norm);
			*/
			/******* Validation curve *******/
			/*double accuracy[] = new double[12];
			double lambda = 0.01;
			for (int i = 0; i < accuracy.length; i++) {
				System.out.println("lambda = " + lambda);
				System.out.println("accuracy = " + (accuracy[i] = lr.learn("./errors.csv", lambda, 100)));
				lambda *= 2;
			}
			lambda = 0.01;
			for (int i = 0; i < accuracy.length; i++) {
				System.out.println("lambda = " + lambda + ", accuracy = " + accuracy[i]);
				lambda *= 2;
			}
			*/
			double lambda = 0.0;
			//Double[] accuracy = lr.batchLearn(lambda, 500, 500);
			Double[] accuracy = lr.onlineLearn(lambda, 1000, 100);
			int i = 0;
			for(double acc : accuracy)
				System.out.println("epoch " + (i++) + "accuracy = " + acc);
			lr.errorAnalysis("./errors.csv");
			lr.saveEstimates("./nn_estimates.csv");
			
		} else {
			NeuralNet lr = new NeuralNet();
			lr.loadEstimates("./nn_estimates.csv");
			lr.predict("./test_pca.csv", "./submission.csv", false);	
		}

	}
}
