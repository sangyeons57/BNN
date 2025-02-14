using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using UnityEngine;

public class NeuralNetwork
{
	public double LearningRate { get; set; } = 0.1;
	public class DenseLayer
	{
		private Matrix<double> Weights;
		private Vector<double> Biases;
		private Matrix<double> GradientWeights;
		private Vector<double> GradientBiases;
		
		
		private string Activation;
		public Matrix<double> Inputs;
		public Matrix<double> Outputs;

		public int inputSize { get; private set; }
		public int outputSize { get; private set; }

		private int miniBatchCount = 0;

		public DenseLayer(int inputSize, int outputSize, string activation = null)
		{
			Weights = Matrix<double>.Build.Dense(inputSize, outputSize);
			Biases = Vector<double>.Build.Dense(outputSize, 0);
			GradientWeights = Matrix<double>.Build.Dense(inputSize, outputSize, 0);
			GradientBiases = Vector<double>.Build.Dense(outputSize, 0);
			
			this.inputSize = inputSize;
			this.outputSize = outputSize;

			this.Activation = activation;
		}
		
		public void InitWeights (Func<DenseLayer, double> format)
		{
			for (int i = 0; i < Weights.RowCount; i++)
			{
				for (int j = 0; j < Weights.ColumnCount; j++)
				{
					Weights[i, j] = format(this);
				}
			}
		}
		
		public Matrix<double> Forward(Matrix<double> inputs)
		{
			this.Inputs = inputs;
			//Debug.Log(inputs);
			Matrix<double> z = inputs * Weights;
			for (int i = 0; i < z.RowCount; i++)
			{
				z.SetRow(i, z.Row(i).Add(Biases));
			}
			Outputs = Activate(z);
			return Outputs;
		}

		public Matrix<double> Backward(Matrix<double> error, double learningRate)
		{
			int m = Inputs.RowCount;
			error = error.PointwiseMultiply(ActivationDerivative(Outputs));

			Matrix<double> dWeights = Inputs.Transpose() * error / m;
			Vector<double> dBiases = error.ColumnSums() / m;
			Matrix<double> dInput = error * Weights.Transpose();
			
			Weights -= learningRate * dWeights;
			Biases -= learningRate * dBiases;
			//miniBatchCount++;

			return dInput;
		}

		public void ApplyMiniBatch(double learningRate)
		{
			Weights -= learningRate * GradientWeights / miniBatchCount;
			Biases -= learningRate * GradientBiases / miniBatchCount;
			
			GradientWeights.Clear();
			GradientBiases.Clear();
			miniBatchCount = 0;
		}
		private Matrix<double> Activate(Matrix<double> z)
		{
			switch (Activation)
			{
				case "relu":
					return z.PointwiseMaximum(0);
				case "sigmoid":
					return 1 / (1 + (-z).PointwiseExp());
				case "tanh":
					return z.PointwiseTanh();
				default:
					return z;
			}
		}

		private Matrix<double> ActivationDerivative(Matrix<double> z)
		{
			switch (Activation)
			{
				case "relu":
					return z.Map(v => v > 0.0 ? 1.0 : 0.0);
				case "sigmoid":
					Matrix<double> sigmoid = 1 / (1 + (-z).PointwiseExp());
					return sigmoid.PointwiseMultiply(1 - sigmoid);
				case "tanh":
					return 1 - z.PointwiseTanh().PointwisePower(2);
				default:
					return Matrix<double>.Build.Dense(z.RowCount, z.ColumnCount, 1);
			}
		}
	}
	
	private readonly List<DenseLayer> Layers = new List<DenseLayer>();

	public void AddLayer(int inputSize, int outputSize, string activation = null)
	{
		Layers.Add(new DenseLayer(inputSize, outputSize, activation));
	}

	public void InitWeights(Func<DenseLayer, double> format)
	{
		foreach (DenseLayer layer in Layers)
		{
			layer.InitWeights(format);
		}
	}

	public Matrix<double> Forward(Matrix<double> x)
	{
		foreach (DenseLayer layer in Layers)
		{
			x = layer.Forward(x);
		}

		return x;
	}

	public void Backword(Matrix<double> error)
	{
		for (int i = Layers.Count - 1; i >= 0; i--)
		{
			error = Layers[i].Backward(error, LearningRate);
		}
	}

	public void Train(Matrix<double> x, Matrix<double> y, int epochs = 1000)
	{
		for (int epoch = 0; epoch < epochs; epoch++)
		{
			Matrix<double> outputs = SoftMax(Forward(x));
			double loss = ComputeLoss(y, outputs);
			Backword(outputs - y);

			if (epoch % 100 == 0)
			{
				Debug.Log($"Epoch {epoch} / {epochs}, Loss: {loss:F4}");
			}
		}
	}

	public Vector<double> Predict(Matrix<double> x)
	{
		Matrix<double> outputs = SoftMax(Forward(x));
		
		for (int i = 0; i < outputs.RowCount; i++)
		{
			Vector<double> row = outputs.Row(i);
			
			string str = ""+ x.Row(i)[0];
			str += " [" + string.Join(", ", row.Enumerate().Select(x => x)) + "]";
			Debug.Log(str);
		}
		return RowMaximumIndex(outputs);
	}

	public static Matrix<double> SoftMax(Matrix<double> x)
	{
		Matrix<double> result = Matrix<double>.Build.Dense(x.RowCount, x.ColumnCount);
		for (int i = 0; i < result.RowCount; i++)
		{
			double maxVal = x.Row(i).Max(); 
			Vector<double> expX = (x.Row(i) - maxVal).PointwiseExp();
			
			for (int j = 0; j < result.ColumnCount; j++)
			{
				result[i, j] = expX[j] / expX.Sum();
			}
		}

		return result;
	}

	private static Vector<double> GetRowmax(Matrix<double> matrix)
	{
		Vector<double> rowmax = Vector<double>.Build.Dense(matrix.RowCount);
		for (int r = 0; r < matrix.RowCount; r++)
		{
			rowmax[r] = double.MinValue;
			for (int c = 0; c < matrix.RowCount; c++)
			{
				rowmax[r] = Math.Max(matrix[r, c], rowmax[r]);
			}
		}

		return rowmax;
	}

	private static double ComputeLoss(Matrix<double> yTrue, Matrix<double> yPred)
	{
		double epsilon = 1e-8;
		return -yTrue.PointwiseMultiply((yPred + epsilon).PointwiseLog()).RowSums().Sum() / yTrue.RowCount;
	}
	
	public static Vector<double> RowMaximumIndex(Matrix<double> matrix)
	{
		int rowCount = matrix.RowCount; 
		Vector<double> result = Vector<double>.Build.Dense(rowCount);

		for (int i = 0; i < rowCount; i++)
		{
			Vector<double> row = matrix.Row(i);
			result[i] = row.MaximumIndex();
		}

		return result;
	}
}
