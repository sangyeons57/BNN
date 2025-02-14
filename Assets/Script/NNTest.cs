using MathNet.Numerics.LinearAlgebra;
using UnityEngine;
using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Complex32;
using Random = System.Random;

public class NNTest : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Random rand = new Random();
        NeuralNetwork nn = new();
        nn.AddLayer(1, 5, "relu");
        nn.AddLayer(5, 5, "relu");
        nn.AddLayer(5, 3);

        nn.InitWeights((layer) => {
            return rand.NextDouble() * Math.Sqrt(2.0 / layer.inputSize);
        });
        
        var (xTrain, yTrain) = GenerateData(1000);
        Matrix<double> yTrainOneHot = OneHotEncode(yTrain, 3);
        Debug.Log(xTrain);
        nn.LearningRate = 0.1;
        nn.Train(xTrain, yTrainOneHot);
        
        var (xTest, yTest) = GenerateData(10);
        Debug.Log("\nTest Data:");
        Debug.Log(xTest * 1000);

        var predictions = nn.Predict(xTest);
        
        String printResult = $"\n" +
                             $"True Labels: {string.Join(" ", yTest.Enumerate().Select(val => val))}" + "\n" +
                             $"Predictions: {string.Join(" ", predictions.Enumerate().Select(val => val))}";
        Debug.Log(printResult);
    }

    private Matrix<double> OneHotEncode(Vector<double> labels, int numClasses)
    {
        Matrix<double> encoded = Matrix<double>.Build.Dense(labels.Count, numClasses, 0); 
        for (int i = 0; i < labels.Count; i++)
        {
            encoded[i, (int)labels[i]] = 1;
        }

        return encoded;
    } 
    private (Matrix<double>, Vector<double>) GenerateData(int numSamples)
    {
        Random rand = new Random();
        Vector<double> xData = Vector<double>.Build.Dense(numSamples);
        Vector<double> yData = Vector<double>.Build.Dense(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            int choice = rand.Next(1, 4);
            int num = choice switch
            {
                1 => rand.Next(1, 10),
                2 => rand.Next(10, 100),
                3 => rand.Next(100, 1000),
                _ => 0
            };

            xData[i] = num / 1000.0;
            yData[i] = choice - 1;
        }

        return (xData.ToColumnMatrix(), yData);
    }
}
