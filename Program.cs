using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace machineLearningTest
{
    public class Program
    {

        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _DataPath => Path.Combine(_appPath, "..", "..", "...", "Data");
        private static string _ModelPath => Path.Combine(_appPath, "..", "..", "...", "Models");
        private static string _trainData => Path.Combine(_DataPath, "issues_train.tsv");
        private static string _testData => Path.Combine(_DataPath, "issues_test.tsv");
        private static string _modelPath => Path.Combine(_ModelPath, "model.zip");


        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predictionEngine;
        private static ITransformer _trainedModel;
        private static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            Console.WriteLine(_trainData);
            _mlContext = new MLContext(seed: 0);
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainData, hasHeader: true);

            IEstimator<ITransformer> pipeLine = ProcessData();
            IEstimator<ITransformer> trainingPipeline = 
                BuildAndTrainModel(_trainingDataView, pipeLine);
            
            Evaluate(_trainingDataView.Schema);
            
            PredictIssue();
        }
        
        private static void PredictIssue()
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
            
            GitHubIssue singleIssue = new GitHubIssue()
            {
                Title = "Entity Framework crashes", 
                Description = "When connecting to the database, EF is crashing" 
            };

            _predictionEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

            var prediction = _predictionEngine.Predict(singleIssue);
            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }

        private static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = _mlContext
                .Data
                .LoadFromTextFile<GitHubIssue>(_testData, hasHeader: true);

            var testMetrics = _mlContext.MulticlassClassification
                .Evaluate(_trainedModel.Transform(testDataView));
            
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
        }
        
        private static void SaveModelAsFile(MLContext mlContext,DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            _mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }

        private static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView,
            IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainingDataView);

            _predictionEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

            GitHubIssue issue = new GitHubIssue() {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = _predictionEngine.Predict(issue);
            
            Console.WriteLine($"Single Prediction just-trained-model - Result: {prediction.Area}");

            return trainingPipeline;
        }

        private static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms
                .Conversion
                .MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text
                    .FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text
                    .FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms
                    .Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);
            return pipeline;
        }
    }
}