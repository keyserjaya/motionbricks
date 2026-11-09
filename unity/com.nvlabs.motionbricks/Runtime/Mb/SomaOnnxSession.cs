using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Runs the SOMA ONNX model with the ONNX Runtime. The bundled ONNX files work with any
    /// inference library (see <c>docs/onnx_inference.md</c> of the MotionBricks repository), so the
    /// native library is the only dependency here.
    /// </summary>
    public sealed class SomaOnnxSession : ISomaInferenceBackend
    {
        private readonly InferenceSession _session;
        private readonly object _lock = new object();
        private readonly string[] _outputNames = { SomaModelContract.PredGlobalMotionsTensor, SomaModelContract.PredNumTokensTensor };

        /// <summary>Name of the active execution provider, for diagnostics.</summary>
        public string ExecutionProvider { get; }

        /// <summary>Path of the loaded model file.</summary>
        public string ModelPath { get; }

        /// <summary>Loads a model file with the DirectML execution provider, falling back to the CPU.</summary>
        public SomaOnnxSession(string modelPath, bool preferDirectML = true)
        {
            if (string.IsNullOrEmpty(modelPath))
            {
                throw new ArgumentException("modelPath cannot be empty.", nameof(modelPath));
            }

            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"MotionBricks model file not found: {modelPath}", modelPath);
            }

            ModelPath = modelPath;
            if (preferDirectML)
            {
                try
                {
                    var options = CreateOptions();
                    options.AppendExecutionProvider_DML(0);
                    _session = new InferenceSession(modelPath, options);
                    ExecutionProvider = "DirectML";
                    return;
                }
                catch (Exception exception)
                {
                    Debug.LogWarning(
                        $"[MotionBricks] DirectML is unavailable ({exception.Message}). Falling back to the CPU provider. " +
                        "CPU inference is far slower; a DirectML-capable GPU is recommended.");
                }
            }

            _session = new InferenceSession(modelPath, CreateOptions());
            ExecutionProvider = "CPU";
        }

        private static SessionOptions CreateOptions()
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            };
            return options;
        }

        /// <summary>Runs one generation with the given flat tensors.</summary>
        public void Run(SomaGenerationInputs inputs, SomaGenerationOutputs outputs)
        {
            lock (_lock)
            {
                var globalRootValues = new DenseTensor<float>(
                    inputs.GlobalRootValues, new[] { 1, MbContract.ContextAndTargetPoses, 5 });
                var localRootValues = new DenseTensor<float>(
                    inputs.LocalRootValues, new[] { 1, MbContract.ContextAndTargetPoses, 4 });
                var localPoses = new DenseTensor<float>(
                    inputs.LocalPoses, new[] { 1, MbContract.ContextAndTargetPoses, 267 });
                var hasGlobalRootValues = new DenseTensor<int>(
                    inputs.HasGlobalRootValues, new[] { 1, MbContract.ContextAndTargetPoses });
                var hasLocalRootValues = new DenseTensor<int>(
                    inputs.HasLocalRootValues, new[] { 1, MbContract.ContextAndTargetPoses });
                var hasLocalPoses = new DenseTensor<int>(
                    inputs.HasLocalPoses, new[] { 1, MbContract.ContextAndTargetPoses });
                var allowedTokens = new DenseTensor<int>(
                    inputs.AllowedPredNumTokens,
                    new[] { 1, SomaModelContract.AllowedPredNumTokensSize });

                var namedInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(SomaModelContract.GlobalRootValuesTensor, globalRootValues),
                    NamedOnnxValue.CreateFromTensor(SomaModelContract.LocalRootValuesTensor, localRootValues),
                    NamedOnnxValue.CreateFromTensor(SomaModelContract.LocalPosesTensor, localPoses),
                    NamedOnnxValue.CreateFromTensor(SomaModelContract.HasGlobalRootValuesTensor, hasGlobalRootValues),
                    NamedOnnxValue.CreateFromTensor(SomaModelContract.HasLocalRootValuesTensor, hasLocalRootValues),
                    NamedOnnxValue.CreateFromTensor(SomaModelContract.HasLocalPosesTensor, hasLocalPoses),
                    NamedOnnxValue.CreateFromTensor(SomaModelContract.AllowedPredNumTokensTensor, allowedTokens),
                };

                using var results = _session.Run(namedInputs, _outputNames);
                foreach (var result in results)
                {
                    if (result.Name == SomaModelContract.PredGlobalMotionsTensor)
                    {
                        CopyTo(result.AsTensor<float>(), outputs.PredGlobalMotions);
                    }
                    else if (result.Name == SomaModelContract.PredNumTokensTensor)
                    {
                        CopyTo(result.AsTensor<int>(), outputs.PredNumTokens);
                    }
                }
            }
        }

        private static void CopyTo(Tensor<float> tensor, float[] destination)
        {
            if (tensor is DenseTensor<float> dense)
            {
                dense.Buffer.Span.CopyTo(destination.AsSpan(0, Math.Min(dense.Buffer.Length, destination.Length)));
                return;
            }

            var index = 0;
            foreach (var value in tensor)
            {
                destination[index++] = value;
            }
        }

        private static void CopyTo(Tensor<int> tensor, int[] destination)
        {
            if (tensor is DenseTensor<int> dense)
            {
                dense.Buffer.Span.CopyTo(destination.AsSpan(0, Math.Min(dense.Buffer.Length, destination.Length)));
                return;
            }

            var index = 0;
            foreach (var value in tensor)
            {
                destination[index++] = value;
            }
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
