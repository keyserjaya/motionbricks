using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Runs the SOMA ONNX model with the Unity Inference Engine. The engine converts the ONNX file to
    /// a model asset in the Editor and executes it with compute shaders, so the same code runs in the
    /// Editor and in a built player without native plugins.
    /// </summary>
    public sealed class SomaInferenceEngineBackend : ISomaInferenceBackend
    {
        private Worker _worker;
        private readonly Model _model;
        private BackendType _backendType;
        private bool _fallbackChecked;
        private readonly Tensor<float> _globalRootValues;
        private readonly Tensor<float> _localRootValues;
        private readonly Tensor<float> _localPoses;
        private readonly Tensor<int> _hasGlobalRootValues;
        private readonly Tensor<int> _hasLocalRootValues;
        private readonly Tensor<int> _hasLocalPoses;
        private readonly Tensor<int> _allowedPredNumTokens;
        private readonly Tensor<float> _predGlobalMotions;
        private readonly Tensor<int> _predNumTokens;
        private readonly bool _hasLogitsOutput;
        private readonly float[] _tokenLogitsScratch = new float[12];
        private readonly object _lock = new object();

        /// <summary>
        /// The name of the token-count logits output. The Unity-simplified model
        /// (<c>tools/unity_golden/simplify_token_output.py</c>) exposes the masked 12-wide logits that
        /// the token count is computed from, so the argmax can run in C# where it is reliable.
        /// </summary>
        public const string PredNumTokenLogitsTensor = "pred_num_token_logits";

        /// <summary>Name of the active backend, for diagnostics.</summary>
        public string ExecutionProvider { get; private set; }

        /// <summary>The model asset that was loaded.</summary>
        public ModelAsset ModelAsset { get; }

        /// <summary>Loads a model asset. The GPU provider is probed first and falls back to the CPU
        /// when it produces invalid logits (some InferenceEngine versions mis-execute this model's
        /// integer shape subgraphs on the GPU).</summary>
        public SomaInferenceEngineBackend(ModelAsset modelAsset, BackendType backendType = BackendType.GPUCompute)
        {
            ModelAsset = modelAsset ?? throw new ArgumentNullException(nameof(modelAsset));
            _model = ModelLoader.Load(modelAsset);
            _backendType = backendType;
            _worker = new Worker(_model, backendType);
            ExecutionProvider = backendType.ToString();

            _hasLogitsOutput = _model.outputs.Exists(output => output.name == PredNumTokenLogitsTensor);

            _globalRootValues = new Tensor<float>(new TensorShape(1, MbContract.ContextAndTargetPoses, 5));
            _localRootValues = new Tensor<float>(new TensorShape(1, MbContract.ContextAndTargetPoses, 4));
            _localPoses = new Tensor<float>(new TensorShape(1, MbContract.ContextAndTargetPoses, 267));
            _hasGlobalRootValues = new Tensor<int>(new TensorShape(1, MbContract.ContextAndTargetPoses));
            _hasLocalRootValues = new Tensor<int>(new TensorShape(1, MbContract.ContextAndTargetPoses));
            _hasLocalPoses = new Tensor<int>(new TensorShape(1, MbContract.ContextAndTargetPoses));
            _allowedPredNumTokens = new Tensor<int>(new TensorShape(1, SomaModelContract.AllowedPredNumTokensSize));
            _predGlobalMotions = new Tensor<float>(
                new TensorShape(1, SomaModelContract.PredRawFrames, SomaModelContract.PredRowSize));
            _predNumTokens = new Tensor<int>(new TensorShape(1));
        }

        /// <summary>Runs one generation with the given flat tensors.</summary>
        public void Run(SomaGenerationInputs inputs, SomaGenerationOutputs outputs)
        {
            lock (_lock)
            {
                _globalRootValues.Upload(inputs.GlobalRootValues);
                _localRootValues.Upload(inputs.LocalRootValues);
                _localPoses.Upload(inputs.LocalPoses);
                _hasGlobalRootValues.Upload(inputs.HasGlobalRootValues);
                _hasLocalRootValues.Upload(inputs.HasLocalRootValues);
                _hasLocalPoses.Upload(inputs.HasLocalPoses);
                _allowedPredNumTokens.Upload(inputs.AllowedPredNumTokens);

                _worker.SetInput(SomaModelContract.GlobalRootValuesTensor, _globalRootValues);
                _worker.SetInput(SomaModelContract.LocalRootValuesTensor, _localRootValues);
                _worker.SetInput(SomaModelContract.LocalPosesTensor, _localPoses);
                _worker.SetInput(SomaModelContract.HasGlobalRootValuesTensor, _hasGlobalRootValues);
                _worker.SetInput(SomaModelContract.HasLocalRootValuesTensor, _hasLocalRootValues);
                _worker.SetInput(SomaModelContract.HasLocalPosesTensor, _hasLocalPoses);
                _worker.SetInput(SomaModelContract.AllowedPredNumTokensTensor, _allowedPredNumTokens);

                _worker.Schedule();

                CopyTo(_worker.PeekOutput(SomaModelContract.PredGlobalMotionsTensor), outputs.PredGlobalMotions);
                var logitsRead = _hasLogitsOutput &&
                                 CopyTo(_worker.PeekOutput(PredNumTokenLogitsTensor), _tokenLogitsScratch);
                if (logitsRead)
                {
                    // The token count is a masked argmax over the logits, computed here because the
                    // ONNX graph's own count path uses integer shape ops that some backends
                    // mis-evaluate. The rule is validated against the reference engine by
                    // tools/unity_golden/simplify_token_output.py.
                    if (!_fallbackChecked)
                    {
                        _fallbackChecked = true;
                        if (HasInvalidLogits(_tokenLogitsScratch) && _backendType != BackendType.CPU)
                        {
                            // The GPU backend mis-executed the model's shape subgraphs. Re-run on the
                            // CPU backend, which produces correct results at ~80 ms per generation.
                            Debug.LogWarning(
                                "[MotionBricks] The GPU backend produced invalid logits for this model. " +
                                "Falling back to the CPU backend.");
                            _worker.Dispose();
                            _worker = new Worker(_model, BackendType.CPU);
                            _backendType = BackendType.CPU;
                            ExecutionProvider = BackendType.CPU.ToString();
                            Run(inputs, outputs);
                            return;
                        }
                    }

                    outputs.PredNumTokens[0] = ArgMaxTokenCount(_tokenLogitsScratch, inputs.AllowedPredNumTokens);
                }
                else
                {
                    CopyTo(_worker.PeekOutput(SomaModelContract.PredNumTokensTensor), outputs.PredNumTokens);
                }
            }
        }

        /// <summary>
        /// Returns the predicted token count from the 12-wide masked logits: the argmax over the 11
        /// allowed token counts (6 to 16), with the twelfth column ignored. Column values whose
        /// allowed flag is zero are skipped even if the backend left them unmasked.
        /// </summary>
        public static int ArgMaxTokenCount(float[] logits, int[] allowedPredNumTokens)
        {
            var bestIndex = 0;
            var bestScore = float.NegativeInfinity;
            for (var column = 0; column < SomaModelContract.AllowedPredNumTokensSize; column++)
            {
                if (allowedPredNumTokens[column] == 0)
                {
                    continue;
                }

                var score = logits[column];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestIndex = column;
                }
            }

            return MbContract.MinTokenCount + bestIndex;
        }

        private static bool HasInvalidLogits(float[] logits)
        {
            foreach (var value in logits)
            {
                if (float.IsNaN(value))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool CopyTo(Tensor tensor, float[] destination)
        {
            var readable = tensor.ReadbackAndClone();
            try
            {
                if (readable is Tensor<float> typed)
                {
                    Array.Copy(typed.DownloadToArray(), destination, Math.Min(typed.shape.length, destination.Length));
                    return true;
                }

                return false;
            }
            finally
            {
                readable.Dispose();
            }
        }

        private static void CopyTo(Tensor tensor, int[] destination)
        {
            var readable = tensor.ReadbackAndClone();
            try
            {
                if (readable is Tensor<int> typed)
                {
                    Array.Copy(typed.DownloadToArray(), destination, Math.Min(typed.shape.length, destination.Length));
                }
            }
            finally
            {
                readable.Dispose();
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                _globalRootValues?.Dispose();
                _localRootValues?.Dispose();
                _localPoses?.Dispose();
                _hasGlobalRootValues?.Dispose();
                _hasLocalRootValues?.Dispose();
                _hasLocalPoses?.Dispose();
                _allowedPredNumTokens?.Dispose();
                _predGlobalMotions?.Dispose();
                _predNumTokens?.Dispose();
                _worker?.Dispose();
            }
        }
    }
}
