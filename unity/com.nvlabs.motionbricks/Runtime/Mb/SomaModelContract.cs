using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// The SOMA model contract (see <c>motionbricks/model/soma/contract.py</c> and the manifest
    /// <c>soma_n1.json</c>): the model's joint names, heading joints, and tensor interface for a
    /// batch of one.
    /// </summary>
    public static class SomaModelContract
    {
        public const string ModelName = "mmm_v6_soma";

        /// <summary>The 30 joints the model predicts, in model order. Hips is the root.</summary>
        public static readonly string[] JointNames =
        {
            "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye",
            "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd",
            "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
            "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot",
            "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
        };

        public const int NumModelJoints = 30;

        /// <summary>Leg joints whose positions give a pose's heading, as in training.</summary>
        public const string RightHeadingJoint = "RightLeg";
        public const string LeftHeadingJoint = "LeftLeg";

        /// <summary>SOMA joints of the four foot contact values, in contact order.</summary>
        public static readonly string[] FootContactJoints =
        {
            "LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase",
        };

        // Tensor names and batch-one shapes.
        public const string GlobalRootValuesTensor = "global_root_values";
        public const string LocalRootValuesTensor = "local_root_values";
        public const string LocalPosesTensor = "local_poses";
        public const string HasGlobalRootValuesTensor = "has_global_root_values";
        public const string HasLocalRootValuesTensor = "has_local_root_values";
        public const string HasLocalPosesTensor = "has_local_poses";
        public const string AllowedPredNumTokensTensor = "allowed_pred_num_tokens";
        public const string PredGlobalMotionsTensor = "pred_global_motions";
        public const string PredNumTokensTensor = "pred_num_tokens";

        public const int GlobalRootValuesSize = MbContract.ContextAndTargetPoses * 5; // [8, 5]
        public const int LocalRootValuesSize = MbContract.ContextAndTargetPoses * 4;  // [8, 4]
        public const int LocalPosesSize = MbContract.ContextAndTargetPoses * 267;     // [8, 267]
        public const int LocalPosesRowSize = 267; // one frame: 29 positions of 3 and 30 rotations of 6
        public const int MaskSize = MbContract.ContextAndTargetPoses;                 // [8]
        public const int AllowedPredNumTokensSize = MbContract.MaxTokenCount - MbContract.MinTokenCount + 1; // [11]
        public const int PredRawFrames = MbContract.MaxTokenCount * MbContract.PosesPerToken; // 64
        public const int PredRowSize = 366; // global root 5, positions 29*3, rotations 30*6, velocities 30*3, contacts 4
        public const int PredFootContactOffset = PredRowSize - MbContract.FootContactCount; // 362

        // Rows of the SOMA decode layout.
        public const int DecodePositionOffset = 5;
        public const int DecodeRotationOffset = DecodePositionOffset + (NumModelJoints - 1) * 3; // 92
    }

    /// <summary>Flat input tensors of one SOMA generation with a batch of one.</summary>
    public sealed class SomaGenerationInputs
    {
        public readonly float[] GlobalRootValues = new float[SomaModelContract.GlobalRootValuesSize];
        public readonly float[] LocalRootValues = new float[SomaModelContract.LocalRootValuesSize];
        public readonly float[] LocalPoses = new float[SomaModelContract.LocalPosesSize];
        public readonly int[] HasGlobalRootValues = new int[SomaModelContract.MaskSize];
        public readonly int[] HasLocalRootValues = new int[SomaModelContract.MaskSize];
        public readonly int[] HasLocalPoses = new int[SomaModelContract.MaskSize];
        public readonly int[] AllowedPredNumTokens = new int[SomaModelContract.AllowedPredNumTokensSize];
    }

    /// <summary>Flat output tensors of one SOMA generation with a batch of one.</summary>
    public sealed class SomaGenerationOutputs
    {
        public readonly float[] PredGlobalMotions = new float[SomaModelContract.PredRawFrames * SomaModelContract.PredRowSize];
        public readonly int[] PredNumTokens = new int[1];
    }

    /// <summary>Backend that runs one SOMA generation. Implemented by the ONNX Runtime session.</summary>
    public interface ISomaInferenceBackend : System.IDisposable
    {
        /// <summary>Name of the active execution provider, for diagnostics.</summary>
        string ExecutionProvider { get; }

        /// <summary>Runs one generation with the given flat tensors.</summary>
        void Run(SomaGenerationInputs inputs, SomaGenerationOutputs outputs);
    }

    /// <summary>Generation settings ported from <c>MotionBricksInferenceConfig</c>.</summary>
    public sealed class MotionBricksInferenceConfig
    {
        public const int DefaultTokenCount = MbContract.DefaultMinTokenCount; // historical alias

        /// <summary>Token counts the model may choose, each from 6 to 16. Default allows 8 to 14.</summary>
        public System.Collections.Generic.List<int> AllowedTokenCounts = DefaultCounts();

        /// <summary>
        /// Spreads small corrections over the motion so it starts at the last context pose and ends at
        /// the first target pose.
        /// </summary>
        public bool CorrectGeneratedDiff = true;

        /// <summary>
        /// The ground direction inference turns the last context pose to face, in model space, as (x, z).
        /// (1, 0) faces +X.
        /// </summary>
        public Vector2 HeadingAlignment = new Vector2(1f, 0f);

        public static System.Collections.Generic.List<int> DefaultCounts()
        {
            var counts = new System.Collections.Generic.List<int>(MbContract.DefaultMaxTokenCount - MbContract.DefaultMinTokenCount + 1);
            for (var count = MbContract.DefaultMinTokenCount; count <= MbContract.DefaultMaxTokenCount; count++)
            {
                counts.Add(count);
            }

            return counts;
        }

        public MotionBricksInferenceConfig Clone()
        {
            return new MotionBricksInferenceConfig
            {
                AllowedTokenCounts = new System.Collections.Generic.List<int>(AllowedTokenCounts),
                CorrectGeneratedDiff = CorrectGeneratedDiff,
                HeadingAlignment = HeadingAlignment,
            };
        }

        internal int AllowedTokenBits() => MbContract.AllowedTokenBits(AllowedTokenCounts);
    }

    /// <summary>Per-target generation settings ported from <c>TargetPoseInferenceParams</c>.</summary>
    public struct TargetPoseInferenceParams
    {
        /// <summary>How many frames before the end of the generated motion the transition point sits.</summary>
        public int TransitionEarlyByFrames;

        /// <summary>How many of the target's four poses inform the model, from 1 to 4.</summary>
        public int NumPosesToUse;

        /// <summary>
        /// Hands and feet that IK pins to the target pose during playback. The corrector eases the
        /// pinned limbs onto the target's first pose from half of the segment to its transition point.
        /// </summary>
        public MbTwoBoneIk.IkPin IkPinMask;

        /// <summary>
        /// Token counts the model may choose for this target. Null keeps the counts of the generation config.
        /// </summary>
        public int[] AllowedTokenCounts;

        public TargetPoseInferenceParams(int numPosesToUse, int transitionEarlyByFrames, int[] allowedTokenCounts = null)
        {
            NumPosesToUse = numPosesToUse;
            TransitionEarlyByFrames = transitionEarlyByFrames;
            IkPinMask = MbTwoBoneIk.IkPin.None;
            AllowedTokenCounts = allowedTokenCounts;
        }
    }
}
