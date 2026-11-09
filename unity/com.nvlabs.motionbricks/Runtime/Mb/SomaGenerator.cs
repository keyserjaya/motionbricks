using System;
using System.Diagnostics;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Generates SOMA motion from context and target poses, ported from
    /// <c>motionbricks/model/soma/engine.py</c> and its Warp kernels. The steps are the same as the
    /// Python engine: encode the guide poses into the model's input tensors, run the model, decode
    /// the prediction into local joint transforms, and correct the endpoints.
    /// </summary>
    public sealed class SomaGenerator
    {
        private readonly ISomaInferenceBackend _backend;
        private readonly SomaSkeleton _skeleton;
        private readonly object _lock = new object();

        private readonly Pose[] _guideWorld;
        private readonly Pose[] _localScratch;
        private readonly Pose[] _worldScratch;
        private readonly Pose[] _startDeltas;
        private readonly Pose[] _endDeltas;

        private Pose _alignment;
        private readonly SomaGenerationInputs _inputs = new SomaGenerationInputs();
        private readonly SomaGenerationOutputs _outputs = new SomaGenerationOutputs();

        /// <summary>Token count the model chose in the last generation.</summary>
        public int LastTokenCount { get; private set; }

        /// <summary>Number of valid frames in the last generation.</summary>
        public int LastFrameCount { get; private set; }

        /// <summary>Milliseconds the last generation spent in each stage.</summary>
        public float LastEncodeMilliseconds { get; private set; }

        public float LastInferenceMilliseconds { get; private set; }

        public float LastDecodeMilliseconds { get; private set; }

        /// <summary>The alignment transform used by the last generation, for diagnostics.</summary>
        public Pose LastAlignment => _alignment;

        /// <summary>The packed input tensors of the last generation, for diagnostics and parity tests.</summary>
        public SomaGenerationInputs LastInputs => _inputs;

        public SomaGenerator(ISomaInferenceBackend backend, SomaSkeleton skeleton)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _skeleton = skeleton ?? throw new ArgumentNullException(nameof(skeleton));
            _guideWorld = new Pose[MbContract.ContextAndTargetPoses * skeleton.NumJoints];
            _localScratch = new Pose[skeleton.NumJoints];
            _worldScratch = new Pose[skeleton.NumJoints];
            _startDeltas = new Pose[skeleton.NumJoints];
            _endDeltas = new Pose[skeleton.NumJoints];
        }

        /// <summary>
        /// Generates motion toward the target from the context poses and writes it into
        /// <paramref name="output"/>.
        /// </summary>
        public void Generate(
            SomaPoseToken context,
            SomaPoseToken target,
            SomaAnimationBuffer output,
            MotionBricksInferenceConfig config = null)
        {
            // The generator keeps scratch buffers, so only one generation runs at a time. This also
            // lets a background generation run while the main thread keeps playing the previous one.
            lock (_lock)
            {
                GenerateLocked(context, target, output, config);
            }
        }

        private void GenerateLocked(
            SomaPoseToken context,
            SomaPoseToken target,
            SomaAnimationBuffer output,
            MotionBricksInferenceConfig config)
        {
            config ??= new MotionBricksInferenceConfig();
            if (context.NumPoses != MbContract.PosesPerToken)
            {
                throw new ArgumentException(
                    $"Context token must contain {MbContract.PosesPerToken} poses.", nameof(context));
            }

            if (target.NumPoses < 1 || target.NumPoses > MbContract.PosesPerToken)
            {
                throw new ArgumentException(
                    $"Target token must contain 1 to {MbContract.PosesPerToken} poses.", nameof(target));
            }

            var stopwatch = Stopwatch.StartNew();
            EncodeGuidePoses(context, target, config);
            LastEncodeMilliseconds = (float)stopwatch.Elapsed.TotalMilliseconds;

            stopwatch.Restart();
            _backend.Run(_inputs, _outputs);
            LastInferenceMilliseconds = (float)stopwatch.Elapsed.TotalMilliseconds;

            stopwatch.Restart();
            DecodeGeneratedPoses(output);
            if (config.CorrectGeneratedDiff)
            {
                CorrectEndpoints(context, target, output);
            }

            LastDecodeMilliseconds = (float)stopwatch.Elapsed.TotalMilliseconds;
        }

        /// <summary>
        /// Encodes the context and target poses into the model's input tensors without running the
        /// model. Used by the parity tests.
        /// </summary>
        public void EncodeForTest(SomaPoseToken context, SomaPoseToken target, MotionBricksInferenceConfig config)
        {
            EncodeGuidePoses(context, target, config ?? new MotionBricksInferenceConfig());
        }

        /// <summary>
        /// Encodes the context and target poses into the model's input tensors. Ported from
        /// <c>_encode_guide_poses</c> and <c>encode_motionbricks_inputs_kernel</c>.
        /// </summary>
        private void EncodeGuidePoses(SomaPoseToken context, SomaPoseToken target, MotionBricksInferenceConfig config)
        {
            var numJoints = _skeleton.NumJoints;
            var identity = new Pose(Vector3.zero, Quaternion.identity);

            // The guide poses are composed into world transforms, which for Unity are the model space
            // transforms themselves (model_to_runtime is the identity).
            for (var pose = 0; pose < MbContract.PosesPerToken; pose++)
            {
                for (var joint = 0; joint < numJoints; joint++)
                {
                    _localScratch[joint] = context.Poses[pose * numJoints + joint];
                }

                MbPose.ComposeWorldPoses(_localScratch, _skeleton.ParentIndices, _worldScratch, identity);
                Array.Copy(_worldScratch, 0, _guideWorld, pose * numJoints, numJoints);
            }

            for (var pose = 0; pose < MbContract.PosesPerToken; pose++)
            {
                for (var joint = 0; joint < numJoints; joint++)
                {
                    _localScratch[joint] = target.Poses[pose * numJoints + joint];
                }

                MbPose.ComposeWorldPoses(_localScratch, _skeleton.ParentIndices, _worldScratch, identity);
                Array.Copy(_worldScratch, 0, _guideWorld, (MbContract.PosesPerToken + pose) * numJoints, numJoints);
            }

            ComputeGuideAlignment(config);

            var globalRootValues = _inputs.GlobalRootValues;
            var localRootValues = _inputs.LocalRootValues;
            var localPoses = _inputs.LocalPoses;
            Array.Clear(globalRootValues, 0, globalRootValues.Length);
            Array.Clear(localRootValues, 0, localRootValues.Length);
            Array.Clear(localPoses, 0, localPoses.Length);

            var finalFrame = MbContract.ContextAndTargetPoses - 1;
            for (var frame = 0; frame < MbContract.ContextAndTargetPoses; frame++)
            {
                // Root velocities use the next pose, so they are zero at the last context pose and the
                // last target pose.
                var nextFrame = frame == MbContract.PosesPerToken - 1
                    ? frame
                    : Math.Min(frame + 1, finalFrame);

                var currentHips = MbPose.AlignedPosition(_alignment, _guideWorld[frame * numJoints].position);
                var nextHips = MbPose.AlignedPosition(_alignment, _guideWorld[nextFrame * numJoints].position);

                var currentRight = MbPose.AlignedPosition(
                    _alignment, _guideWorld[frame * numJoints + _skeleton.RightLegIndex].position);
                var currentLeft = MbPose.AlignedPosition(
                    _alignment, _guideWorld[frame * numJoints + _skeleton.LeftLegIndex].position);
                var nextRight = MbPose.AlignedPosition(
                    _alignment, _guideWorld[nextFrame * numJoints + _skeleton.RightLegIndex].position);
                var nextLeft = MbPose.AlignedPosition(
                    _alignment, _guideWorld[nextFrame * numJoints + _skeleton.LeftLegIndex].position);

                var currentHeading = MbPose.HeadingFromHipPositions(currentRight, currentLeft);
                var nextHeading = MbPose.HeadingFromHipPositions(nextRight, nextLeft);
                var headingDelta = MbPose.WrapAngle(nextHeading - currentHeading);

                var rootOffset = frame * 5;
                globalRootValues[rootOffset] = currentHips.x;
                globalRootValues[rootOffset + 1] = currentHips.y;
                globalRootValues[rootOffset + 2] = currentHips.z;
                globalRootValues[rootOffset + 3] = Mathf.Cos(currentHeading);
                globalRootValues[rootOffset + 4] = Mathf.Sin(currentHeading);

                var localRootOffset = frame * 4;
                localRootValues[localRootOffset] = headingDelta * MbContract.SampleRate;
                localRootValues[localRootOffset + 1] = (nextHips.x - currentHips.x) * MbContract.SampleRate;
                localRootValues[localRootOffset + 2] = (nextHips.z - currentHips.z) * MbContract.SampleRate;
                localRootValues[localRootOffset + 3] = currentHips.y;

                var rootPosition = new Vector3(currentHips.x, 0f, currentHips.z);
                var frameRowOffset = frame * SomaModelContract.LocalPosesRowSize;
                for (var modelJoint = 0; modelJoint < SomaModelContract.NumModelJoints; modelJoint++)
                {
                    var poseJoint = _skeleton.ModelToPose[modelJoint];
                    var jointPosition = MbPose.AlignedPosition(_alignment, _guideWorld[frame * numJoints + poseJoint].position);
                    var jointRotation = MbPose.Conjugate(_alignment.rotation) * _guideWorld[frame * numJoints + poseJoint].rotation;

                    if (modelJoint > 0)
                    {
                        var positionOffset = frameRowOffset + (modelJoint - 1) * 3;
                        var centered = jointPosition - rootPosition;
                        localPoses[positionOffset] = centered.x;
                        localPoses[positionOffset + 1] = centered.y;
                        localPoses[positionOffset + 2] = centered.z;
                    }

                    var rotationOffset = frameRowOffset +
                                         (SomaModelContract.NumModelJoints - 1) * 3 + modelJoint * 6;
                    MbPose.Write6D(localPoses, rotationOffset, 0, jointRotation);
                }
            }

            WriteConditions(target.NumPoses, config);
        }

        /// <summary>
        /// Computes the heading alignment of the last context pose to the desired heading. Ported
        /// from <c>compute_guide_alignment_kernel</c>. The result maps aligned poses back to SOMA model
        /// space and keeps the root height.
        /// </summary>
        private void ComputeGuideAlignment(MotionBricksInferenceConfig config)
        {
            var numJoints = _skeleton.NumJoints;
            var lastContextFrame = MbContract.PosesPerToken - 1;
            var hipsPosition = _guideWorld[lastContextFrame * numJoints].position;
            var currentAngle = MbPose.HeadingFromHipPositions(
                _guideWorld[lastContextFrame * numJoints + _skeleton.RightLegIndex].position,
                _guideWorld[lastContextFrame * numJoints + _skeleton.LeftLegIndex].position);
            var desiredAngle = Mathf.Atan2(config.HeadingAlignment.x, config.HeadingAlignment.y);
            _alignment = new Pose(
                new Vector3(hipsPosition.x, 0f, hipsPosition.z),
                Quaternion.AngleAxis((currentAngle - desiredAngle) * Mathf.Rad2Deg, Vector3.up));
        }

        /// <summary>
        /// Tells the model which context and target poses are given, and which token counts it may
        /// choose. Ported from <c>write_motionbricks_conditions_kernel</c>.
        /// </summary>
        private void WriteConditions(int numTargetPoses, MotionBricksInferenceConfig config)
        {
            var bits = config.AllowedTokenBits();
            for (var index = 0; index < MbContract.ContextAndTargetPoses; index++)
            {
                var value = index < MbContract.PosesPerToken + numTargetPoses ? 1 : 0;
                _inputs.HasGlobalRootValues[index] = value;
                _inputs.HasLocalPoses[index] = value;
                // Local root values stay unmarked at the last context pose and the last target pose,
                // because their derivatives are unknown there.
                _inputs.HasLocalRootValues[index] =
                    index == MbContract.PosesPerToken - 1 ||
                    index == MbContract.PosesPerToken + numTargetPoses - 1
                        ? 0
                        : value;
            }

            for (var index = 0; index < SomaModelContract.AllowedPredNumTokensSize; index++)
            {
                _inputs.AllowedPredNumTokens[index] = (bits >> index) & 1;
            }
        }

        /// <summary>
        /// Decodes the predicted motion into local joint transforms and foot contacts. Ported from
        /// <c>decode_motionbricks_output_kernel</c>. The boundary frames are dropped.
        /// </summary>
        private void DecodeGeneratedPoses(SomaAnimationBuffer output)
        {
            var numJoints = _skeleton.NumJoints;
            var predictions = _outputs.PredGlobalMotions;
            var tokenCount = _outputs.PredNumTokens[0];
            LastTokenCount = tokenCount;
            if (tokenCount < MbContract.MinTokenCount || tokenCount > MbContract.MaxTokenCount)
            {
                throw new InvalidOperationException(
                    $"The model selected {tokenCount} tokens, outside the allowed set " +
                    $"{MbContract.MinTokenCount} to {MbContract.MaxTokenCount}.");
            }

            var rawCount = Mathf.Clamp(tokenCount * MbContract.PosesPerToken, 0, SomaModelContract.PredRawFrames);
            var frameCount = Math.Max(0, rawCount - 2 * MbContract.BoundaryFrames);
            LastFrameCount = frameCount;
            output.NumFrames = frameCount;

            for (var frame = 0; frame < frameCount; frame++)
            {
                var rawFrame = MbContract.BoundaryFrames + frame;
                var rowOffset = rawFrame * SomaModelContract.PredRowSize;

                for (var contact = 0; contact < MbContract.FootContactCount; contact++)
                {
                    output.FootContacts[frame * MbContract.FootContactCount + contact] = Mathf.Clamp01(
                        predictions[rowOffset + SomaModelContract.PredFootContactOffset + contact]);
                }

                var hipsPosition = new Vector3(
                    predictions[rowOffset],
                    predictions[rowOffset + 1],
                    predictions[rowOffset + 2]);
                var hipsRotation = MbPose.Read6D(predictions, rowOffset, SomaModelContract.DecodeRotationOffset);

                // This also restores the ground offset removed when encoding the context and target poses.
                var hipsTransform = MbPose.Mul(_alignment, new Pose(hipsPosition, hipsRotation));
                output.Poses[frame * numJoints] = hipsTransform;

                for (var joint = 1; joint < numJoints; joint++)
                {
                    var modelJoint = _skeleton.PoseToModel[joint];
                    if (modelJoint < 0)
                    {
                        output.Poses[frame * numJoints + joint] = _skeleton.ReferenceLocal[joint];
                        continue;
                    }

                    var predictedRotation = MbPose.Read6D(
                        predictions, rowOffset, SomaModelContract.DecodeRotationOffset + modelJoint * 6);
                    var parentModelJoint = _skeleton.PoseToModel[_skeleton.JointParent(joint)];
                    var parentRotation = MbPose.Read6D(
                        predictions, rowOffset, SomaModelContract.DecodeRotationOffset + parentModelJoint * 6);

                    // The common alignment cancels in parent-to-child space.
                    var localRotation = MbPose.Mul(MbPose.Conjugate(parentRotation), predictedRotation);
                    output.Poses[frame * numJoints + joint] = new Pose(
                        _skeleton.ReferenceLocal[joint].position,
                        MbPose.Normalize(localRotation));
                }
            }
        }

        /// <summary>
        /// Matches the generated motion's endpoints to the last context pose and the first target pose.
        /// Ported from <c>compute_endpoint_deltas_kernel</c> and <c>apply_endpoint_deltas_kernel</c>.
        /// </summary>
        private void CorrectEndpoints(SomaPoseToken context, SomaPoseToken target, SomaAnimationBuffer output)
        {
            var frameCount = output.NumFrames;
            if (frameCount < 2)
            {
                return;
            }

            var numJoints = _skeleton.NumJoints;
            var lastFrame = frameCount - 1;
            var lastContextPose = MbContract.PosesPerToken - 1;

            _startDeltas[0] = MbPose.RootEndpointDelta(
                output.Poses[0],
                context.Poses[lastContextPose * numJoints]);
            _endDeltas[0] = MbPose.RootEndpointDelta(
                output.Poses[lastFrame * numJoints],
                target.Poses[0]);

            for (var joint = 1; joint < numJoints; joint++)
            {
                _startDeltas[joint] = MbPose.Delta(
                    output.Poses[joint],
                    context.Poses[lastContextPose * numJoints + joint]);
                _endDeltas[joint] = MbPose.Delta(
                    output.Poses[lastFrame * numJoints + joint],
                    target.Poses[joint]);
            }

            for (var frame = 0; frame < frameCount; frame++)
            {
                var endWeight = frame / (float)(frameCount - 1);
                var startWeight = 1f - endWeight;

                var root = output.Poses[frame * numJoints];
                output.Poses[frame * numJoints] = MbPose.ApplyRootEndpointCorrection(
                    root, _startDeltas[0], _endDeltas[0], endWeight);

                for (var joint = 1; joint < numJoints; joint++)
                {
                    var pose = MbPose.ApplyDelta(
                        output.Poses[frame * numJoints + joint], _startDeltas[joint], startWeight);
                    output.Poses[frame * numJoints + joint] = MbPose.ApplyDelta(
                        pose, _endDeltas[joint], endWeight);
                }
            }
        }
    }
}
