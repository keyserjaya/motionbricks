using System;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Stores a fixed number of recent poses and their frame durations, ported from
    /// <c>motionbricks/runtime/motion_recorder.py</c>. The poses are local joint transforms, oldest
    /// first. The latest pose's duration is always zero.
    /// </summary>
    public sealed class SomaMotionRecorder
    {
        /// <summary>Recorded poses, laid out as [slot, joint].</summary>
        public Pose[] Poses { get; }

        /// <summary>Time from each recorded pose to the next, in seconds. The last value is zero.</summary>
        public float[] DeltaTimes { get; }

        public int Capacity { get; }

        public int NumJoints { get; }

        public SomaMotionRecorder(SomaSkeleton skeleton, int capacity = 32)
        {
            if (capacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "capacity must be positive.");
            }

            Capacity = capacity;
            NumJoints = skeleton.NumJoints;
            Poses = new Pose[capacity * NumJoints];
            DeltaTimes = new float[capacity];
            Fill(skeleton.ReferenceLocal);
        }

        /// <summary>Fills every slot with one pose and zero durations, as if the character held it.</summary>
        public void Fill(Pose[] pose)
        {
            var numJoints = NumJoints;
            for (var slot = 0; slot < Capacity; slot++)
            {
                Array.Copy(pose, 0, Poses, slot * numJoints, numJoints);
            }

            Array.Clear(DeltaTimes, 0, DeltaTimes.Length);
        }

        /// <summary>Stores a pose sampled <paramref name="deltaTime"/> seconds after the previous pose.</summary>
        public void Record(float deltaTime, Pose[] pose)
        {
            if (float.IsNaN(deltaTime) || float.IsInfinity(deltaTime) || deltaTime <= 0f)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(deltaTime), deltaTime, "deltaTime must be finite and positive.");
            }

            var numJoints = NumJoints;
            DeltaTimes[Capacity - 1] = deltaTime;
            Array.Copy(DeltaTimes, 1, DeltaTimes, 0, Capacity - 1);
            DeltaTimes[Capacity - 1] = 0f;

            for (var slot = 0; slot < Capacity - 1; slot++)
            {
                Array.Copy(Poses, (slot + 1) * numJoints, Poses, slot * numJoints, numJoints);
            }

            Array.Copy(pose, 0, Poses, (Capacity - 1) * numJoints, numJoints);
        }

        /// <summary>Copies the recorded poses and durations.</summary>
        public SomaMotionRecorderSnapshot Snapshot()
        {
            return new SomaMotionRecorderSnapshot(Poses, DeltaTimes, NumJoints);
        }

        /// <summary>
        /// Finds samples at a fixed rate, ending at the latest recorded pose. Ported from
        /// <c>history_sample_indices</c>. Returns the lower frame, upper frame, and blend of each
        /// sample. Samples before the available history clamp to its oldest pose.
        /// </summary>
        public static void HistorySampleIndices(
            float[] deltaTimes,
            int numSamples,
            float sampleRate,
            out int[] lowerFrames,
            out int[] upperFrames,
            out float[] blends)
        {
            var capacity = deltaTimes.Length;
            lowerFrames = new int[numSamples];
            upperFrames = new int[numSamples];
            blends = new float[numSamples];

            // Pose times run backwards from the latest pose at zero.
            var poseTimes = new float[capacity];
            for (var frame = capacity - 2; frame >= 0; frame--)
            {
                poseTimes[frame] = poseTimes[frame + 1] - deltaTimes[frame];
            }

            var firstSample = -(numSamples - 1) / sampleRate;
            for (var sample = 0; sample < numSamples; sample++)
            {
                var sampleTime = firstSample + sample / sampleRate;

                // The pose times ascend, so a binary search finds the first time after the sample.
                var upper = capacity - 1;
                var low = 0;
                var high = capacity - 1;
                while (low <= high)
                {
                    var middle = (low + high) / 2;
                    if (poseTimes[middle] > sampleTime)
                    {
                        upper = middle;
                        high = middle - 1;
                    }
                    else
                    {
                        low = middle + 1;
                    }
                }

                if (upper > capacity - 1)
                {
                    upper = capacity - 1;
                }

                var lower = Math.Max(upper - 1, 0);
                var frameTime = poseTimes[upper] - poseTimes[lower];
                float blend;
                if (frameTime > 0f)
                {
                    blend = MbPose.Clamp01((float)((sampleTime - poseTimes[lower]) / frameTime));
                }
                else
                {
                    lower = upper;
                    blend = 0f;
                }

                lowerFrames[sample] = lower;
                upperFrames[sample] = upper;
                blends[sample] = blend;
            }
        }
    }

    /// <summary>An immutable copy of a recorder's poses and durations, for background generation.</summary>
    public sealed class SomaMotionRecorderSnapshot
    {
        public Pose[] Poses { get; }

        public float[] DeltaTimes { get; }

        public int NumJoints { get; }

        public SomaMotionRecorderSnapshot(Pose[] poses, float[] deltaTimes, int numJoints)
        {
            Poses = (Pose[])poses.Clone();
            DeltaTimes = (float[])deltaTimes.Clone();
            NumJoints = numJoints;
        }

        /// <summary>Samples four poses at 30 Hz from the snapshot, ending at the latest recorded pose.</summary>
        public void SampleInto(Pose[] destination, int numSamples, float sampleRate)
        {
            SomaMotionRecorder.HistorySampleIndices(
                DeltaTimes, numSamples, sampleRate, out var lowerFrames, out var upperFrames, out var blends);

            for (var sample = 0; sample < numSamples; sample++)
            {
                var lower = lowerFrames[sample] * NumJoints;
                var upper = upperFrames[sample] * NumJoints;
                var blend = blends[sample];
                for (var joint = 0; joint < NumJoints; joint++)
                {
                    destination[sample * NumJoints + joint] = MbPose.Blend(
                        Poses[lower + joint], Poses[upper + joint], blend);
                }
            }
        }
    }
}
