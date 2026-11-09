using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Pose math ported from <c>motionbricks/utils/pose_utils.py</c> and the SOMA kernels in
    /// <c>motionbricks/model/soma/kernels.py</c>. Poses are local joint transforms stored as
    /// <see cref="Pose"/> (position + rotation). Unity's Y-up world plays the role of the SOMA
    /// model space (Y up, XZ ground), so no additional space conversion is applied.
    /// </summary>
    public static class MbPose
    {
        /// <summary>Composes two transforms: apply <paramref name="b"/> first, then <paramref name="a"/>.</summary>
        public static Pose Mul(Pose a, Pose b)
        {
            return new Pose(a.position + a.rotation * b.position, a.rotation * b.rotation);
        }

        /// <summary>Composes two rotations: apply <paramref name="b"/> first, then <paramref name="a"/>.</summary>
        public static Quaternion Mul(Quaternion a, Quaternion b)
        {
            return a * b;
        }

        /// <summary>Slerps between two rotations, keeping the shortest path.</summary>
        public static Quaternion SlerpSafe(Quaternion a, Quaternion b, float t)
        {
            if (t <= 0f)
            {
                return a;
            }

            if (t >= 1f)
            {
                return b;
            }

            return Normalize(Quaternion.Slerp(Normalize(a), Normalize(b), t));
        }

        /// <summary>Returns the inverse transform.</summary>
        public static Pose Inverse(Pose a)
        {
            var inverseRotation = Normalize(Conjugate(a.rotation));
            return new Pose(inverseRotation * -a.position, inverseRotation);
        }

        /// <summary>Interpolates positions linearly and rotations spherically between two transforms.</summary>
        public static Pose Blend(Pose a, Pose b, float blend)
        {
            return new Pose(
                Vector3.LerpUnclamped(a.position, b.position, blend),
                Quaternion.Slerp(a.rotation, b.rotation, blend));
        }

        /// <summary>Returns the weighted delta that moves <paramref name="src"/> onto <paramref name="dst"/>.</summary>
        public static Pose Delta(Pose src, Pose dst)
        {
            return Mul(Inverse(src), dst);
        }

        /// <summary>Applies a weighted transform delta: <c>pose * Blend(identity, delta, weight)</c>.</summary>
        public static Pose ApplyDelta(Pose pose, Pose delta, float weight)
        {
            return Mul(pose, Blend(new Pose(Vector3.zero, Quaternion.identity), delta, Clamp01(weight)));
        }

        /// <summary>
        /// Returns the endpoint correction that moves a generated root onto a context or target root.
        /// The position delta is a world space offset; the rotation delta applies after the generated rotation.
        /// </summary>
        public static Pose RootEndpointDelta(Pose generatedRoot, Pose guideRoot)
        {
            var rotation = Mul(Conjugate(Normalize(generatedRoot.rotation)), Normalize(guideRoot.rotation));
            return new Pose(guideRoot.position - generatedRoot.position, Normalize(rotation));
        }

        /// <summary>
        /// Applies the start and end root corrections at one frame of the generated motion.
        /// <paramref name="endWeight"/> runs from 0 at the first frame to 1 at the last frame.
        /// Positions blend linearly so the root does not swing along an arc.
        /// </summary>
        public static Pose ApplyRootEndpointCorrection(Pose root, Pose startDelta, Pose endDelta, float endWeight)
        {
            var startWeight = 1f - endWeight;
            var rotation = Mul(
                Mul(root.rotation, SlerpSafe(Quaternion.identity, startDelta.rotation, startWeight)),
                SlerpSafe(Quaternion.identity, endDelta.rotation, endWeight));
            return new Pose(
                root.position + startDelta.position * startWeight + endDelta.position * endWeight,
                Normalize(rotation));
        }

        /// <summary>
        /// The heading of a pose, in radians, from the positions of the right and left leg joints.
        /// Mirrors the training representation's <c>compute_heading_method="hips_pos"</c>.
        /// </summary>
        public static float HeadingFromHipPositions(Vector3 rightHip, Vector3 leftHip)
        {
            var across = rightHip - leftHip;
            across.y = 0f;
            var length = across.magnitude;
            if (length <= 1e-6f)
            {
                return 0f;
            }

            across /= length;
            var forward = Vector3.Cross(Vector3.up, across);
            return Mathf.Atan2(forward.x, forward.z);
        }

        /// <summary>Wraps an angle difference to the range [-pi, pi] via atan2(sin, cos).</summary>
        public static float WrapAngle(float angle)
        {
            return Mathf.Atan2(Mathf.Sin(angle), Mathf.Cos(angle));
        }

        /// <summary>
        /// Builds a rotation from orthonormalized axis columns, ported from the 6D-rotation kernels
        /// (<c>_rotation_from_6d</c>): x and y in, with z derived by cross products.
        /// </summary>
        public static Quaternion RotationFrom6D(Vector3 xRaw, Vector3 yRaw)
        {
            var xAxis = xRaw.normalized;
            var zAxis = Vector3.Cross(xAxis, yRaw).normalized;
            var yAxis = Vector3.Cross(zAxis, xAxis);
            return FromColumns(xAxis, yAxis, zAxis);
        }

        /// <summary>Builds a rotation whose axes are the given (orthonormal) columns.</summary>
        public static Quaternion FromColumns(Vector3 xAxis, Vector3 yAxis, Vector3 zAxis)
        {
            var m = Matrix4x4.identity;
            m.SetColumn(0, xAxis);
            m.SetColumn(1, yAxis);
            m.SetColumn(2, zAxis);
            return m.rotation;
        }

        /// <summary>
        /// Reads the 6D rotation at <paramref name="offset"/> of one row of the flat feature buffer.
        /// The layout is x axis XYZ then y axis XYZ.
        /// </summary>
        public static Quaternion Read6D(float[] features, int rowOffset, int offset)
        {
            var baseIndex = rowOffset + offset;
            float x0 = features[baseIndex], x1 = features[baseIndex + 1], x2 = features[baseIndex + 2];
            float y0 = features[baseIndex + 3], y1 = features[baseIndex + 4], y2 = features[baseIndex + 5];
            return RotationFrom6D(new Vector3(x0, x1, x2), new Vector3(y0, y1, y2));
        }

        /// <summary>Writes the 6D representation (x axis, y axis) of a rotation into a flat buffer.</summary>
        public static void Write6D(float[] features, int rowOffset, int offset, Quaternion rotation)
        {
            var baseIndex = rowOffset + offset;
            var xAxis = rotation * Vector3.right;
            var yAxis = rotation * Vector3.up;
            features[baseIndex] = xAxis.x;
            features[baseIndex + 1] = xAxis.y;
            features[baseIndex + 2] = xAxis.z;
            features[baseIndex + 3] = yAxis.x;
            features[baseIndex + 4] = yAxis.y;
            features[baseIndex + 5] = yAxis.z;
        }

        /// <summary>Retarget-safe aligned position: subtract, rotate by the inverse, exactly as the Warp kernel.</summary>
        public static Vector3 AlignedPosition(Pose alignment, Vector3 position)
        {
            // Subtract before rotating to retain precision far from the origin.
            return Conjugate(alignment.rotation) * (position - alignment.position);
        }

        /// <summary>Returns <paramref name="q"/> with unit length (identity when degenerate).</summary>
        public static Quaternion Normalize(Quaternion q)
        {
            var lengthSquared =
                q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
            if (lengthSquared <= 1e-12f || float.IsNaN(lengthSquared))
            {
                return Quaternion.identity;
            }

            var factor = 1f / Mathf.Sqrt(lengthSquared);
            return new Quaternion(q.x * factor, q.y * factor, q.z * factor, q.w * factor);
        }

        /// <summary>Conjugate of a quaternion, which equals the inverse for unit quaternions.</summary>
        public static Quaternion Conjugate(Quaternion q)
        {
            return new Quaternion(-q.x, -q.y, -q.z, q.w);
        }

        public static float Clamp01(float value)
        {
            if (value < 0f) return 0f;
            if (value > 1f) return 1f;
            return value;
        }

        /// <summary>
        /// Composes local joint transforms through the hierarchy. Each parent index refers to an
        /// earlier joint, so one forward pass builds every world transform.
        /// </summary>
        public static void ComposeWorldPoses(Pose[] localPoses, int[] parentIndices, Pose[] worldPoses, Pose rootTx)
        {
            worldPoses[0] = Mul(rootTx, localPoses[0]);
            for (var joint = 1; joint < parentIndices.Length; joint++)
            {
                worldPoses[joint] = Mul(worldPoses[parentIndices[joint]], localPoses[joint]);
            }
        }

        /// <summary>
        /// Returns the frame at a time, the next frame, and the blend toward the next frame. Both
        /// frames are clamped so the last frame is its own next frame. Ported from
        /// <c>frame_index_from_time</c>.
        /// </summary>
        public static void FrameIndexFromTime(
            double time, float sampleRate, int numFrames, out int frame, out int nextFrame, out float blend)
        {
            var lastFrame = Math.Max(0, numFrames - 1);
            var fractional = time * sampleRate;
            frame = Mathf.Clamp((int)Math.Floor(fractional), 0, lastFrame);
            blend = 0f;
            if (frame < lastFrame)
            {
                blend = Clamp01((float)(fractional - frame));
            }

            nextFrame = Math.Min(frame + 1, lastFrame);
        }
    }
}
