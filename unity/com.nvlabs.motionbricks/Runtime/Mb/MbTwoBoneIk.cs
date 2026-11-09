using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Solves a two-bone IK chain so its tip reaches a pinned transform. A direct port of
    /// <c>solve_two_bone_ik</c> from <c>motionbricks/animation/ik.py</c>, which is derived from Daniel
    /// Holden's ik_two_bone implementation (MIT license, see the package NOTICE).
    /// </summary>
    public static class MbTwoBoneIk
    {
        private const float Epsilon = 1e-3f;

        /// <summary>
        /// The IK pin a chain serves. A chain config holds the joint indices of one hand or foot chain.
        /// </summary>
        [System.Flags]
        public enum IkPin
        {
            None = 0,
            LeftHand = 1 << 0,
            RightHand = 1 << 1,
            LeftFoot = 1 << 2,
            RightFoot = 1 << 3,
            All = LeftHand | RightHand | LeftFoot | RightFoot,
        }

        /// <summary>
        /// Solves one two-bone chain. The inputs are the world transforms of the chain start's parent,
        /// the chain start (A), the middle joint (B), the tip (C), and the pinned tip (T).
        /// <paramref name="useHint"/> bends the chain toward <paramref name="hintWorld"/>.
        /// Returns the solved world transforms of A, B, and C. <paramref name="weight"/> blends from
        /// the input pose at 0 to the solved pose at 1.
        /// </summary>
        public static void SolveTwoBoneIk(
            float weight,
            Pose aParentWorld,
            Pose aWorld,
            Pose bWorld,
            Pose cWorld,
            Pose tWorld,
            bool useHint,
            Vector3 hintWorld,
            out Pose solvedA,
            out Pose solvedB,
            out Pose solvedC)
        {
            weight = Mathf.Clamp01(weight);
            var a = aWorld.position;
            var b = bWorld.position;
            var c = cWorld.position;
            var t = Vector3.LerpUnclamped(c, tWorld.position, weight);

            var lengthAb = Vector3.Distance(a, b);
            var lengthCb = Vector3.Distance(b, c);
            var lengthAt = Mathf.Clamp(Vector3.Distance(a, t), Epsilon, lengthAb + lengthCb - Epsilon);

            // The rotation axis of the elbow bend: from the (C - A) vector toward the hint, or toward B.
            var bendToward = useHint ? hintWorld - a : b - a;
            var bendAxis = Vector3.Cross(c - a, bendToward).normalized;

            var angleAcAb0 = SignedAngle(bendAxis, c - a, b - a);
            var angleBaBc0 = SignedAngle(bendAxis, a - b, c - b);
            var angleAcAb1 = Mathf.Acos(Mathf.Clamp(
                (lengthAb * lengthAb + lengthAt * lengthAt - lengthCb * lengthCb) / (2f * lengthAb * lengthAt),
                -1f, 1f));
            var angleBaBc1 = Mathf.Acos(Mathf.Clamp(
                (lengthAb * lengthAb + lengthCb * lengthCb - lengthAt * lengthAt) / (2f * lengthAb * lengthCb),
                -1f, 1f));

            // The chain start rotates about its own local axis (the world bend axis expressed in the
            // bone's local frame), by the angle the solver adds.
            var rotationA = Quaternion.AngleAxis(
                (angleAcAb1 - angleAcAb0) * Mathf.Rad2Deg, MbPose.Conjugate(aWorld.rotation) * bendAxis);
            var rotationB = Quaternion.AngleAxis(
                (angleBaBc1 - angleBaBc0) * Mathf.Rad2Deg, MbPose.Conjugate(bWorld.rotation) * bendAxis);

            var aLocal = MbPose.Mul(MbPose.Mul(MbPose.Inverse(aParentWorld), aWorld),
                new Pose(Vector3.zero, rotationA));
            var aWorldSolved = MbPose.Mul(aParentWorld, aLocal);

            // Turn A so its C direction points at the pinned tip.
            var acLocal = MbPose.Conjugate(aWorldSolved.rotation) * (c - a);
            var atLocal = MbPose.Conjugate(aWorldSolved.rotation) * (t - a);
            var turn = Quaternion.FromToRotation(acLocal.normalized, atLocal.normalized);
            aLocal = MbPose.Mul(aLocal, new Pose(Vector3.zero, turn));
            aWorldSolved = MbPose.Mul(aParentWorld, aLocal);

            // B's local offset is measured against the ORIGINAL chain start (the parameter, not the
            // turned A), so the elbow rotates about the shoulder with it.
            var bLocal = MbPose.Mul(MbPose.Mul(MbPose.Inverse(aWorld), bWorld),
                new Pose(Vector3.zero, rotationB));
            var bWorldSolved = MbPose.Mul(aWorldSolved, bLocal);

            // C keeps its original local offset from B, re-applied under the solved B.
            var cWorldSolved = MbPose.Mul(bWorldSolved, MbPose.Mul(MbPose.Inverse(bWorld), cWorld));

            // The tip slerps toward the pinned tip's rotation, renormalized to keep errors from
            // building up.
            solvedA = new Pose(aWorldSolved.position, MbPose.Normalize(aWorldSolved.rotation));
            solvedB = new Pose(bWorldSolved.position, MbPose.Normalize(bWorldSolved.rotation));
            solvedC = new Pose(cWorldSolved.position,
                MbPose.Normalize(MbPose.SlerpSafe(cWorldSolved.rotation, tWorld.rotation, weight)));
        }

        /// <summary>The signed angle from vector a to vector b around the given axis, in radians.</summary>
        private static float SignedAngle(Vector3 axis, Vector3 a, Vector3 b)
        {
            return Mathf.Atan2(Vector3.Dot(axis, Vector3.Cross(a, b)), Vector3.Dot(a, b));
        }
    }
}
