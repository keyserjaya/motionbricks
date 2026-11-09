using UnityEngine;

namespace MotionBricks
{
    /// <summary>
    /// Places target poses in the world, ported from <c>place_target_poses</c> and
    /// <c>adjust_target_pose_speed</c> in <c>motionbricks/runtime/target_pose.py</c>.
    /// The Python runtime works in a Z-up world; Unity is Y-up, so the ground plane is XZ and the
    /// yaw turns around the Y axis.
    /// </summary>
    public static class SomaTargetPlacement
    {
        /// <summary>
        /// Returns the transform that moves a target to a ground position, turned to face a direction.
        /// </summary>
        public static Pose GroundPlacement(Vector3 groundPosition, Vector3 facingDirection)
        {
            var yaw = Mathf.Atan2(facingDirection.x, facingDirection.z);
            return new Pose(groundPosition, Quaternion.AngleAxis(yaw * Mathf.Rad2Deg, Vector3.up));
        }

        /// <summary>
        /// Places target poses at a ground position, turned to a facing direction. Only the root moves;
        /// the other joints keep their local transforms.
        /// </summary>
        /// <param name="source">Local joint transforms of the targets, laid out as [pose, joint].</param>
        /// <param name="destination">Array of the same shape that receives the placed poses.</param>
        public static void PlaceTargetPoses(
            Vector3 groundPosition,
            Vector3 facingDirection,
            Pose[] source,
            Pose[] destination,
            int numPoses,
            int numJoints)
        {
            var placement = GroundPlacement(groundPosition, facingDirection);
            for (var pose = 0; pose < numPoses; pose++)
            {
                for (var joint = 0; joint < numJoints; joint++)
                {
                    var value = source[pose * numJoints + joint];
                    destination[pose * numJoints + joint] = joint == 0
                        ? MbPose.Mul(placement, value)
                        : value;
                }
            }
        }

        /// <summary>
        /// Scales the root's ground motion to a desired mean speed, with poses sampled at 30 Hz. One
        /// scale applies to each pose's ground offset from the first pose. Height, rotations, and the
        /// other joints are kept. Targets that barely move stay unchanged.
        /// </summary>
        public static void AdjustTargetPoseSpeed(Pose[] targetPoses, int numPoses, int numJoints, float desiredSpeed)
        {
            if (numPoses < 2)
            {
                return;
            }

            var start = targetPoses[0].position;
            var end = targetPoses[(numPoses - 1) * numJoints].position;
            var scale = GroundSpeedScale(start, end, numPoses, desiredSpeed);
            for (var pose = 1; pose < numPoses; pose++)
            {
                var index = pose * numJoints;
                var position = targetPoses[index].position;
                targetPoses[index] = new Pose(
                    new Vector3(
                        start.x + (position.x - start.x) * scale,
                        position.y,
                        start.z + (position.z - start.z) * scale),
                    targetPoses[index].rotation);
            }
        }

        /// <summary>
        /// Returns the scale that brings the mean ground speed from start to end to the desired speed.
        /// The poses are 30 Hz apart. Targets that barely move keep a scale of 1.
        /// </summary>
        public static float GroundSpeedScale(Vector3 start, Vector3 end, int poseCount, float desiredSpeed)
        {
            var velocity = (end - start) * (MbContract.SampleRate / (float)(poseCount - 1));
            var sourceSpeed = new Vector2(velocity.x, velocity.z).magnitude;
            if (sourceSpeed <= 1e-6f)
            {
                return 1f;
            }

            return desiredSpeed / sourceSpeed;
        }

        /// <summary>
        /// Removes the clip's own placement and heading, so each generation can place the poses wherever
        /// the character is heading. The result starts at the ground origin and travels toward +Z,
        /// which is the direction <see cref="GroundPlacement"/> turns.
        /// </summary>
        /// <param name="clip">The clip to sample.</param>
        /// <param name="sampleTime">The clip time of the first pose of the window.</param>
        /// <param name="destination">Receives the canonical poses, laid out as [pose, joint].</param>
        /// <param name="numPoses">How many poses the window holds.</param>
        /// <param name="numJoints">Joints per pose.</param>
        /// <param name="desiredSpeed">Ground speed the canonical poses should have, in m/s.</param>
        public static void CanonicalizeTarget(
            SomaClip clip,
            float sampleTime,
            Pose[] destination,
            int numPoses,
            int numJoints,
            float desiredSpeed)
        {
            var scratch = new Pose[numJoints];
            for (var pose = 0; pose < numPoses; pose++)
            {
                clip.Sample(sampleTime + pose / MbContract.SampleRate, scratch, new Pose(Vector3.zero, Quaternion.identity));
                System.Array.Copy(scratch, 0, destination, pose * numJoints, numJoints);
            }

            // The clip's own placement and heading are removed. The origin holds the first root's
            // ground position, so the inverse transform maps it to the ground origin with its height.
            var start = destination[0].position;
            var end = destination[(numPoses - 1) * numJoints].position;
            float yaw;
            var travel = new Vector2(end.x - start.x, end.z - start.z);
            if (travel.magnitude > 0.03f)
            {
                yaw = Mathf.Atan2(travel.x, travel.y);
            }
            else
            {
                // A static window barely travels, so its travel yaw is measurement noise (a crouch
                // idle drifted 10 mm and came out rotated 88 degrees). The SOMA root's local +Y axis
                // is the body's forward direction — the skeleton's root frame is tilted — so take the
                // heading from the root rotation instead.
                var forward = destination[0].rotation * Vector3.up;
                yaw = Mathf.Atan2(forward.x, forward.z);
            }

            var origin = new Pose(
                new Vector3(start.x, 0f, start.z),
                Quaternion.AngleAxis(yaw * Mathf.Rad2Deg, Vector3.up));

            for (var pose = 0; pose < numPoses; pose++)
            {
                for (var joint = 0; joint < numJoints; joint++)
                {
                    var index = pose * numJoints + joint;
                    destination[index] = joint == 0
                        ? MbPose.Mul(MbPose.Inverse(origin), destination[index])
                        : destination[index];
                }
            }

            AdjustTargetPoseSpeed(destination, numPoses, numJoints, desiredSpeed);
        }
    }
}
